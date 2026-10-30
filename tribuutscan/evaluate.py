"""Measure a tracker on saved recordings and synthetic data; compare runs.

    .venv/bin/python -m tribuutscan.evaluate                       # all datasets, current tracker
    .venv/bin/python -m tribuutscan.evaluate tilt synth --label x  # some datasets
    .venv/bin/python -m tribuutscan.evaluate --against captures/eval/<run>/report.json

Writes captures/eval/<stamp>-<label>/report.json and 3-view renders (none for private datasets).
Metrics: tracked/lost/re-found frames, pose jumps, double-surface ratio, marker scatter, loop gap,
ms per frame; ATE/RPE against ground truth on synthetic data. (A keyframe residual and a split-half
Chamfer were tried and dropped: they read the same on the doubled and the good scan, because each
half / keyframe already contains the shifted frames.)
Distances rest on the estimated depth law and field of view (except synthetic, which uses them too).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import cloud as cl
from . import frames as fr
from . import markers as mk
from . import metrics
from . import synth
from .image import write_png

EVAL = fr.CAPTURES / "eval"
DATASETS = {  # name: (path or glob under captures/, notes, scan mode)
    "tilt": ("20261030-122746-scan/*-rec", "black part with markers, scanner tilted: the doubling case", "still"),
    "standing": ("20261030-123412-scan/*-rec", "black part, scanner held still", "still"),
    "kettle": ("20261030-094413-rec", "kettle, moved by hand", "turntable"),
    "hand": ("20261030-102704-rec", "a hand: deforms, robustness only", "free"),
    "face": ("20261030-101700-rec", "PRIVATE face scan: deforms, metrics only, never rendered", "free"),
    "synth": ("synth/box-tilt-rec", "synthetic box with markers, tilting path, ground truth", "still"),
    "synth7": ("synth/box-tilt-7fps-rec", "the same at ~7 fps (every 3rd frame), like a slow live worker", "still"),
    "synthhard": ("synth/box-hard-rec", "synthetic at 10 fps with a jerk and a stretch beyond the depth window",
                  "still"),
}
SYNTH = {"synth": dict(every=1), "synth7": dict(every=3), "synthhard": dict(every=2, hard=True)}
PRIVATE = {"face"}


def dataset_path(name):
    pattern = DATASETS[name][0]
    if name in SYNTH and not (fr.CAPTURES / pattern).exists():
        synth.generate(fr.CAPTURES / pattern, **SYNTH[name])
    hits = sorted(fr.CAPTURES.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{name}: no recording at captures/{pattern}")
    return hits[0]


def cached_markers(name, directory):
    """Marker detections per frame (full-resolution colour pixels), cached: detection is the slow part."""
    cache = EVAL / "cache" / f"{name}-markers.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    found = {i: [list(m) for m in mk.detect_half(f["color"])] for i, f in enumerate(fr.read_recording(directory))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(found))
    return found


TRACKERS = {  # name: Scanner options (the scan mode comes from the dataset)
    "current": {},
    "baseline": dict(clean_depth=False, gate_refind=False, motion_dt=False, track_with="icp", mode="free"),
    "p2": dict(track_with="icp"),  # frame-to-frame ICP with P1/P2 (gate, cleanup, scan modes)
    "frame": dict(target="frame"),  # joint solve against the previous frame
    "nomarkers": dict(track_with="jointnm"),  # joint solve against the model, without markers
    "markers": dict(track_with="markers"),  # markers alone (P3)
}


def make_scanner(tracker, preset, mode):
    options = dict(mode=mode, **TRACKERS[tracker]) if tracker in TRACKERS else None
    if options is None:
        raise ValueError(f"unknown tracker {tracker}; one of {list(TRACKERS)}")
    return cl.Scanner(preset, log=lambda m: None, **options)


def run(name, tracker="baseline", preset="small"):
    directory = dataset_path(name)
    found = cached_markers(name, directory)
    scanner = make_scanner(tracker, preset, DATASETS[name][2])
    sightings, elapsed, n = [], 0.0, 0
    for i, frame in enumerate(fr.read_recording(directory)):
        frame["markers2d"] = found.get(i, [])
        t0 = time.perf_counter()
        scanner.feed(frame, i)
        elapsed += time.perf_counter() - t0
        n += 1
        index, status, pose = scanner.track_log[-1]
        if index == i and pose is not None:
            for loc in mk.locate(frame["depth"], found.get(i, [])):
                if loc is not None:
                    sightings.append(loc[0] @ pose[:3, :3].T + pose[:3, 3] * 1000)
    scanner.trim_table()
    xyz, rgb, seen, normals = scanner.cloud.points(cl.MIN_SEEN, with_normals=True)
    tracked = [(i, p) for i, s, p in scanner.track_log if p is not None]
    centre = xyz.mean(0) if len(xyz) else np.zeros(3)
    gap = metrics.loop_gap(tracked, centre)
    result = {
        "dataset": name, "notes": DATASETS[name][1], "tracker": tracker, "preset": preset,
        "mode": scanner.scene.mode, "plane_frames": scanner.stats["plane_frames"],
        "frames": n, "tracked": len(tracked), "merged": scanner.stats["merged"],
        "lost": scanner.stats["lost"], "relocked": scanner.stats["relocked"],
        "ms_per_frame": round(1000 * elapsed / max(n, 1), 1),
        "points": int(len(xyz)),
        "pose_jumps": metrics.pose_jumps(tracked, centre),
        "flat_frames": scanner.stats.get("flat", 0),
        "double_surface_ratio": round(metrics.double_surface_ratio(xyz, normals), 4),
        "marker_scatter": metrics.marker_scatter(sightings),
        "loop_gap": {"mm": round(gap[0], 1), "deg": round(gap[1], 1)},
    }
    if name in SYNTH:
        truth = synth.read_truth(directory)
        result["ground_truth"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in metrics.ate_rpe(dict(tracked), truth).items()}
    return result, (xyz, rgb, normals)


def render_views(xyz, normals, path, size=360):
    """Grey shaded front / side / top views of a cloud, side by side, as PNG."""
    if not len(xyz):
        return
    v = xyz - xyz.mean(0)

    def rot(axis, deg):
        a = np.radians(deg)
        k = np.array(axis, float)
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.cos(a) * np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * np.outer(k, k)

    views = []
    for r in (cl.FLIP, cl.FLIP @ rot([0, 1, 0], 90), cl.FLIP @ rot([1, 0, 0], -90)):
        p, nn = v @ r.T, normals @ r.T
        scale = (size - 20) / max(np.ptp(p[:, :2], 0).max(), 1e-9)
        xy = ((p[:, :2] - p[:, :2].min(0)) * scale + 10).astype(int)
        img = np.full((size, size, 3), 30, np.uint8)
        order = np.argsort(p[:, 2])
        shade = (60 + 190 * np.abs(nn[order, 2])).astype(np.uint8)
        img[size - 1 - np.clip(xy[order, 1], 0, size - 1), np.clip(xy[order, 0], 0, size - 1)] = shade[:, None]
        views.append(img)
    write_png(path, np.concatenate(views, 1))


def show(results, against=None):
    """Print one line per dataset; with `against`, show before -> after."""
    old = {r["dataset"]: r for r in against} if against else {}

    def cell(r, key):
        if r is None:
            return "-"
        v = r
        for k in key.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, list):
            v = len(v)
        return "-" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))

    cols = [("tracked", "tracked"), ("lost", "lost"), ("relocked", "re-found"), ("pose_jumps", "jumps"),
            ("double_surface_ratio", "double"), ("marker_scatter.scatter_mm", "mk mm"),
            ("marker_scatter.markers", "mk n"), ("loop_gap.mm", "gap mm"),
            ("ground_truth.ate_mm", "ATE mm"), ("ground_truth.rpe_mm", "RPE mm"), ("ms_per_frame", "ms")]
    print(f"{'dataset':9s} " + " ".join(f"{h:>14s}" for _, h in cols))
    for r in results:
        o = old.get(r["dataset"])
        cells = [(f"{cell(o, k)}->{cell(r, k)}" if o else cell(r, k)) for k, _ in cols]
        print(f"{r['dataset']:9s} " + " ".join(f"{c:>14s}" for c in cells))


def main():
    args = sys.argv[1:]
    opt = lambda flag, default=None: args[args.index(flag) + 1] if flag in args else default
    tracker, preset, label = opt("--tracker", "current"), opt("--preset", "small"), opt("--label", "run")
    against = json.loads(Path(opt("--against")).read_text()) if "--against" in args else None
    skip = {opt(f) for f in ("--tracker", "--preset", "--label", "--against")}
    names = [a for a in args if not a.startswith("--") and a not in skip] or list(DATASETS)
    out = EVAL / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        t0 = time.time()
        result, (xyz, rgb, normals) = run(name, tracker, preset)
        results.append(result)
        if name not in PRIVATE:
            render_views(xyz, normals, out / f"{name}.png")
        print(f"  {name}: done in {time.time() - t0:.0f} s", flush=True)
    (out / "report.json").write_text(json.dumps(results, indent=2))
    show(results, against)
    print(f"-> {out / 'report.json'}")


if __name__ == "__main__":
    main()
