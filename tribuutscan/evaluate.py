"""Measure a tracker on saved recordings and synthetic data; compare runs.

    .venv/bin/python -m tribuutscan.evaluate                       # all datasets, current tracker
    .venv/bin/python -m tribuutscan.evaluate tilt synth --label x  # some datasets
    .venv/bin/python -m tribuutscan.evaluate --against captures/eval/<run>/report.json
    .venv/bin/python -m tribuutscan.evaluate synth synthplate --mesh               # also fuse a mesh (fusion.py)

Writes captures/eval/<stamp>-<label>/report.json and 3-view renders (none for private datasets).
Metrics: tracked/lost/re-found frames, pose jumps, double-surface ratio, marker scatter, loop gap,
ms per frame; on synthetic data ATE/RPE against the true poses and the cloud's distance to the true
surface (in place, so tracking drift counts; points within 5 mm are on the object). With --mesh, the
merged frames are also fused (fusion.py) at the tracked poses and, on synthetic sets, at the TRUE
poses (fusion alone: the table cut by height), each scored against the true surface; the plate's two
faces must each be covered within 0.5 mm. (A keyframe residual and a split-half
Chamfer were tried and dropped: they read the same on the doubled and the good scan, because each
half / keyframe already contains the shifted frames.)
Real recordings use calib.CAMERA (board calibration, else the scanner's factory one); synthetic sets
always use the estimated camera they were rendered with. The first datasets (tilt, standing, kettle,
hand, face) were deleted on 2026-10-30; their numbers stay in CLAUDE.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import calib
from . import cloud as cl
from . import frames as fr
from . import markers as mk
from . import metrics
from . import synth
from .calib import CAMERA
from .image import write_png

EVAL = fr.CAPTURES / "eval"
DATASETS = {  # name: (path or glob under captures/, notes, scan mode)
    "markers": ("20261030-194411-scan/*-rec", "the marker part, hand-held, 581 frames: the first recording with "
                "the calibration and frames paired by capture time", "still"),
    "synth": ("synth/box-tilt-rec", "synthetic box with markers, tilting path, ground truth", "still"),
    "synth7": ("synth/box-tilt-7fps-rec", "the same at ~7 fps (every 3rd frame), like a slow live worker", "still"),
    "synthhard": ("synth/box-hard-rec", "synthetic at 10 fps with a jerk and a stretch beyond the depth window",
                  "still"),
    "synthfp": ("synth/box-pattern-rec", "synthetic box, 14-bit, a fixed per-pixel depth error like the scanner's "
                "(0.14 mm RMS at 20 cm, growing with z^2; its pattern an estimate) and 0.05 mm frame noise", "still"),
    "synthplate": ("synth/plate-rec", "synthetic 1.8 mm plate standing on the table, seen from both sides; "
                   "14-bit with the fixed error of synthfp", "still"),
}
SENSOR = dict(pattern_mm=0.144, noise_mm=0.05, bits=14)  # synth.py: closer to the real sensor
SYNTH = {"synth": dict(every=1), "synth7": dict(every=3), "synthhard": dict(every=2, hard=True),
         "synthfp": dict(every=1, **SENSOR), "synthplate": dict(every=1, kind="plate", **SENSOR)}
PRIVATE = set()  # datasets never rendered (none at the moment)


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
    "nofeatures": dict(use_features=False),  # joint solve without the ORB starting pose (P4 part 1)
    "markers": dict(track_with="markers"),  # markers alone (P3)
    "inline": dict(threaded_merge=False),  # merging in the tracking thread
}


def make_scanner(tracker, preset, mode):
    options = dict(mode=mode, **TRACKERS[tracker]) if tracker in TRACKERS else None
    if options is None:
        raise ValueError(f"unknown tracker {tracker}; one of {list(TRACKERS)}")
    return cl.Scanner(preset, log=lambda m: None, **options)


def run(name, tracker="baseline", preset="small", live_markers=False, optimise_after=False, mesh=False):
    """live_markers: let the tracker detect markers itself (markers.detect_fast, as live) instead of
    being handed the cached reference detections. Synthetic sets run with the estimated camera they
    were rendered with."""
    calibrated = CAMERA.spec
    if name in SYNTH:
        calib.use(calib.ESTIMATES)
    try:
        return _run(name, tracker, preset, live_markers, optimise_after, mesh)
    finally:
        calib.use(calibrated)


def surface_check(directory, vertices, triangles):
    """A synthetic set's mesh against its true surface (mm); for the plate, each face's share within 0.5 mm."""
    import open3d as o3d
    d = synth.truth_distance(directory, vertices)
    on = d < 5.0
    out = {"median": round(float(np.median(d[on])), 3) if on.any() else None,
           "p90": round(float(np.percentile(d[on], 90)), 3) if on.any() else None,
           "on_object": round(float(on.mean()), 3) if len(d) else None}
    if json.loads((Path(directory) / "info.json").read_text()).get("kind") == "plate" and len(triangles):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(triangles.astype(np.uint32)))
        out["faces_within_0.5mm"] = [round(float((scene.compute_distance(o3d.core.Tensor(f.astype(np.float32))).numpy()
                                                  < 0.5).mean()), 3) for f in synth.plate_faces(directory)]
    return out


def mesh_result(name, directory, scanner):
    """The merged frames fused at their tracked poses (and on synthetic sets at the true ones)."""
    from . import fusion
    if not scanner.sources:
        scanner.add_source(Path(directory).resolve(), 0)
    t = time.perf_counter()
    engine, ms = fusion.fuse(fusion.scan_frames(scanner, "merged"), log=lambda m: None)
    vertices, triangles, _ = fusion.extract(engine, scanner.output_mask)
    out = {"frames": engine.frames, "triangles": int(len(triangles)), "s": round(time.perf_counter() - t, 1),
           "integrate_ms": ms["integrate"]}
    if name in SYNTH:
        out["tracked"] = surface_check(directory, vertices, triangles)
        truth = synth.read_truth(directory)
        engine, _ = fusion.fuse(((f, truth[i], None) for i, f in enumerate(fr.read_recording(directory)) if i in truth),
                                log=lambda m: None)
        vertices, triangles, _ = fusion.extract(engine, lambda xyz: synth.height_above_table(directory, xyz) > 1.0)
        out["true_poses"] = surface_check(directory, vertices, triangles)
    return out


def _run(name, tracker, preset, live_markers, optimise_after, mesh=False):
    directory = dataset_path(name)
    found = cached_markers(name, directory)
    scanner = make_scanner(tracker, preset, DATASETS[name][2])
    sightings, elapsed, n = [], 0.0, 0
    for i, frame in enumerate(fr.read_recording(directory)):
        if not live_markers:
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
    scanner.flush()
    opt_report = None
    if optimise_after:
        from . import optimise
        opt_report = optimise.optimise(scanner, log=lambda m: None)
    scanner.trim_table()
    xyz, rgb, seen, normals = scanner.output_cloud().points(cl.MIN_SEEN, with_normals=True)  # as saved
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
        "feature_starts": scanner.feature_starts,
        "double_surface_ratio": round(metrics.double_surface_ratio(xyz, normals), 4),
        "marker_scatter": metrics.marker_scatter(sightings),
        "loop_gap": {"mm": round(gap[0], 1), "deg": round(gap[1], 1)},
        "keyframe_marker_scatter": metrics.marker_scatter(
            [m @ p[:3, :3].T + p[:3, 3] * 1000 for _, p, _, _, mk in scanner.keyframes for m in mk]),
        "optimise": opt_report,
    }
    if mesh:
        result["mesh"] = mesh_result(name, directory, scanner)
    if name in SYNTH:
        truth = synth.read_truth(directory)
        result["ground_truth"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in metrics.ate_rpe(dict(tracked), truth).items()}
        result["keyframe_ground_truth"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                           for k, v in metrics.ate_rpe(dict(scanner.poses), truth).items()}
        d = synth.truth_distance(directory, xyz)
        on = d < 5.0
        result["truth_mm"] = {"median": round(float(np.median(d[on])), 3) if on.any() else None,
                              "p90": round(float(np.percentile(d[on], 90)), 3) if on.any() else None,
                              "on_object": round(float(on.mean()), 3) if len(d) else None}
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

    cols = [("tracked", "tracked"), ("relocked", "re-found"), ("pose_jumps", "jumps"),
            ("double_surface_ratio", "double"), ("marker_scatter.scatter_mm", "mk mm"),
            ("keyframe_marker_scatter.scatter_mm", "kf mk mm"), ("loop_gap.mm", "gap mm"),
            ("ground_truth.ate_mm", "ATE mm"), ("keyframe_ground_truth.ate_mm", "kf ATE"), ("truth_mm.median", "truth mm"),
            ("mesh.tracked.median", "mesh mm"), ("mesh.true_poses.median", "fused mm"),
            ("optimise.loops_kept", "loops"), ("ms_per_frame", "ms")]
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
        result, (xyz, rgb, normals) = run(name, tracker, preset, "--live-markers" in args, "--optimise" in args,
                                          "--mesh" in args)
        results.append(result)
        if name not in PRIVATE:
            render_views(xyz, normals, out / f"{name}.png")
        print(f"  {name}: done in {time.time() - t0:.0f} s", flush=True)
    (out / "report.json").write_text(json.dumps(results, indent=2))
    show(results, against)
    print(f"-> {out / 'report.json'}")


if __name__ == "__main__":
    main()
