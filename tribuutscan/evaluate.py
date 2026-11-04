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
    "hand": ("20261102-081622-scan/*-rec", "the first live GPU scan's raw frames: a hand model (~22 cm) standing on "
             "a tiled floor with cardboard and cables, long range at 35-50 cm, ~9 fps, no markers", "still"),
    "synth": ("synth/box-tilt-rec", "synthetic box with markers, tilting path, ground truth", "still"),
    "synth7": ("synth/box-tilt-7fps-rec", "the same at ~7 fps (every 3rd frame), like a slow live worker", "still"),
    "synthhard": ("synth/box-hard-rec", "synthetic at 10 fps with a jerk and a stretch beyond the depth window",
                  "still"),
    "synthfp": ("synth/box-pattern-rec", "synthetic box, 14-bit, a fixed per-pixel depth error like the scanner's "
                "(0.14 mm RMS at 20 cm, growing with z^2; its pattern an estimate) and 0.05 mm frame noise", "still"),
    "synthplate": ("synth/plate-rec", "synthetic 1.8 mm plate standing on the table, seen from both sides; "
                   "14-bit with the fixed error of synthfp", "still"),
    "platform": ("20261102-170723-scan/*-rec", "the hand on a platform turned by hand (briskly, ~1 turn), the "
                 "scanner standing, the room still behind; long range ~30 cm, no markers; the white platform top "
                 "gives no depth. Live: lost at the edge-on view, re-found from the room, 9.2% doubled", "still"),
    "platform2": ("20261102-174213-scan/*-rec", "the same, turned slowly (~123 degrees in 94 s, 1109 frames); live: "
                  "all tracked, FLAT near the edge-on view", "still"),
    "synthturn": ("synth/turntable-rec", "synthetic: the test block on a platform turning ~400 degrees at 3-23 "
                  "deg/s, the scanner standing at ~35 cm in a still room (table, wall, boxes), long range, 10 fps, "
                  "14-bit with the fixed error of synthfp; the platform gives no depth", "still"),
}
GAPS = {  # name: (dataset, frames left out): a stretch skipped as if tracking had been lost while the scanner moved
    "markersgap": ("markers", range(250, 330)),
    "handgap": ("hand", range(150, 250)),
    "synthturngap": ("synthturn", range(150, 175)),  # 2.5 s, the block turned ~40 degrees on
}
DATASETS.update({name: (DATASETS[base][0], f"{base} with frames {gap.start}-{gap.stop - 1} left out: tracking "
                        "must find its place again after a jump", DATASETS[base][2]) for name, (base, gap) in GAPS.items()})
SENSOR = dict(pattern_mm=0.144, noise_mm=0.05, bits=14)  # synth.py: closer to the real sensor
SYNTH = {"synth": dict(every=1), "synth7": dict(every=3), "synthhard": dict(every=2, hard=True),
         "synthfp": dict(every=1, **SENSOR), "synthplate": dict(every=1, kind="plate", **SENSOR),
         "synthturn": dict(n_frames=600, every=2, kind="turntable", **SENSOR)}
PRIVATE = set()  # datasets never rendered (none at the moment)


def base_of(name):
    """The dataset a gap variant leaves frames out of (the name itself otherwise)."""
    return GAPS.get(name, (name,))[0]


def dataset_path(name):
    pattern = DATASETS[name][0]
    if base_of(name) in SYNTH and not (fr.CAPTURES / pattern).exists():
        synth.generate(fr.CAPTURES / pattern, **SYNTH[base_of(name)])
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
    "current": {},  # the joint solve on the CPU (the fallback without a usable GPU, or with --force-cpu)
    "gpu": dict(track_with="gpu"),  # the joint solve against a ray cast of a GPU volume (gputrack.py, Phase 3)
    "gpu-proc": dict(track_with="gpu", merge_process=True),  # merged in a process of its own (the live worker's way)
}  # (the earlier variants, baseline to gpu-noverify, went in Phase 6: docs/history.md has their numbers)


def make_scanner(tracker, preset, mode):
    options = dict(mode=mode, **TRACKERS[tracker]) if tracker in TRACKERS else None
    if options is None:
        raise ValueError(f"unknown tracker {tracker}; one of {list(TRACKERS)}")
    return cl.Scanner(preset, log=lambda m: None, **options)


def run(name, tracker="current", preset="small", live_markers=False, optimise_after=False, mesh=False):
    """live_markers: let the tracker detect markers itself (markers.detect_fast, as live) instead of
    being handed the cached reference detections. Synthetic sets run with the estimated camera they
    were rendered with."""
    calibrated = CAMERA.spec
    if base_of(name) in SYNTH:
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


def coverage(directory, vertices, triangles, within_mm=1.0, n=200000):
    """Share of a synthetic set's true surface (sampled evenly) within `within_mm` of a mesh: what the mesh
    covers of the object, the faces it never saw included."""
    import open3d as o3d
    truth = o3d.io.read_triangle_mesh(str(Path(directory) / "true_surface.ply"))
    pts = np.asarray(truth.sample_points_uniformly(n).points) @ np.diag([1000.0, -1000.0, -1000.0])  # scan world mm
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(triangles, np.uint32)))
    return round(float((scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy() < within_mm).mean()), 3)


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
    if base_of(name) in SYNTH:
        out["tracked"] = surface_check(directory, vertices, triangles)
        out["tracked"]["covered_1mm"] = coverage(directory, vertices, triangles)
        from . import solid  # the printable solid (fusion.make_mesh's STL) against the truth too
        cv, ct, _, closing = solid.close(vertices, triangles, None, fusion.printing_planes(scanner, vertices))
        out["closed"] = dict(surface_check(directory, cv, ct), covered_1mm=coverage(directory, cv, ct),
                             edges_closed=closing["edges_closed"], added_share=closing["added_share"])
        truth = synth.read_truth(directory)
        turning = json.loads((Path(directory) / "info.json").read_text()).get("kind") == "turntable"
        # (whole frames, cut at extraction; but a turntable's still room smears round the object at its poses and
        # overran Open3D's voxel blocks (a segfault): there only the object, by the tracked run's labels)
        engine, _ = fusion.fuse(((f, truth[i], scanner.labels.get(i) if turning else None)
                                 for i, f in enumerate(fr.read_recording(directory))
                                 if i in truth and (not turning or i in scanner.labels)),
                                log=lambda m: None)
        vertices, triangles, _ = fusion.extract(engine, lambda xyz: synth.height_above_table(directory, xyz) > 1.0)
        out["true_poses"] = surface_check(directory, vertices, triangles)
    return out


def _run(name, tracker, preset, live_markers, optimise_after, mesh=False):
    directory = dataset_path(name)
    base, gap = GAPS.get(name, (name, range(0)))
    found = cached_markers(base, directory)
    scanner = make_scanner(tracker, preset, DATASETS[name][2])
    sightings, elapsed, n = [], 0.0, 0
    for i, frame in enumerate(fr.read_recording(directory)):
        if i in gap:
            continue
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
    scanner.close()
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
        "gpu_ms": scanner.stats.get("gpu_ms"),
        "points": int(len(xyz)),
        "region_mm": None if scanner.scene.radius is None else round(float(scanner.scene.radius)),
        "pose_jumps": metrics.pose_jumps(tracked, centre),
        "flat_frames": scanner.stats.get("flat", 0),
        "object_turns": scanner.object_turns, "room_votes": dict(scanner.room_votes),
        "shape_refinds": scanner.stats.get("shape_refinds", 0),
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
    if base_of(name) in SYNTH:
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
            ("mesh.tracked.covered_1mm", "mesh cov"), ("mesh.closed.median", "closed mm"),
            ("mesh.closed.covered_1mm", "closed cov"),
            ("optimise.loops_kept", "loops"), ("optimise.largest_correction_mm", "corr mm"), ("optimise.largest_correction_at_object_mm", "corr obj"), ("region_mm", "region mm"),
            ("ms_per_frame", "ms")]
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
