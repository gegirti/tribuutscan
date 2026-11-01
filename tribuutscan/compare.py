"""Score a scan against a reference mesh of the same part (e.g. CrealityScan's result): align, measure.

    .venv/bin/python -m tribuutscan.compare captures/<stamp>-scan [reference.ply] [--seeds 12]
    .venv/bin/python -m tribuutscan.compare captures/<stamp>-scan/mesh.ply --transform captures/<stamp>-scan/compare.json

The reference defaults to captures/reference/part/mesh.ply. A scan directory means its cloud.ply; a
mesh (any .ply with triangles) is sampled evenly over its surface, about one point per 0.25 mm2 (the
0.5 mm cloud's density). Ours is placed on the reference by shape (session.shape_pose: FPFH + RANSAC,
best of --seeds seeds; 12 for before / after comparisons, as the result varies between point sets),
then refined by point-to-plane ICP with a robust kernel (our surroundings are not in the reference).
--transform reuses another comparison's placement instead (clouds and meshes from the same tracking
should be compared in the same place). Reported:
- accuracy: distance from our points to the reference surface, over the points on the part (within
  ON_PART_MM); the others are surroundings
- completeness: share of the reference surface with one of our points within 1 / 2 mm
- size: the reference's main axes (1-99% extents), measured on both
Writes compare.ply (our points coloured by distance: blue 0 mm, red >= 2 mm, grey off the part) and
compare.json next to the scan (compare-<name>.json / .ply for another file, e.g. compare-mesh.json).
The reference is a scan too, not ground truth: sub-millimetre differences say little about which one
is right.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from . import session

REFERENCE = Path(__file__).resolve().parent.parent / "captures" / "reference" / "part" / "mesh.ply"
ON_PART_MM = 5.0
RED_MM = 2.0
SEEDS = 5
MESH_MM2_PER_POINT = 0.25
reg = o3d.pipelines.registration


def millimetres(points):
    """Points in mm: clouds smaller than 5 units across are taken to be in metres (cloud.ply is)."""
    return points * 1000 if np.ptp(points, 0).max() < 5 else points


def align(ours, ref, seeds=SEEDS):
    """4x4 (mm) placing our points on the reference cloud (with normals), and the ICP result."""
    best = None
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ours))
    for seed in range(seeds):
        t = session.shape_pose(ours, np.asarray(ref.points), seed)
        if t is None:
            continue
        t = t.copy()
        t[:3, 3] *= 1000  # shape_pose works in metres
        for dist in (5.0, 2.0, 1.0):
            result = reg.registration_icp(src, ref, dist, t,
                                          reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=dist / 2)))
            t = result.transformation
        if best is None or result.fitness > best[1].fitness:
            best = (t, result)
    return best


def axes_extents(points, axes):
    p = (points - points.mean(0)) @ axes
    return np.percentile(p, 99, 0) - np.percentile(p, 1, 0)


def our_points(path):
    """(N, 3) mm: a cloud's points, or points sampled evenly over a mesh's surface."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles):
        mesh.vertices = o3d.utility.Vector3dVector(millimetres(np.asarray(mesh.vertices)))
        n = int(np.clip(mesh.get_surface_area() / MESH_MM2_PER_POINT, 10_000, 3_000_000))
        o3d.utility.random.seed(0)
        return np.asarray(mesh.sample_points_uniformly(n).points)
    return millimetres(np.asarray(o3d.io.read_point_cloud(str(path)).points))


def compare(path, ref_path=REFERENCE, seeds=SEEDS, transform=None):
    """Report (dict) of a cloud or mesh against the reference mesh; also returns our placed points and
    their distances (mm) for the coloured file. `transform`: a 4x4 (mm) to use instead of aligning."""
    ours = our_points(path)
    mesh = o3d.io.read_triangle_mesh(str(ref_path))
    mesh.vertices = o3d.utility.Vector3dVector(millimetres(np.asarray(mesh.vertices)))
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(0)
    ref = mesh.sample_points_uniformly(300_000)

    if transform is None:
        found = align(ours, ref, seeds)
        if found is None:
            raise RuntimeError("could not place the scan on the reference")
        t, icp = found
    else:
        t = np.asarray(transform, float)
        icp = reg.evaluate_registration(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ours)), ref, 1.0, t)
    placed = ours @ t[:3, :3].T + t[:3, 3]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d = scene.compute_distance(o3d.core.Tensor(placed.astype(np.float32))).numpy()
    on = d < ON_PART_MM
    near = np.asarray(ref.compute_point_cloud_distance(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(placed))))
    ref_pts = np.asarray(ref.points)
    axes = np.linalg.eigh(np.cov((ref_pts - ref_pts.mean(0)).T))[1][:, ::-1]
    report = {
        "scan": str(path), "reference": str(ref_path), "seeds": seeds if transform is None else None,
        "placement": "aligned" if transform is None else "given",
        "icp_fitness_1mm": round(icp.fitness, 3), "icp_rmse_mm": round(icp.inlier_rmse, 3),
        "our_points": int(len(placed)), "on_part_share": round(float(on.mean()), 3),
        "accuracy_mm": {"median": round(float(np.median(d[on])), 3), "mean": round(float(d[on].mean()), 3),
                        "p90": round(float(np.percentile(d[on], 90)), 3)},
        "completeness": {"within_1mm": round(float((near < 1).mean()), 3),
                         "within_2mm": round(float((near < 2).mean()), 3)},
        "size_mm": {"reference": axes_extents(ref_pts, axes).round(1).tolist(),
                    "ours_on_part": axes_extents(placed[on], axes).round(1).tolist()},
        "transform_mm": t.round(6).tolist(),
    }
    return report, placed, d


def write_coloured(path, placed, d):
    """Our points coloured by distance (blue 0 mm, red >= RED_MM, grey off the part), in metres."""
    on = d < ON_PART_MM
    rgb = np.full((len(placed), 3), 0.6)
    f = np.clip(d[on] / RED_MM, 0, 1)[:, None]
    rgb[on] = (1 - f) * np.array([0.1, 0.3, 1.0]) + f * np.array([1.0, 0.1, 0.1])
    colored = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(placed / 1000))  # metres, like cloud.ply
    colored.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(str(path), colored)


def main():
    args = sys.argv[1:]
    opt = lambda name: args[args.index(name) + 1] if name in args else None
    values = {opt("--seeds"), opt("--transform")}
    paths = [Path(a) for a in args if not a.startswith("--") and a not in values]
    if not paths:
        sys.exit(__doc__)
    scan = paths[0]
    ref_path = paths[1] if len(paths) > 1 else REFERENCE
    path = scan / "cloud.ply" if scan.is_dir() else scan
    out = path.parent
    name = "compare" if path.name == "cloud.ply" else f"compare-{path.stem}"
    transform = json.loads(Path(opt("--transform")).read_text())["transform_mm"] if opt("--transform") else None
    try:
        report, placed, d = compare(path, ref_path, int(opt("--seeds") or SEEDS), transform)
    except RuntimeError as e:
        sys.exit(str(e))
    print(json.dumps({k: v for k, v in report.items() if k != "transform_mm"}, indent=2))
    (out / f"{name}.json").write_text(json.dumps(report, indent=2))
    write_coloured(out / f"{name}.ply", placed, d)
    print(f"-> {out / (name + '.json')}, {out / (name + '.ply')} (blue 0 mm, red >= {RED_MM:.0f} mm, grey off the part)")


if __name__ == "__main__":
    main()
