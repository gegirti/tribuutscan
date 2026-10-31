"""Score a scan against a reference mesh of the same part (e.g. CrealityScan's result): align, measure.

    .venv/bin/python -m tribuutscan.compare captures/<stamp>-scan [reference.ply]

The reference defaults to captures/reference/part/mesh.ply. Our cloud is placed on it by shape
(session.shape_pose: FPFH + RANSAC, best of a few seeds), then refined by point-to-plane ICP with a
robust kernel (our surroundings are not in the reference). Reported:
- accuracy: distance from our points to the reference surface, over the points on the part (within
  ON_PART_MM); the others are surroundings
- completeness: share of the reference surface with one of our points within 1 / 2 mm
- size: the reference's main axes (1-99% extents), measured on both
Writes compare.ply (our points coloured by distance: blue 0 mm, red >= 2 mm, grey off the part) and
compare.json next to the scan. The reference is a scan too, not ground truth: sub-millimetre
differences say little about which one is right.
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
reg = o3d.pipelines.registration


def millimetres(points):
    """Points in mm: clouds smaller than 5 units across are taken to be in metres (cloud.ply is)."""
    return points * 1000 if np.ptp(points, 0).max() < 5 else points


def align(ours, ref):
    """4x4 (mm) placing our points on the reference cloud (with normals), and the ICP result."""
    best = None
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ours))
    for seed in range(SEEDS):
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


def main():
    scan = Path(sys.argv[1])
    ref_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REFERENCE
    cloud_file = scan / "cloud.ply" if scan.is_dir() else scan
    out = scan if scan.is_dir() else scan.parent
    ours = millimetres(np.asarray(o3d.io.read_point_cloud(str(cloud_file)).points))
    mesh = o3d.io.read_triangle_mesh(str(ref_path))
    mesh.vertices = o3d.utility.Vector3dVector(millimetres(np.asarray(mesh.vertices)))
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(0)
    ref = mesh.sample_points_uniformly(300_000)

    found = align(ours, ref)
    if found is None:
        sys.exit("could not place the scan on the reference")
    t, icp = found
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
        "scan": str(cloud_file), "reference": str(ref_path),
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
    print(json.dumps({k: v for k, v in report.items() if k != "transform_mm"}, indent=2))
    (out / "compare.json").write_text(json.dumps(report, indent=2))

    rgb = np.full((len(placed), 3), 0.6)
    f = np.clip(d[on] / RED_MM, 0, 1)[:, None]
    rgb[on] = (1 - f) * np.array([0.1, 0.3, 1.0]) + f * np.array([1.0, 0.1, 0.1])
    colored = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(placed / 1000))  # metres, like cloud.ply
    colored.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(str(out / "compare.ply"), colored)
    print(f"-> {out / 'compare.json'}, {out / 'compare.ply'} (blue 0 mm, red >= {RED_MM:.0f} mm, grey off the part)")


if __name__ == "__main__":
    main()
