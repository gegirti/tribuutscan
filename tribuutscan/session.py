"""Continue a saved scan, or merge two scans into one. Open3D + OpenCV (.venv).

    .venv/bin/python -m tribuutscan.session merge captures/A-scan captures/B-scan captures/AB-scan

Continuing: the first new frames must find themselves in the saved scan before tracking can go on
("re-finding"): first by matching their markers to the saved marker map (needs no prediction), else
by FPFH shape features + RANSAC against the saved cloud; either is checked with the joint solve.
Merging: the second scan's cloud is registered onto the first the same way (markers first), its
keyframes and merged frames are moved over, and Optimise ties the two together.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from . import track

reg = o3d.pipelines.registration
FPFH_VOXEL_M = 0.003
ACCEPT_FITNESS, ACCEPT_RMSE_MM = 0.3, 2.0


def _fpfh(points_m, voxel=FPFH_VOXEL_M):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_m)).voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    feature = reg.compute_fpfh_feature(cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=100))
    return cloud, feature


def shape_pose(source_mm, target_mm, seed=0):
    """Coarse transform (4x4, metres) placing source points on target points by shape alone, or None."""
    if len(source_mm) < 200 or len(target_mm) < 200:
        return None
    o3d.utility.random.seed(seed)  # RANSAC: the same answer on a replay
    src, src_f = _fpfh(source_mm / 1000)
    tgt, tgt_f = _fpfh(target_mm / 1000)
    d = FPFH_VOXEL_M * 1.5
    result = reg.registration_ransac_based_on_feature_matching(
        src, tgt, src_f, tgt_f, True, d, reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(d)],
        reg.RANSACConvergenceCriteria(100000, 0.999))
    return result.transformation if result.fitness > 0.1 else None


def accept(info):
    return info["fitness"] >= ACCEPT_FITNESS and info["rmse_mm"] is not None and info["rmse_mm"] <= ACCEPT_RMSE_MM


def refind(scanner, xyz_cam, marker_pts, levels):
    """Camera -> world pose of a frame in the scanner's saved world (markers, then shape), or None."""
    if not scanner.model_ready():
        return None
    starts = []
    if len(marker_pts) >= 3 and len(scanner.marker_map):
        found = scanner.marker_map.match_global(marker_pts)
        if found is not None:
            starts.append(found)
    with scanner.cloud_lock:
        model, _, _ = scanner.cloud.points(min_seen=2)
    found = shape_pose(xyz_cam, model)
    if found is not None:
        starts.append(found)
    for start in starts:
        pose, info = track.solve(levels, scanner.model_levels, start)
        if accept(info):
            return pose
    return None


def merge(dir_a, dir_b, out, log=print):
    """Merge scan B into scan A's world and save the result (optimised) into `out`."""
    from . import cloud as cl
    from . import optimise
    a = cl.Scanner.restore(dir_a, log=log)
    b = cl.Scanner.restore(dir_b, log=log)
    t = None
    if len(b.marker_map) >= 3 and len(a.marker_map) >= 3:
        found = a.marker_map.match_global(b.marker_map.pos[np.argsort(-b.marker_map.count)])  # best first
        if found is not None:
            t = found
    pa, _, _ = a.cloud.points(min_seen=2)
    pb, _, _ = b.cloud.points(min_seen=2)
    if t is None:
        t = shape_pose(pb, pa)
    if t is None:
        raise RuntimeError("could not place the second scan on the first (no shared markers or shape)")
    t, info = track.solve(track.levels(pb / 1000), track.levels(pa / 1000), t)
    if not accept(info):
        raise RuntimeError(f"the two scans do not line up well enough ({info})")
    offset = max(i for i, _ in a.poses) + 1000  # keep frame indices apart
    a.keyframes += [(i + offset, t @ p, *rest) for i, p, *rest in b.keyframes]
    a.poses += [(i + offset, t @ p) for i, p in b.poses]
    a.merged_data.update({i + offset: d for i, d in b.merged_data.items()})
    a.remerge({})
    report = optimise.optimise(a, log)
    Path(out).mkdir(parents=True, exist_ok=True)
    a.save(Path(out))
    (Path(out) / "optimise.json").write_text(json.dumps(report, indent=2))
    log(f"merged: {len(a.poses)} keyframes, {report['loops_kept']} loop closures")
    return a


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "merge":
        merge(*sys.argv[2:])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
