"""Optimise a finished scan: find loop closures between keyframes, correct all poses together in a
pose graph, and merge the cloud again with the corrected poses. Open3D + OpenCV (.venv).

    .venv/bin/python -m tribuutscan.optimise captures/<stamp>-scan   # (sessions saved by live scans)

Tracking runs frame after frame, so small errors add up; when the scan comes back to a place it has
seen, a loop closure says where it really is, and the pose graph spreads the correction over every
keyframe. Candidates come from markers seen in both keyframes, ORB feature matches, and keyframes
whose poses are close; each is checked with the joint solve (track.solve) before it is used, and
the graph's line process drops loops that disagree with the rest (preference 2.0: on a synthetic
loop it kept the true closure and pruned a 30 mm false one; at 1.0 it pruned both).
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from . import track
from .markermap import kabsch

reg = o3d.pipelines.registration
MIN_GAP = 10  # keyframes closer in time than this are already tied by odometry
MAX_PER_KEYFRAME = 3  # loop candidates tried per keyframe (the best first)
NEAR_MM, NEAR_DEG = 60.0, 35.0  # poses this close are loop candidates even without markers or features
SHARED_MARKER_MM = 6.0  # markers within this distance (world, current poses) are the same marker
LOOP_FITNESS, LOOP_RMSE_MM, LOOP_MARKER_MM = 0.3, 1.5, 2.0
PREFERENCE = 2.0
ROT_SIGMA_DEG, POS_SIGMA_MM = 0.5, 1.0  # per edge, for the information matrices


def relative(a, b):
    """Transform taking camera b's coordinates to camera a's (both camera -> world, metres)."""
    return np.linalg.inv(a) @ b


def info_matrix(source, target, t):
    """Information of an edge from the fine-level clouds (Open3D), scaled to our noise estimate."""
    m = reg.get_information_matrix_from_point_clouds(source, target, 0.004, t)
    return m if np.isfinite(m).all() and m[3, 3] > 0 else np.diag([1 / np.radians(ROT_SIGMA_DEG) ** 2] * 3 +
                                                                  [1 / (POS_SIGMA_MM / 1000) ** 2] * 3)


def shared_markers(kf_a, kf_b):
    """Marker pairs (camera mm in a, camera mm in b) for markers both keyframes saw."""
    if not len(kf_a["markers"]) or not len(kf_b["markers"]):
        return None
    wa = kf_a["markers"] @ kf_a["pose"][:3, :3].T + kf_a["pose"][:3, 3] * 1000
    wb = kf_b["markers"] @ kf_b["pose"][:3, :3].T + kf_b["pose"][:3, 3] * 1000
    d = np.linalg.norm(wa[:, None] - wb[None], axis=2)
    pairs = [(i, int(np.argmin(d[i]))) for i in range(len(wa)) if d[i].min() < SHARED_MARKER_MM]
    if len(pairs) < 3:
        return None
    return kf_a["markers"][[p[0] for p in pairs]], kf_b["markers"][[p[1] for p in pairs]]


def find_loops(kfs, levels, rng, log=print):
    """Verified loop closures: list of (i, j, transform j -> i, information, kind).

    For each keyframe, at most MAX_PER_KEYFRAME earlier ones are tried, best first: shared markers,
    then the strongest feature matches, then the nearest poses (checking every pair took ~60 s for
    70 keyframes and gave the graph no more than a few good loops per keyframe do).
    """
    loops = []
    feats = None
    for j in range(len(kfs)):
        cands = []  # (priority, -strength, i, start transform, kind, marker pairs)
        for i in range(j - MIN_GAP + 1):
            current = relative(kfs[i]["pose"], kfs[j]["pose"])
            moved = np.linalg.norm(current[:3, 3]) * 1000
            turned = np.degrees(np.arccos(np.clip((np.trace(current[:3, :3]) - 1) / 2, -1, 1)))
            markers = shared_markers(kfs[i], kfs[j])
            if markers is not None:
                t = kabsch(markers[1], markers[0])
                t[:3, 3] /= 1000
                cands.append((0, -len(markers[0]), i, t, "markers", (markers[1], markers[0])))
            elif moved < NEAR_MM * 2 and kfs[i]["feat"] is not None and kfs[j]["feat"] is not None:
                if feats is None:
                    from . import features as feats
                t, inliers = feats.relative_pose(kfs[j]["feat"], kfs[i]["feat"], rng)
                if t is not None:
                    cands.append((1, -inliers, i, t, "features", None))
                elif moved < NEAR_MM and turned < NEAR_DEG:
                    cands.append((2, moved, i, current, "near", None))
            elif moved < NEAR_MM and turned < NEAR_DEG:
                cands.append((2, moved, i, current, "near", None))
        cands.sort(key=lambda c: (c[0], c[1]))
        for _, _, i, start, kind, pairs in cands[:MAX_PER_KEYFRAME]:
            t, info = track.solve(levels[j], levels[i], start, marker_pairs=pairs)
            if (info["fitness"] >= LOOP_FITNESS and info["rmse_mm"] is not None and info["rmse_mm"] <= LOOP_RMSE_MM
                    and (info["marker_rmse_mm"] is None or info["marker_rmse_mm"] <= LOOP_MARKER_MM)):
                loops.append((i, j, t, info_matrix(levels[j][-1], levels[i][-1], t), kind))
    log(f"loop closures: {len(loops)} verified")
    return loops


def optimise_poses(kfs, rng, log=print):
    """New camera -> world poses for the keyframes (list of dicts: pose, points mm, feat, markers)."""
    if len(kfs) < MIN_GAP + 1:
        return [k["pose"] for k in kfs], {"keyframes": len(kfs), "loops_found": 0, "loops_kept": 0}
    levels = [track.levels(k["points"] / 1000) for k in kfs]
    graph = reg.PoseGraph()
    for k in kfs:
        graph.nodes.append(reg.PoseGraphNode(k["pose"]))
    for i in range(len(kfs) - 1):
        t = relative(kfs[i]["pose"], kfs[i + 1]["pose"])
        graph.edges.append(reg.PoseGraphEdge(i + 1, i, t, info_matrix(levels[i + 1][-1], levels[i][-1], t), uncertain=False))
    loops = find_loops(kfs, levels, rng, log)
    for i, j, t, info, _ in loops:
        graph.edges.append(reg.PoseGraphEdge(j, i, t, info, uncertain=True))
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(),
                            reg.GlobalOptimizationOption(max_correspondence_distance=0.004, edge_prune_threshold=0.25,
                                                         preference_loop_closure=PREFERENCE, reference_node=0))
    kept = sum(e.uncertain for e in graph.edges)
    log(f"pose graph: {len(kfs)} keyframes, {len(loops)} loops, {kept} kept")
    kinds = {}
    for *_, kind in loops:
        kinds[kind] = kinds.get(kind, 0) + 1
    return [np.asarray(n.pose) for n in graph.nodes], {"keyframes": len(kfs), "loops_found": len(loops),
                                                       "loops_kept": kept, "loop_kinds": kinds}


def optimise(scanner, log=print):
    """Optimise a Scanner's keyframes and merge its cloud again with the corrected poses."""
    scanner.flush()
    kfs = [{"index": i, "pose": pose, "points": pts, "feat": feat, "markers": mk}
           for i, pose, pts, feat, mk in scanner.keyframes]
    new_poses, report = optimise_poses(kfs, scanner.rng, log)
    moved = [np.linalg.norm((np.linalg.inv(a["pose"]) @ b)[:3, 3]) * 1000 for a, b in zip(kfs, new_poses)]
    report["largest_correction_mm"] = round(float(max(moved)), 2) if moved else 0.0
    scanner.remerge({k["index"]: p for k, p in zip(kfs, new_poses)})
    return report


def make_mesh(scanner, directory, depth=9):
    """Poisson surface from the cloud's oriented normals -> directory/mesh.ply (metres, y up like
    cloud.ply). Low-density vertices (surface guessed far from any point) are dropped. Returns the
    triangle count. (A TSDF mesh needs the depth images; the merged frames keep only points.)"""
    from . import cloud as cl
    xyz, rgb, _, nrm = scanner.cloud.points(cl.MIN_SEEN, with_normals=True)
    ok = np.linalg.norm(nrm, axis=1) > 0.5
    if ok.sum() < 100:
        return 0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[ok] @ cl.FLIP.T / 1000))
    pcd.normals = o3d.utility.Vector3dVector(nrm[ok] @ cl.FLIP.T)
    pcd.colors = o3d.utility.Vector3dVector(rgb[ok] / 255.0)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, 0.05))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box().scale(1.05, pcd.get_center()))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(Path(directory) / "mesh.ply"), mesh)
    return len(mesh.triangles)


def process(scanner, directory, action, log=print):
    """A Process-menu action on a finished scan: "optimise", "small"/"medium"/"large" (merge again at
    that point distance) or "mesh". Saves the result into `directory`; returns a short message."""
    from . import cloud as cl
    directory = Path(directory)
    if action == "optimise":
        report = optimise(scanner, log)
        scanner.save(directory)
        (directory / "optimise.json").write_text(json.dumps(report, indent=2))
        return (f"optimised: {report['loops_kept']} loop closures, largest correction "
                f"{report['largest_correction_mm']} mm")
    if action in cl.PRESETS_MM:
        scanner.remerge({}, point_mm=cl.PRESETS_MM[action])
        scanner.preset = action
        _, kept = scanner.save(directory)
        return f"merged again at {cl.PRESETS_MM[action]} mm: {kept:,} points"
    if action == "mesh":
        return f"mesh: {make_mesh(scanner, directory):,} triangles -> mesh.ply"
    raise ValueError(f"unknown action {action}")


def main():
    """Optimise a saved session (needs keyframes.npz and merged.npz)."""
    from . import cloud as cl
    directory = Path(sys.argv[1])
    scanner = cl.Scanner.restore(directory)
    report = optimise(scanner)
    scanner.save(directory)
    (directory / "optimise.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
