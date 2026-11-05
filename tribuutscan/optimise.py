"""Optimise a finished scan: find loop closures between keyframes, correct all poses together in a
pose graph, and merge the cloud again with the corrected poses. Open3D + OpenCV (.venv).

    .venv/bin/python -m tribuutscan.optimise captures/<stamp>-scan   # (sessions saved by live scans)

Tracking runs frame after frame, so small errors add up; when the scan comes back to a place it has
seen, a loop closure says where it really is, and the pose graph spreads the correction over every
keyframe. Candidates come from markers seen in both keyframes, ORB feature matches, and keyframes
whose poses are close; each is checked with the joint solve (track.solve) before it is used, and
the graph's line process drops loops that disagree with the rest (preference 2.0: on a synthetic
loop it kept the true closure and pruned a 30 mm false one; at 1.0 it pruned both). A loop whose views
the shape cannot pin (track.DEGENERATE) is not used unless markers held it: on the first platform scan
such loops between edge-on views of the hand were 8-44 degrees off (degeneracy 0.0003-0.004, the good
ones 0.013 and up), and the graph kept them all.
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
JOIN_KEYFRAMES = 12  # a scan piece's keyframes compared by ORB with each joined piece's (spread over both)
JOIN_ORB_TRIES = 4  # ...the pairs with the most matches tried
JOIN_GRID_MM = 2.0  # pieces' object points thinned to this grid for the join's checks


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
                    and (info["marker_rmse_mm"] is None or info["marker_rmse_mm"] <= LOOP_MARKER_MM)
                    and (info["markers_used"] or (info["degeneracy"] or 0.0) >= track.DEGENERATE)):
                loops.append((i, j, t, info_matrix(levels[j][-1], levels[i][-1], t), kind))
    log(f"loop closures: {len(loops)} verified")
    return loops


def optimise_poses(kfs, rng, log=print, breaks=(), joins=()):
    """New camera -> world poses for the keyframes (list of dicts: pose, points mm, feat, markers). `breaks`:
    positions whose keyframe starts a scan piece (no odometry edge from the one before: the scan was lost
    between them); `joins`: (a, b) position pairs whose relative pose join_pieces verified (certain edges)."""
    if len(kfs) < MIN_GAP + 1:
        return [k["pose"] for k in kfs], {"keyframes": len(kfs), "loops_found": 0, "loops_kept": 0}
    levels = [track.levels(k["points"] / 1000) for k in kfs]
    graph = reg.PoseGraph()
    for k in kfs:
        graph.nodes.append(reg.PoseGraphNode(k["pose"]))
    breaks = set(breaks)
    for i in range(len(kfs) - 1):
        if i + 1 in breaks:
            continue
        t = relative(kfs[i]["pose"], kfs[i + 1]["pose"])
        graph.edges.append(reg.PoseGraphEdge(i + 1, i, t, info_matrix(levels[i + 1][-1], levels[i][-1], t), uncertain=False))
    for a, b in joins:
        t = relative(kfs[b]["pose"], kfs[a]["pose"])
        graph.edges.append(reg.PoseGraphEdge(a, b, t, info_matrix(levels[a][-1], levels[b][-1], t), uncertain=False))
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


def object_features(feat, scene, pose):
    """A keyframe's colour features (points mm in its camera, descriptors) on the object only (scene.on_object
    at the keyframe's pose), for scans whose room stood still while the object turned: there the room's
    features say "nothing moved" (the first platform scan closed 102 loops on them, largest correction 196 mm)."""
    if feat is None or feat[1] is None or not len(feat[0]):
        return feat
    from .cloud import scene_transform
    keep = scene.on_object(np.asarray(feat[0], float), scene_transform(pose))
    return feat[0][keep], feat[1][keep]


def piece_points(kfs):
    """A scan piece's object points (world mm) from its keyframes (index, pose, points, ...), thinned."""
    from .gputrack import grid_points
    pts = [k[2] @ k[1][:3, :3].T + k[1][:3, 3] * 1000 for k in kfs if len(k[2])]
    return grid_points(np.concatenate(pts), JOIN_GRID_MM) if pts else np.zeros((0, 3))


def spread(items, n):
    return [items[i] for i in sorted(set(np.linspace(0, len(items) - 1, min(n, len(items))).round().astype(int)))] \
        if items else []


def join_candidates(scanner, groups, k, placed, maps):
    """Transforms (metres) that may take scan piece k's world into the joined world, with how each was found:
    shared markers, ORB between keyframes, shape. `placed`: joined piece -> its transform."""
    from . import features, session
    out = []
    mk = maps[k]
    for j, tj in placed.items():
        if len(mk) >= 3 and len(maps[j]) >= 3:
            found = maps[j].match_global(mk.pos[np.argsort(-mk.count)])
            if found is not None:
                out.append((tj @ found, "markers"))
    turns = getattr(scanner, "object_turns", False)
    feat_of = lambda kf: object_features(kf[3], scanner.scene_of(kf[0]), kf[1]) if turns else kf[3]
    mine = [(kf, feat_of(kf)) for kf in spread([kf for kf in groups[k] if kf[3] is not None], JOIN_KEYFRAMES)]
    theirs = [(kf, feat_of(kf), placed[j]) for j in placed
              for kf in spread([kf for kf in groups[j] if kf[3] is not None], JOIN_KEYFRAMES)]
    pairs = sorted(((len(features.good_matches(fa, fb)), ia, ib) for ia, (_, fa) in enumerate(mine)
                    for ib, (_, fb, _) in enumerate(theirs) if fa is not None and fb is not None), reverse=True)
    for count, ia, ib in pairs[:JOIN_ORB_TRIES]:
        if count < features.MIN_INLIERS:
            break
        (ka, fa), (kb, fb, tb) = mine[ia], theirs[ib]
        found, inliers = features.relative_pose(fa, fb, scanner.rng)
        if found is not None:
            out.append((tb @ kb[1] @ found @ np.linalg.inv(ka[1]), "features"))
    src = piece_points(groups[k])
    dst = np.concatenate([piece_points(groups[j]) @ placed[j][:3, :3].T + placed[j][:3, 3] * 1000 for j in placed])
    found = session.shape_pose(src, dst)
    if found is not None:
        out.append((found, "shape"))
    return out, src, dst


def join_pieces(scanner, log=print):
    """After Stop, before Optimise: put a scan's pieces (cloud.Scanner.start_piece) into one world, the first
    piece's (the biggest one's when the first is too small to keep). Each later piece is placed by shared markers, ORB between keyframes or shape (join_candidates),
    each candidate refined by the joint solve against the pieces placed so far and taken only as Optimise
    takes a loop: session.accept, and markers or a shape that pins the pose (track.DEGENERATE). A piece that
    joins moves with all its frames; one that joins nothing (or has fewer than PIECE_MIN_KEYFRAMES) is left
    out of the scan: its points are kept apart (scanner.left_pieces, saved as piece-N.ply). Returns a report,
    or None for a scan of one piece."""
    from . import cloud as cl
    from . import session
    pieces = scanner.pieces
    if len(pieces) < 2 or all("joined" in p for p in pieces):
        return None
    scanner.flush()
    groups = [[kf for kf in scanner.keyframes if scanner.piece_of(kf[0]) == k] for k in range(len(pieces))]
    maps = scanner.piece_maps + [scanner.marker_map]
    # The scan's world is its first piece's: the scan started there, aimed at the object, and found its base and
    # region from that view. A later piece starts mid-motion after a loss: on handgap (CPU) the second piece's
    # region came out 25 mm wide, and as the anchor it cut the joined cloud to 73k points (the first's: 135k).
    # The biggest piece only when the first is too small to keep.
    sizes = [len(g) for g in groups]
    anchor = 0 if sizes[0] >= cl.PIECE_MIN_KEYFRAMES else int(np.argmax(sizes))
    placed, joins, how = {anchor: np.eye(4)}, [], {}
    waiting = [k for k in range(len(pieces)) if k != anchor and len(groups[k]) >= cl.PIECE_MIN_KEYFRAMES]
    progress = True
    while waiting and progress:  # (a piece may join through one placed after it)
        progress = False
        for k in list(waiting):
            candidates, src, dst = join_candidates(scanner, groups, k, placed, maps)
            best = None
            if len(src) >= 200 and len(dst) >= 200:
                ls, ld = track.levels(src / 1000), track.levels(dst / 1000)
                for start, kind in candidates:
                    t, info = track.solve(ls, ld, start)
                    if session.accept(info) and (info["markers_used"] or (info["degeneracy"] or 0) >= track.DEGENERATE) \
                            and (best is None or info["fitness"] > best[1]["fitness"]):
                        best = (t, info, kind)
            if best is None:
                continue
            placed[k], how[k] = best[0], {"by": best[2], "fitness": round(best[1]["fitness"], 3),
                                          "rmse_mm": round(best[1]["rmse_mm"], 3)}
            waiting.remove(k)
            progress = True
            mine = np.array([(best[0] @ kf[1])[:3, 3] for kf in groups[k]])  # the join's edge in the pose graph:
            others = [(kf, placed[j]) for j in placed if j != k for kf in groups[j]]  # the keyframe pair whose
            theirs = np.array([(tb @ kf[1])[:3, 3] for kf, tb in others])  # cameras are nearest once placed
            d = np.linalg.norm(mine[:, None] - theirs[None], axis=2)
            a, b = np.unravel_index(int(np.argmin(d)), d.shape)
            joins.append((groups[k][a][0], others[b][0][0]))
    left = [k for k in range(len(pieces)) if k not in placed]
    scene = scanner.scene_of(pieces[anchor]["first_index"])
    for k, t in placed.items():  # the scan's region takes in what the joined pieces saw of the object
        if k != anchor:
            scene.cover(piece_points(groups[k]) @ t[:3, :3].T + t[:3, 3] * 1000)
    for k, t in placed.items():  # every joined piece's frames into the scan's world
        if k == anchor:
            continue
        lo, hi = pieces[k]["first_index"], pieces[k + 1]["first_index"] if k + 1 < len(pieces) else np.inf
        inside = lambda i: lo <= i < hi
        scanner.keyframes = [(i, t @ p, *rest) if inside(i) else (i, p, *rest) for i, p, *rest in scanner.keyframes]
        scanner.poses = [(i, t @ p if inside(i) else p) for i, p in scanner.poses]
        scanner.track_log = [(i, st, t @ p if p is not None and inside(i) else p) for i, st, p in scanner.track_log]
    scanner.left_pieces = scanner.take_pieces_out(left)
    scanner.scene = scene
    scanner.piece_joins = joins
    for k, p in enumerate(pieces):
        p["joined"] = k in placed
    scanner.remerge({})
    report = {"pieces": len(pieces), "anchor": anchor + 1, "joined": {k + 1: how[k] for k in how},
              "left_out": [k + 1 for k in left]}
    log(f"pieces: {len(placed)} of {len(pieces)} in one world" + (f", {len(left)} left out" if left else "")
        + "".join(f"; piece {k + 1} by {h['by']} (fitness {h['fitness']}, {h['rmse_mm']} mm)" for k, h in how.items()))
    return report


def optimise(scanner, log=print):
    """Optimise a Scanner's keyframes and merge its cloud again with the corrected poses (its pieces joined
    first, join_pieces)."""
    scanner.flush()
    pieces_report = join_pieces(scanner, log) if len(getattr(scanner, "pieces", ())) > 1 else None
    kfs = [{"index": i, "pose": pose, "points": pts, "feat": feat, "markers": mk}
           for i, pose, pts, feat, mk in scanner.keyframes]
    if getattr(scanner, "object_turns", False):
        for k in kfs:
            k["feat"] = object_features(k["feat"], scanner.scene, k["pose"])
    breaks, joins = (), ()
    if len(getattr(scanner, "pieces", ())) > 1:
        position = {k["index"]: n for n, k in enumerate(kfs)}
        breaks = [n for n in range(1, len(kfs)) if scanner.piece_of(kfs[n]["index"]) != scanner.piece_of(kfs[n - 1]["index"])]
        joins = [(position[a], position[b]) for a, b in getattr(scanner, "piece_joins", []) if a in position and b in position]
    new_poses, report = optimise_poses(kfs, scanner.rng, log, breaks, joins)
    if pieces_report is not None:
        report["pieces"] = pieces_report
    moved = [np.linalg.norm((np.linalg.inv(a["pose"]) @ b)[:3, 3]) * 1000 for a, b in zip(kfs, new_poses)]
    report["largest_correction_mm"] = round(float(max(moved)), 2) if moved else 0.0  # (at the camera)
    at_object, turned = [], []
    for k, b in zip(kfs, new_poses):  # where each keyframe's view of the object moves: what the surface feels
        c = k["points"].mean(0) / 1000 if len(k["points"]) else np.zeros(3)
        at_object.append(np.linalg.norm((b[:3, :3] - k["pose"][:3, :3]) @ c + b[:3, 3] - k["pose"][:3, 3]) * 1000)
        r = (np.linalg.inv(k["pose"]) @ b)[:3, :3]
        turned.append(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
    report["largest_correction_at_object_mm"] = round(float(max(at_object)), 2) if at_object else 0.0
    report["largest_correction_deg"] = round(float(max(turned)), 2) if turned else 0.0
    scanner.remerge({k["index"]: p for k, p in zip(kfs, new_poses)})
    return report


def make_mesh(scanner, directory, depth=9):
    """Poisson surface from the cloud's oriented normals -> directory/mesh.ply (metres, y up like
    cloud.ply). Low-density vertices (surface guessed far from any point) are dropped. Returns the
    triangle count. Only for scans whose recordings are not at hand: fusion.py fuses the frames
    themselves (a TSDF), which is closer to the surface."""
    from . import cloud as cl
    xyz, rgb, _, nrm = scanner.output_cloud().points(cl.MIN_SEEN, with_normals=True)  # as saved (base or not)
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


def process(scanner, directory, action, log=print, box=None):
    """A Process-menu action on a finished scan: "optimise", "small"/"medium"/"large" (merge again at
    that point distance), "mesh", "base-off"/"base-on" (save the cloud without / with the base it
    stands on), or "crop" (keep what lies inside `box`: lo xyz, hi xyz, mm along Scanner.crop_axes) /
    "crop-off". Saves the result into `directory` (after a crop the mesh too, when there is one);
    returns a short message."""
    from . import cloud as cl
    directory = Path(directory)
    if action == "optimise":
        report = optimise(scanner, log)
        scanner.save(directory)
        (directory / "optimise.json").write_text(json.dumps(report, indent=2))
        pieces = report.get("pieces") or {}
        return (f"optimised: {report['loops_kept']} loop closures, largest correction "
                f"{report['largest_correction_at_object_mm']} mm at the object ({report['largest_correction_deg']} degrees)"
                + (f"; pieces: {len(pieces['joined']) + 1} of {pieces['pieces']} joined" if pieces else "")
                + (f", piece {', '.join(map(str, pieces['left_out']))} left out (piece-N.ply)" if pieces.get("left_out") else ""))
    if action in cl.PRESETS_MM:
        scanner.remerge({}, point_mm=cl.PRESETS_MM[action])
        scanner.preset = action
        _, kept = scanner.save(directory)
        return f"merged again at {cl.PRESETS_MM[action]} mm: {kept:,} points"
    if action == "mesh":
        from . import fusion
        try:
            report = fusion.make_mesh(directory, log=log, scanner=scanner)
        except (ValueError, FileNotFoundError) as e:  # no recordings noted or kept, or another camera
            return f"{e}. Made a rough mesh from the cloud instead: {make_mesh(scanner, directory):,} triangles"
        closed = report.get("solid", {}).get("edges_closed")
        return (f"mesh: {report['triangles']:,} triangles from {report['frames']} frames in "
                f"{report['fuse_s'] + report['extract_s'] + report.get('solid', {}).get('seconds', 0):.0f} s -> mesh.ply, "
                + ("mesh.stl (closed for printing), solid.ply" if closed else "mesh.stl (left open)"))
    if action in ("base-off", "base-on"):
        if scanner.scene.plane is None:
            return "no base was found in this scan: nothing to remove"
        scanner.remove_base = action == "base-off"
        _, kept = scanner.save(directory)
        return f"base {'removed' if scanner.remove_base else 'kept'}: {kept:,} points (make the mesh again)"
    if action in ("crop", "crop-off"):
        if action == "crop":
            box = np.asarray([] if box is None else box, float)
            if box.shape != (6,) or not np.isfinite(box).all() or (box[:3] >= box[3:]).any():
                return "crop needs a box: six numbers in mm, each low below its high"
            axes = scanner.crop["axes"] if scanner.crop is not None else scanner.crop_axes().round(6).tolist()
            scanner.crop = {"axes": axes, "lo": box[:3].tolist(), "hi": box[3:].tolist()}
        else:
            scanner.crop = None
        _, kept = scanner.save(directory)
        message = f"{'cropped' if scanner.crop else 'crop cleared'}: {kept:,} points"
        if (directory / "mesh.ply").exists():  # the mesh, its solid and the STL follow
            message += "; " + process(scanner, directory, "mesh", log)
        return message
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
