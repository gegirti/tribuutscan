"""Tracking-quality measures that need no ground truth (plus ATE/RPE for synthetic data). numpy only.

Poses are camera -> world 4x4 in metres (as cloud.Scanner keeps them); points are mm.
"""
import numpy as np

from .cloud import pack

JUMP_MM, JUMP_DEG = 30.0, 15.0  # a step larger than this between tracked frames is a pose jump
MARKER_CLUSTER_MM = 8.0  # sightings of one marker closer than this are the same marker


def angle_deg(r):
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def pose_jumps(poses, centre_mm):
    """Steps between consecutive tracked poses that are too large, judged at the object's centre.

    poses: list of (index, 4x4). Returns a list of (index_a, index_b, moved_mm, turned_deg).
    """
    c = np.append(np.asarray(centre_mm) / 1000, 1.0)
    jumps = []
    for (ia, pa), (ib, pb) in zip(poses, poses[1:]):
        ca, cb = (np.linalg.inv(pa) @ c)[:3], (np.linalg.inv(pb) @ c)[:3]  # object centre seen from each camera
        moved, turned = np.linalg.norm(cb - ca) * 1000, angle_deg((np.linalg.inv(pa) @ pb)[:3, :3])
        if moved > JUMP_MM or turned > JUMP_DEG:
            jumps.append((ia, ib, round(moved, 1), round(turned, 1)))
    return jumps


def loop_gap(poses, centre_mm):
    """(mm, deg) between the first and last tracked view of the object centre (information only)."""
    if len(poses) < 2:
        return 0.0, 0.0
    c = np.append(np.asarray(centre_mm) / 1000, 1.0)
    (_, a), (_, b) = poses[0], poses[-1]
    return (float(np.linalg.norm((np.linalg.inv(a) @ c)[:3] - (np.linalg.inv(b) @ c)[:3]) * 1000),
            angle_deg((np.linalg.inv(a) @ b)[:3, :3]))


def double_surface_ratio(xyz, normals, cell_mm=1.0, near_mm=2.5, far_mm=8.0, step_mm=1.0, min_dot=0.9):
    """Share of points that have a second sheet facing the SAME way 2.5-8 mm along their normal.

    A pose error merges a shifted copy of a surface: a parallel sheet with the same orientation.
    A thin wall's two sides face opposite ways and do not count. Points without a normal are ignored.
    Real nearby parallel surfaces (a lid seam) also count, so compare runs on the same data.
    """
    ok = np.linalg.norm(normals, axis=1) > 0.5
    xyz, normals = xyz[ok], normals[ok]
    if not len(xyz):
        return 0.0
    keys, inv = np.unique(pack(np.floor(xyz / cell_mm)), return_inverse=True)
    cell_n = np.stack([np.bincount(inv, normals[:, i], len(keys)) for i in range(3)], 1)
    cell_n /= np.maximum(np.linalg.norm(cell_n, axis=1, keepdims=True), 1e-9)
    doubled = np.zeros(len(xyz), bool)
    for s in np.arange(near_mm, far_mm + 1e-9, step_mm):
        for sign in (1.0, -1.0):
            probe = pack(np.floor((xyz + sign * s * normals) / cell_mm))
            pos = np.minimum(np.searchsorted(keys, probe), len(keys) - 1)
            hit = keys[pos] == probe
            doubled |= hit & (np.einsum("ij,ij->i", cell_n[pos], normals) > min_dot)
    return float(doubled.mean())


def marker_scatter(sightings, cluster_mm=MARKER_CLUSTER_MM, min_obs=3):
    """How well marker sightings agree in the world frame: independent of shape-based tracking.

    sightings: (N, 3) world positions (mm) of every marker seen in every tracked frame. Groups them
    into markers (greedy, by distance), then returns dict(markers=number of groups seen min_obs times,
    scatter_mm=median RMS spread within those groups, stray=share of sightings in small groups).
    Wrong poses smear a marker (larger scatter) or split it into extra groups (more markers, strays).
    """
    sightings = np.asarray(sightings, float)
    if not len(sightings):
        return {"markers": 0, "scatter_mm": None, "stray": None}
    centres, members = [], []
    for p in sightings:
        if centres:
            d = np.linalg.norm(np.array(centres) - p, axis=1)
            k = int(np.argmin(d))
            if d[k] < cluster_mm:
                members[k].append(p)
                centres[k] = np.mean(members[k], axis=0)
                continue
        centres.append(p.copy())
        members.append([p])
    big = [np.array(m) for m in members if len(m) >= min_obs]
    spreads = [float(np.sqrt(((m - m.mean(0)) ** 2).sum(1).mean())) for m in big]
    stray = 1 - sum(len(m) for m in big) / len(sightings)
    return {"markers": len(big), "scatter_mm": float(np.median(spreads)) if spreads else None, "stray": round(stray, 3)}


def ate_rpe(estimated, truth):
    """Absolute and relative trajectory errors against ground truth (both: dict index -> 4x4, metres).

    Both start at the first frame's camera, so no alignment is needed. Returns mm / degrees:
    dict(ate_mm, ate_deg, rpe_mm, frames) over the frames that have an estimate.
    """
    common = sorted(set(estimated) & set(truth))
    if not common:
        return {"ate_mm": None, "ate_deg": None, "rpe_mm": None, "frames": 0}
    dt = [np.linalg.norm(estimated[i][:3, 3] - truth[i][:3, 3]) * 1000 for i in common]
    dr = [angle_deg(estimated[i][:3, :3].T @ truth[i][:3, :3]) for i in common]
    rel = []
    for a, b in zip(common, common[1:]):
        e = np.linalg.inv(estimated[a]) @ estimated[b]
        g = np.linalg.inv(truth[a]) @ truth[b]
        rel.append(np.linalg.norm((np.linalg.inv(g) @ e)[:3, 3]) * 1000)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
    return {"ate_mm": rms(dt), "ate_deg": rms(dr), "rpe_mm": rms(rel), "frames": len(common)}
