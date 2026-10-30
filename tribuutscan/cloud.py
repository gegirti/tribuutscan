"""Merge many depth frames into one compact point cloud: the "point distance" rule real scanners use.

    python3 -m tribuutscan.cloud captures/<stamp>-rec [small|medium|large]
    .venv/bin/python -m tribuutscan.cloud ...      # with Open3D: tracks motion (ICP) between frames

Space is cut into cubes of side `point distance`. Every point that lands in a cube is averaged into
that cube's one point, so the cloud grows with the object's surface, not with scan time. A cube
must be seen in `min_seen` frames to be kept, which drops noise and flying pixels.

Each frame is first cut down to the object: the depth window, then the table or turntable plane
(if the object stands on it), then the connected piece at the image centre.

Writes captures/<stamp>-scan/: cloud.ply (metres, x right, y up, z toward the scanner; like fuse.py),
poses.txt, session.json. Distances rest on the ESTIMATED depth law and field of view.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import frames as fr
from . import markers
from . import pointcloud as pc

PRESETS_MM = {"small": 0.5, "medium": 1.0, "large": 2.0}  # point distance; depth noise ~1 mm (estimate)
MIN_SEEN = 3  # frames a cube must be seen in to be kept
NORMAL_RADIUS_MM = 2.0  # neighbourhood for normals

PLANE_TOL_MM = 3.0  # points this close to the support plane are removed
PLANE_MIN_SHARE = 0.1  # a plane must hold this share of the points to count as the support
PLANE_MIN_FACING = 0.3  # |cos| between plane normal and view ray; below it the plane is seen edge-on
BORDER_PX = 40  # a support plane must have points this close to the image edge
PLANE_MIN_CM2 = 80  # ...and cover this much area. A slice through a curved part covered 33-53 cm2,
                    # a mostly hidden table 100 cm2 (one test each; revisit with turntable scans)
CLUSTER_CELL_MM = 6.0  # cell size for finding the connected piece at the centre

KEY_MOVE_MM, KEY_TURN_DEG = 3.0, 2.0  # merge a frame only after this much motion since the last one
BIAS = 1 << 20  # voxel coordinates are packed into one int64 key, 21 bits each
FLIP = np.diag([1.0, -1.0, -1.0])  # camera (y down, z forward) -> y up, z toward scanner


def pack(cells):
    c = cells.astype(np.int64) + BIAS
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def unpack(keys):
    return np.stack([(keys >> 42) & 0x1FFFFF, (keys >> 21) & 0x1FFFFF, keys & 0x1FFFFF], 1) - BIAS


class VoxelCloud:
    """Sparse cubes of side `point_mm`, each holding the average of every point that fell in it."""

    def __init__(self, point_mm):
        self.point_mm = float(point_mm)
        self.keys = np.empty(0, np.int64)  # sorted
        self.xyz = np.empty((0, 3))  # sums
        self.rgb = np.empty((0, 3))  # sums
        self.n = np.empty(0, np.int64)  # points summed
        self.seen = np.empty(0, np.int32)  # frames that hit the cube
        self.frames = 0
        self.raw_points = 0

    def __len__(self):
        return len(self.keys)

    def add(self, xyz, rgb):
        """Merge one frame's points (mm, world frame) and colours (uint8)."""
        if not len(xyz):
            return
        keys, inv = np.unique(pack(np.floor(xyz / self.point_mm)), return_inverse=True)
        sums = lambda v: np.stack([np.bincount(inv, v[:, i], len(keys)) for i in range(3)], 1)
        xyz_s, rgb_s, n = sums(xyz), sums(rgb.astype(np.float64)), np.bincount(inv, minlength=len(keys))
        pos = np.searchsorted(self.keys, keys)
        old = pos < len(self.keys)
        old[old] = self.keys[pos[old]] == keys[old]
        p = pos[old]
        self.xyz[p] += xyz_s[old]
        self.rgb[p] += rgb_s[old]
        self.n[p] += n[old]
        self.seen[p] += 1
        new = ~old
        self.keys = np.concatenate([self.keys, keys[new]])
        self.xyz = np.concatenate([self.xyz, xyz_s[new]])
        self.rgb = np.concatenate([self.rgb, rgb_s[new]])
        self.n = np.concatenate([self.n, n[new]])
        self.seen = np.concatenate([self.seen, np.ones(new.sum(), np.int32)])
        order = np.argsort(self.keys, kind="stable")
        self.keys, self.xyz, self.rgb = self.keys[order], self.xyz[order], self.rgb[order]
        self.n, self.seen = self.n[order], self.seen[order]
        self.frames += 1
        self.raw_points += len(xyz)

    def points(self, min_seen=MIN_SEEN):
        """(xyz mm, rgb uint8, seen) of the cubes seen in at least `min_seen` frames."""
        keep = self.seen >= min_seen
        n = self.n[keep, None]
        return (self.xyz[keep] / n, np.clip(self.rgb[keep] / n, 0, 255).astype(np.uint8), self.seen[keep])

    def normals(self, xyz, radius_mm=NORMAL_RADIUS_MM):
        """Unit normals of the kept points from their neighbours' spread (PCA), unoriented."""
        cells = np.floor(xyz / self.point_mm).astype(np.int64)
        keys = pack(cells)
        order = np.argsort(keys)
        keys_sorted, xyz_sorted = keys[order], xyz[order]
        r = max(1, int(round(radius_mm / self.point_mm)))
        s1, s2, cnt = np.zeros((len(xyz), 3)), np.zeros((len(xyz), 3, 3)), np.zeros(len(xyz))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nk = pack(cells + (dx, dy, dz))
                    pos = np.clip(np.searchsorted(keys_sorted, nk), 0, len(keys_sorted) - 1)
                    hit = keys_sorted[pos] == nk
                    q = xyz_sorted[pos[hit]] - xyz[hit]  # relative, for precision
                    s1[hit] += q
                    s2[hit] += q[:, :, None] * q[:, None, :]
                    cnt[hit] += 1
        mean = s1 / cnt[:, None]
        cov = s2 / cnt[:, None, None] - mean[:, :, None] * mean[:, None, :]
        return np.linalg.eigh(cov)[1][:, :, 0]  # eigenvector of the smallest eigenvalue


def frame_points(frame):
    """Camera-frame points (mm; x right, y down, z forward), colours, and pixel coords of one frame."""
    d = frame["depth"]
    h, w = d.shape
    v, u = np.nonzero(d > 0)
    z = pc.depth_to_mm(d[v, u])
    f = (w / 2) / np.tan(np.radians(pc.HFOV_DEG) / 2)
    xyz = np.stack([(u - w / 2) * z / f, (v - h / 2) * z / f, z], 1)
    return xyz, pc.register_color(frame["color"])[v, u], np.stack([u, v], 1)


def support_plane(xyz, rng, pix=None, shape=None, iterations=100, sample=2000, tries=3):
    """The table/turntable plane (normal n, offset c with n.p + c = 0, n toward the scanner), or None.

    A plane counts only if it holds PLANE_MIN_SHARE of the points AND most other points lie on the
    scanner's side of it (the object stands on it) AND the scanner is not looking along it (an
    object's side wall seen edge-on is not a table). With `pix`/`shape` given, it must also run out
    of the picture: a table does, a flat lid or face in the middle of the view does not. Finally it
    must cover PLANE_MIN_CM2: a plane slicing a curved surface only catches thin bands.
    A plane that fails is set aside and the next largest plane is tried.
    """
    if len(xyz) < 100:
        return None
    pick = rng.choice(len(xyz), min(sample, len(xyz)), replace=False)
    pts = xyz[pick]
    border = np.zeros(len(pts), bool)
    if pix is not None:
        h, w = shape
        m = BORDER_PX
        u, v = pix[pick, 0], pix[pick, 1]
        border = (u < m) | (u >= w - m) | (v < m) | (v >= h - m)
    rest, rest_border = pts, border
    for _ in range(tries):
        plane, inliers = _ransac_plane(rest, rng, iterations)
        if plane is None or inliers.sum() < PLANE_MIN_SHARE * len(pts):
            return None
        n, c = plane
        if c < 0:  # orient so the scanner (origin) is on the positive side
            n, c = -n, -c
        dist = pts @ n + c
        off = np.abs(dist) >= PLANE_TOL_MM
        centre = rest[inliers].mean(0)
        facing = abs(n @ centre) / np.linalg.norm(centre)  # 0 = seen edge-on, 1 = head-on
        reaches_edge = pix is None or rest_border[inliers].any()
        if facing >= PLANE_MIN_FACING and reaches_edge and (not off.any() or (dist[off] > 0).mean() >= 0.8) \
                and plane_area_cm2(xyz, n, c) >= PLANE_MIN_CM2:
            return n, c
        rest, rest_border = rest[~inliers], rest_border[~inliers]
    return None


def plane_area_cm2(xyz, n, c, cell_mm=10.0):
    """Area covered by the points within PLANE_TOL_MM of the plane, counted in cell_mm squares."""
    on = xyz[np.abs(xyz @ n + c) < PLANE_TOL_MM]
    a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    cells = np.floor(np.c_[on @ a, on @ np.cross(n, a)] / cell_mm).astype(np.int64)
    return len(np.unique(cells, axis=0)) * (cell_mm / 10) ** 2


def _ransac_plane(pts, rng, iterations):
    """Best plane through `pts` by RANSAC: ((n, c), inlier mask) or (None, None)."""
    best, best_mask = None, None
    if len(pts) < 3:
        return None, None
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        mask = np.abs((pts - a) @ n) < PLANE_TOL_MM
        if best_mask is None or mask.sum() > best_mask.sum():
            best, best_mask = (n, -n @ a), mask
    return best, best_mask


def centre_piece(xyz, pix, shape):
    """Mask of the points connected (through CLUSTER_CELL_MM cells) to the point nearest the image centre."""
    cells = np.floor(xyz / CLUSTER_CELL_MM).astype(np.int64)
    keys = pack(cells)
    uniq, inv = np.unique(keys, return_inverse=True)
    h, w = shape
    seed = uniq[inv[np.argmin(np.hypot(pix[:, 0] - w / 2, pix[:, 1] - h / 2))]]
    occupied = set(uniq.tolist())
    steps = [(dx << 42) + (dy << 21) + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
             if (dx, dy, dz) != (0, 0, 0)]
    piece, todo = {seed}, [seed]
    while todo:
        k = todo.pop()
        for s in steps:
            nk = k + s
            if nk in occupied and nk not in piece:
                piece.add(nk)
                todo.append(nk)
    return np.isin(keys, np.fromiter(piece, np.int64))


def isolate(xyz, rgb, pix, shape, rng):
    """Cut a frame down to the object: drop the support plane, keep the piece at the centre."""
    plane = support_plane(xyz, rng, pix, shape)
    if plane is not None:
        keep = np.abs(xyz @ plane[0] + plane[1]) >= PLANE_TOL_MM
        xyz, rgb, pix = xyz[keep], rgb[keep], pix[keep]
    if len(xyz):
        keep = centre_piece(xyz, pix, shape)
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb, plane is not None


def turned(t):
    """(movement mm, rotation degrees) of a 4x4 rigid transform in metres."""
    return np.linalg.norm(t[:3, 3]) * 1000, np.degrees(np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1)))


def scan(directory, preset="small", log=print):
    """Merge a recording into a VoxelCloud. Returns (cloud, poses, stats)."""
    try:
        from . import fuse  # Open3D: frame-to-frame ICP
    except ImportError:
        fuse = None
        log("Open3D not found: no tracking, frames are merged as if nothing moved")
    rng = np.random.default_rng(0)
    cloud = VoxelCloud(PRESETS_MM[preset])
    pose, motion, prev, last_merged = np.eye(4), np.eye(4), None, None
    poses, stats = [], {"frames": 0, "merged": 0, "lost": 0, "plane_frames": 0, "markers": []}
    for i, frame in enumerate(fr.read_recording(directory)):
        stats["frames"] += 1
        xyz, rgb, pix = frame_points(frame)
        xyz, rgb, had_plane = isolate(xyz, rgb, pix, frame["depth"].shape, rng)
        stats["plane_frames"] += had_plane
        if len(xyz) < 500:
            stats["lost"] += 1
            continue
        if fuse is not None:
            levels = fuse.levels(xyz / 1000)
            if prev is not None:
                result = fuse.icp(levels, prev, motion)
                if result.fitness < fuse.MIN_FITNESS or not fuse.plausible(result.transformation):
                    stats["lost"] += 1
                    continue
                motion = result.transformation
                pose = pose @ motion
            prev = levels
        if last_merged is not None:
            move, turn = turned(np.linalg.inv(last_merged) @ pose)
            if fuse is not None and move < KEY_MOVE_MM and turn < KEY_TURN_DEG:
                continue
        world = xyz @ pose[:3, :3].T + pose[:3, 3] * 1000
        cloud.add(world, rgb)
        last_merged = pose.copy()
        poses.append((i, pose.copy()))
        stats["merged"] += 1
        if not stats["markers"]:
            stats["markers"] = [[round(x, 1), round(y, 1)] for x, y, *_ in markers.detect(
                markers.luma(frame["color"]), rgb=frame["color"])]
        if stats["merged"] % 10 == 1:
            log(f"frame {i:4d}: merged {stats['merged']:3d}, cubes {len(cloud):7d}, raw points so far {cloud.raw_points:9d}")
    return cloud, poses, stats


def save(directory, cloud, poses, stats, preset, min_seen=MIN_SEEN):
    """Write cloud.ply, poses.txt and session.json; returns (path, number of points)."""
    directory.mkdir(parents=True, exist_ok=True)
    xyz, rgb, seen = cloud.points(min_seen)
    normals = cloud.normals(xyz) if len(xyz) else np.empty((0, 3))
    if poses:  # face the normals toward the cameras that saw the points
        centres = np.array([p[:3, 3] for _, p in poses]) * 1000
        flip = np.einsum("ij,ij->i", normals, centres.mean(0) - xyz) < 0
        normals[flip] *= -1
    out = directory / "cloud.ply"
    pc.write_ply(out, (xyz @ FLIP.T / 1000).astype(np.float32), rgb, normals=normals @ FLIP.T, seen=seen)
    with open(directory / "poses.txt", "w") as f:
        for i, p in poses:
            f.write(f"{i} " + " ".join(f"{v:.6f}" for v in p.ravel()) + "\n")
    (directory / "session.json").write_text(json.dumps({
        "preset": preset, "point_distance_mm": cloud.point_mm, "min_seen": min_seen,
        "raw_points": cloud.raw_points, "cubes": len(cloud), "points_kept": len(xyz), **stats,
        "note": "distances rest on the estimated depth law and field of view",
    }, indent=2))
    return out, len(xyz)


def main():
    directory = Path(sys.argv[1])
    preset = sys.argv[2] if len(sys.argv) > 2 else "small"
    t0 = time.time()
    cloud, poses, stats = scan(directory, preset)
    out_dir = directory.parent / (directory.name.removesuffix("-rec") + "-scan")
    out, kept = save(out_dir, cloud, poses, stats, preset)
    raw_mb = cloud.raw_points * 15 / 1e6  # float32 xyz + rgb per raw point
    print(f"{stats['frames']} frames, {stats['merged']} merged, {stats['lost']} lost or skipped, "
          f"support plane in {stats['plane_frames']}")
    print(f"raw points {cloud.raw_points} (~{raw_mb:.0f} MB) -> {len(cloud)} cubes at {cloud.point_mm} mm "
          f"-> {kept} kept (seen >= {MIN_SEEN}), {out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
