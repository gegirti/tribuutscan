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
from .image import yuyv_pixels, yuyv_to_rgb

PRESETS_MM = {"small": 0.5, "medium": 1.0, "large": 2.0}  # point distance; depth noise ~1 mm (estimate)
PIXEL_MM = 0.25  # a depth pixel spans ~0.23 mm at 25 cm (estimate); frames are thinned to ~1 point per cube
MIN_SEEN = 3  # frames a cube must be seen in to be kept
NORMAL_BLOCK_MM = 2.0  # normals are fitted over 3x3x3 blocks of this size (6 mm)

PLANE_TOL_MM = 3.0  # points this close to the support plane are removed
PLANE_MIN_SHARE = 0.1  # a plane must hold this share of the points to count as the support
PLANE_MIN_FACING = 0.3  # |cos| between plane normal and view ray; below it the plane is seen edge-on
BORDER_PX = 40  # a support plane must have points this close to the image edge
PLANE_MIN_CM2 = 80  # ...and cover this much area. A slice through a curved part covered 33-53 cm2,
                    # a mostly hidden table 100 cm2 (one test each; revisit with turntable scans)
CLUSTER_CELL_MM = 6.0  # cell size for finding the connected piece at the centre

KEY_MOVE_MM, KEY_TURN_DEG = 3.0, 2.0  # merge a frame only after this much motion since the last one
MAX_STEP_MM, MAX_STEP_DEG = 30.0, 15.0  # larger jumps between tracked frames are tracking failures
RELOCK_FITNESS = 0.6  # after a loss, a frame must match this share of its points to the cloud so far
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
        self.nrm = np.empty((0, 3))  # sums of oriented unit normals (zero when a frame gave none)
        self.seen = np.empty(0, np.int32)  # frames that hit the cube
        self.frames = 0
        self.raw_points = 0

    def __len__(self):
        return len(self.keys)

    def add(self, xyz, rgb, nrm=None):
        """Merge one frame's points (mm, world frame), colours (uint8) and optional oriented normals."""
        if not len(xyz):
            return
        keys, inv = np.unique(pack(np.floor(xyz / self.point_mm)), return_inverse=True)
        sums = lambda v: np.stack([np.bincount(inv, v[:, i], len(keys)) for i in range(3)], 1)
        xyz_s, rgb_s, n = sums(xyz), sums(rgb.astype(np.float64)), np.bincount(inv, minlength=len(keys))
        nrm_s = sums(nrm) if nrm is not None else np.zeros((len(keys), 3))
        pos = np.searchsorted(self.keys, keys)
        old = pos < len(self.keys)
        old[old] = self.keys[pos[old]] == keys[old]
        p = pos[old]
        self.xyz[p] += xyz_s[old]
        self.rgb[p] += rgb_s[old]
        self.n[p] += n[old]
        self.nrm[p] += nrm_s[old]
        self.seen[p] += 1
        new = ~old
        at = pos[new]  # insertion points keep the keys sorted (no re-sort of the whole cloud)
        self.keys = np.insert(self.keys, at, keys[new])
        self.xyz = np.insert(self.xyz, at, xyz_s[new], axis=0)
        self.rgb = np.insert(self.rgb, at, rgb_s[new], axis=0)
        self.n = np.insert(self.n, at, n[new])
        self.nrm = np.insert(self.nrm, at, nrm_s[new], axis=0)
        self.seen = np.insert(self.seen, at, np.ones(new.sum(), np.int32))
        self.frames += 1
        self.raw_points += len(xyz)

    def points(self, min_seen=MIN_SEEN, with_normals=False):
        """(xyz mm, rgb uint8, seen[, oriented unit normals]) of the cubes seen in >= `min_seen` frames.

        Oriented normals (pointing toward the cameras that saw the cube) are zero where no frame
        supplied one.
        """
        keep = self.seen >= min_seen
        n = self.n[keep, None]
        out = (self.xyz[keep] / n, np.clip(self.rgb[keep] / n, 0, 255).astype(np.uint8), self.seen[keep])
        if not with_normals:
            return out
        s = self.nrm[keep]
        length = np.linalg.norm(s, axis=1, keepdims=True)
        return out + (np.divide(s, length, out=np.zeros_like(s), where=length > 0),)

    def normals(self, xyz, block_mm=NORMAL_BLOCK_MM):
        """Unit normals (unoriented) for shading: PCA of the points in each 3x3x3 group of blocks.

        Computed once per block of `block_mm` and shared by the points in it, so the cost depends on
        the object's size, not on the point distance (per-point neighbourhoods took 30 s at 0.5 mm).
        """
        keys, inv = np.unique(pack(np.floor(xyz / block_mm)), return_inverse=True)
        origin = xyz.mean(0)
        q = xyz - origin  # relative, for precision
        n = np.bincount(inv, minlength=len(keys)).astype(np.float64)
        s1 = np.stack([np.bincount(inv, q[:, i], len(keys)) for i in range(3)], 1)
        s2 = np.stack([np.bincount(inv, q[:, i] * q[:, j], len(keys)) for i in range(3) for j in range(3)], 1)
        a1, a2, an = np.zeros_like(s1), np.zeros_like(s2), np.zeros_like(n)
        for step in np.concatenate([[0], STEPS_26]):
            pos = np.minimum(np.searchsorted(keys, keys + step), len(keys) - 1)
            hit = keys[pos] == keys + step
            a1[hit] += s1[pos[hit]]
            a2[hit] += s2[pos[hit]]
            an[hit] += n[pos[hit]]
        mean = a1 / an[:, None]
        cov = a2.reshape(-1, 3, 3) / an[:, None, None] - mean[:, :, None] * mean[:, None, :]
        return np.linalg.eigh(cov)[1][:, :, 0][inv]  # eigenvector of the smallest eigenvalue


def frame_points(frame, stride=1, normals=False):
    """Camera-frame points (mm; x right, y down, z forward), colours, and pixel coords of one frame.

    `stride` takes every n-th pixel in both directions (more points than cubes is wasted work).
    The frame holds either decoded "color" or the raw "yuyv" buffer (then only the needed
    pixels are converted, which is much cheaper live). With `normals`, also returns unit normals
    from the neighbouring pixels, facing the camera (zero where a neighbour has no depth).
    """
    d = frame["depth"]
    h, w = d.shape
    ds = d[::stride, ::stride]
    valid = ds > 0
    v, u = np.nonzero(valid)
    v, u = v * stride, u * stride
    z = pc.depth_to_mm(d[v, u])
    f = (w / 2) / np.tan(np.radians(pc.HFOV_DEG) / 2)
    xyz = np.stack([(u - w / 2) * z / f, (v - h / 2) * z / f, z], 1)
    if "color" in frame:
        rgb = pc.register_color(frame["color"])[v, u]
    else:
        rgb = yuyv_pixels(frame["yuyv"], w, *pc.color_source(u, v, w, h))
    if not normals:
        return xyz, rgb, np.stack([u, v], 1)
    return xyz, rgb, np.stack([u, v], 1), grid_normals(ds, stride, f)[valid]


def grid_normals(ds, stride, f):
    """Unit normals of an organised depth grid (raw values, every `stride`-th pixel), facing the camera."""
    hs, ws = ds.shape
    h, w = hs * stride, ws * stride
    vv, uu = np.mgrid[0:hs, 0:ws] * stride
    z = np.where(ds > 0, pc.depth_to_mm(ds), np.nan)
    grid = np.stack([(uu - w / 2) * z / f, (vv - h / 2) * z / f, z], -1)
    dx, dy = np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    dx[:, 1:-1] = grid[:, 2:] - grid[:, :-2]
    dy[1:-1] = grid[2:] - grid[:-2]
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / length
        n *= np.where((n * grid).sum(-1, keepdims=True) > 0, -1.0, 1.0)  # face the camera (at the origin)
    return np.nan_to_num(n)


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
    xyz = xyz[::max(1, len(xyz) // 50000)]  # a sample is plenty to fill 1 cm squares
    on = xyz[np.abs(xyz @ n + c) < PLANE_TOL_MM]
    a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    cells = np.floor(np.c_[on @ a, on @ np.cross(n, a)] / cell_mm).astype(np.int64)
    return len(np.unique(cells[:, 0] * 1_000_003 + cells[:, 1])) * (cell_mm / 10) ** 2


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


STEPS_26 = np.array([(dx << 42) + (dy << 21) + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                     if (dx, dy, dz) != (0, 0, 0)], np.int64)  # neighbour offsets in packed-key space


def centre_piece(xyz, pix, shape):
    """Mask of the points connected (through CLUSTER_CELL_MM cells) to the point nearest the image centre."""
    keys = pack(np.floor(xyz / CLUSTER_CELL_MM).astype(np.int64))
    cells, inv = np.unique(keys, return_inverse=True)
    h, w = shape
    start = inv[np.argmin(np.hypot(pix[:, 0] - w / 2, pix[:, 1] - h / 2))]
    piece = np.zeros(len(cells), bool)
    piece[start] = True
    frontier = cells[[start]]
    while frontier.size:  # grow one ring of neighbouring cells at a time (vectorised flood fill)
        cand = (frontier[:, None] + STEPS_26).ravel()
        pos = np.minimum(np.searchsorted(cells, cand), len(cells) - 1)
        found = np.unique(pos[cells[pos] == cand])
        found = found[~piece[found]]
        piece[found] = True
        frontier = cells[found]
    return piece[inv]


def isolate_mask(xyz, pix, shape, rng):
    """Mask of the object's points (support plane dropped, piece at the centre kept), and had_plane."""
    keep = np.ones(len(xyz), bool)
    plane = support_plane(xyz, rng, pix, shape)
    if plane is not None:
        keep = np.abs(xyz @ plane[0] + plane[1]) >= PLANE_TOL_MM
    if keep.any():
        idx = np.nonzero(keep)[0]
        keep[idx[~centre_piece(xyz[idx], pix[idx], shape)]] = False
    return keep, plane is not None


def isolate(xyz, rgb, pix, shape, rng):
    """Cut a frame down to the object: drop the support plane, keep the piece at the centre."""
    keep, had_plane = isolate_mask(xyz, pix, shape, rng)
    return xyz[keep], rgb[keep], had_plane


def plausible_step(t, xyz):
    """Is a frame-to-frame motion believable? Judged at the object's centre, not the camera.

    Turning the object a few degrees in place swings the camera-equivalent pose by centimetres
    (it is ~25 cm away), so a limit on camera movement wrongly rejects it. The object's own centre
    barely moves in that case; a tracking failure moves it a lot.
    """
    c = xyz.mean(0) / 1000
    moved_mm = np.linalg.norm(t[:3, :3] @ c + t[:3, 3] - c) * 1000
    return moved_mm <= MAX_STEP_MM and turned(t)[1] <= MAX_STEP_DEG


def turned(t):
    """(movement mm, rotation degrees) of a 4x4 rigid transform in metres."""
    return np.linalg.norm(t[:3, 3]) * 1000, np.degrees(np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1)))


class Scanner:
    """Feeds frames one at a time: isolate the object, track it (ICP, if Open3D is present), merge."""

    def __init__(self, preset="small", log=print):
        try:
            from . import fuse  # Open3D: frame-to-frame ICP
        except ImportError:
            fuse = None
            log("Open3D not found: no tracking, frames are merged as if nothing moved")
        self.fuse, self.log, self.preset = fuse, log, preset
        self.rng = np.random.default_rng(0)
        self.cloud = VoxelCloud(PRESETS_MM[preset])
        self.stride = max(1, int(PRESETS_MM[preset] / PIXEL_MM))
        self.pose, self.motion, self.prev, self.last_merged = np.eye(4), np.eye(4), None, None
        self.poses = []
        self.model, self.model_merged = None, -1  # cloud-so-far, prepared for relocking
        self.stats = {"frames": 0, "merged": 0, "lost": 0, "relocked": 0, "plane_frames": 0, "markers": [],
                      "tracking": True}
        self.track_log = []  # per frame: (index, status, pose or None); for evaluation
        self.on_merge = None  # optional callback(index, world_xyz_mm, rgb, world_normals, pose)

    def feed(self, frame, index=None):
        """Process one frame set ({'color', 'depth'}); returns True if it was merged into the cloud."""
        stats, fuse = self.stats, self.fuse
        index = stats["frames"] if index is None else index
        stats["frames"] += 1
        xyz, rgb, pix, nrm = frame_points(frame, self.stride, normals=True)
        keep, had_plane = isolate_mask(xyz, pix, frame["depth"].shape, self.rng)
        xyz, rgb, nrm = xyz[keep], rgb[keep], nrm[keep]
        stats["plane_frames"] += had_plane
        if len(xyz) < 500 // self.stride ** 2:
            stats["lost"] += 1
            stats["tracking"] = False
            self.track_log.append((index, "too few points", None))
            return False
        status = "first" if self.prev is None else "tracked"
        if fuse is not None:
            levels = fuse.levels(xyz / 1000)
            if self.prev is not None:
                result = fuse.icp(levels, self.prev, self.motion)
                if result.fitness >= fuse.MIN_FITNESS and plausible_step(result.transformation, xyz):
                    self.motion = result.transformation
                    self.pose = self.pose @ self.motion
                elif self.relock(levels):
                    stats["relocked"] += 1
                    status = "relocked"
                else:
                    stats["lost"] += 1
                    stats["tracking"] = False
                    self.track_log.append((index, "lost", None))
                    return False
            self.prev = levels
            stats["tracking"] = True
        self.track_log.append((index, status, self.pose.copy()))
        if self.last_merged is not None:
            move, turn = turned(np.linalg.inv(self.last_merged) @ self.pose)
            if fuse is not None and move < KEY_MOVE_MM and turn < KEY_TURN_DEG:
                return False
        world = xyz @ self.pose[:3, :3].T + self.pose[:3, 3] * 1000
        world_nrm = nrm @ self.pose[:3, :3].T
        self.cloud.add(world, rgb, world_nrm)
        if self.on_merge is not None:
            self.on_merge(index, world, rgb, world_nrm, self.pose.copy())
        self.last_merged = self.pose.copy()
        self.poses.append((index, self.pose.copy()))
        stats["merged"] += 1
        if not stats["markers"]:
            color = frame["color"] if "color" in frame else yuyv_to_rgb(frame["yuyv"], *frame["depth"].shape[::-1])
            stats["markers"] = [[round(x, 1), round(y, 1)] for x, y, *_ in markers.detect(markers.luma(color), rgb=color)]
        return True

    def relock(self, levels):
        """Tracking failed: try to place the frame against the cloud so far, from the last known pose.

        Works when the scanner comes back near where tracking was lost (like the app's "move back").
        """
        if not len(self.cloud):
            return False
        if self.model_merged != self.stats["merged"]:
            xyz, _, _ = self.cloud.points(min_seen=1)
            self.model, self.model_merged = self.fuse.levels(xyz / 1000), self.stats["merged"]
        result = self.fuse.icp(levels, self.model, self.pose)
        if result.fitness < RELOCK_FITNESS:
            return False
        self.pose, self.motion = result.transformation, np.eye(4)
        return True

    def save(self, directory, min_seen=MIN_SEEN):
        return save(directory, self.cloud, self.poses, self.stats, self.preset, min_seen)


def scan(directory, preset="small", log=print):
    """Merge a recording into a VoxelCloud. Returns (cloud, poses, stats)."""
    scanner = Scanner(preset, log)
    for i, frame in enumerate(fr.read_recording(directory)):
        if scanner.feed(frame, i) and scanner.stats["merged"] % 10 == 1:
            log(f"frame {i:4d}: merged {scanner.stats['merged']:3d}, cubes {len(scanner.cloud):7d}, "
                f"raw points so far {scanner.cloud.raw_points:9d}")
    return scanner.cloud, scanner.poses, scanner.stats


def save(directory, cloud, poses, stats, preset, min_seen=MIN_SEEN):
    """Write cloud.ply, poses.txt and session.json; returns (path, number of points)."""
    directory.mkdir(parents=True, exist_ok=True)
    xyz, rgb, seen = cloud.points(min_seen)
    if not len(xyz) and len(cloud):  # too short a scan for min_seen: keep what there is, and say so
        min_seen = 1
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
