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
from . import prep
from .calib import CAMERA
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

KEY_MOVE_MM, KEY_TURN_DEG = 3.0, 2.0  # merge a frame after this much motion since the last one...
KEY_EVERY = 10  # ...or every 10th tracked frame anyway (a still scanner keeps averaging the view)
MAX_STEP_MM, MAX_STEP_DEG = 30.0, 15.0  # larger jumps between tracked frames are tracking failures
RELOCK_FITNESS = 0.6  # after a loss, a frame must match this share of its points to the cloud so far
RELOCK_MAX_SCALE = 2.0  # a re-found pose may be at most this many normal steps from the last good one
OBJECT_ABOVE_MM = 3.0  # (scene.py) the object starts this far above the table
JOINT_MIN_FITNESS, JOINT_MAX_RMSE_MM = 0.3, 2.0  # accepting a joint-solve pose (markers: see Scanner)
MODEL_EVERY = 5  # target="model": rebuild the model's levels after this many merges
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

    def keep(self, mask):
        """Keep only the cubes in `mask` (e.g. to cut out a table found late)."""
        for name in ("keys", "xyz", "rgb", "n", "nrm", "seen"):
            setattr(self, name, getattr(self, name)[mask])

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
    xyz = CAMERA.backproject(u, v, pc.depth_to_mm(d[v, u]))
    if "color" in frame:
        src_x, src_y = pc.color_source(u, v, w, h)  # only the needed pixels, not the whole image
        rgb = frame["color"][src_y, src_x]
    else:
        rgb = yuyv_pixels(frame["yuyv"], w, *pc.color_source(u, v, w, h))
    if not normals:
        return xyz, rgb, np.stack([u, v], 1)
    return xyz, rgb, np.stack([u, v], 1), pixel_normals(ds, stride, v // stride, u // stride, xyz)


def pixel_normals(ds, stride, rows, cols, xyz):
    """Unit normals at the given cells of an organised depth grid (raw values, every `stride`-th
    pixel), from the neighbouring cells, facing the camera; zero where a neighbour has no depth."""
    hs, ws = ds.shape
    def point(r, c):
        r, c = np.clip(r, 0, hs - 1), np.clip(c, 0, ws - 1)
        raw = ds[r, c]
        p = CAMERA.backproject(c * stride, r * stride, np.where(raw > 0, CAMERA.depth_mm(np.maximum(raw, 1)), np.nan))
        p[(r <= 0) | (r >= hs - 1) | (c <= 0) | (c >= ws - 1)] = np.nan  # no neighbour beyond the edge
        return p
    dx = point(rows, cols + 1) - point(rows, cols - 1)
    dy = point(rows + 1, cols) - point(rows - 1, cols)
    n = np.stack([dx[:, 1] * dy[:, 2] - dx[:, 2] * dy[:, 1], dx[:, 2] * dy[:, 0] - dx[:, 0] * dy[:, 2],
                  dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        n *= np.where((n * xyz).sum(1, keepdims=True) > 0, -1.0, 1.0)  # face the camera (at the origin)
    return np.nan_to_num(n)


def support_plane(xyz, rng, pix=None, shape=None, iterations=60, sample=1500, tries=3):
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
    h, w = shape
    seeds = np.zeros(len(xyz), bool)
    seeds[np.argmin(np.hypot(pix[:, 0] - w / 2, pix[:, 1] - h / 2))] = True
    return connected_piece(xyz, seeds)


def connected_piece(xyz, seeds):
    """Mask of the points connected (through CLUSTER_CELL_MM cells) to any seed point (bool mask)."""
    keys = pack(np.floor(xyz / CLUSTER_CELL_MM).astype(np.int64))
    cells, inv = np.unique(keys, return_inverse=True)
    piece = np.zeros(len(cells), bool)
    piece[inv[seeds]] = True
    frontier = cells[piece]
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


def plausible_step(t, xyz, scale=1.0):
    """Is a frame-to-frame motion believable? Judged at the object's centre, not the camera.

    Turning the object a few degrees in place swings the camera-equivalent pose by centimetres
    (it is ~25 cm away), so a limit on camera movement wrongly rejects it. The object's own centre
    barely moves in that case; a tracking failure moves it a lot. `scale` widens the limits (for
    re-finding after a few lost frames).
    """
    c = xyz.mean(0) / 1000
    moved_mm = np.linalg.norm(t[:3, :3] @ c + t[:3, 3] - c) * 1000
    return moved_mm <= MAX_STEP_MM * scale and turned(t)[1] <= MAX_STEP_DEG * scale


def scale_motion(t, s):
    """The rigid motion `t` (4x4) stretched to `s` times as long (constant velocity), via axis-angle."""
    angle = np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1))
    out = np.eye(4)
    out[:3, 3] = t[:3, 3] * s
    if angle > 1e-9:
        r = t[:3, :3]
        axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(angle))
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        a = angle * s
        out[:3, :3] = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
    return out


def scene_transform(pose):
    """Camera -> world transform with the translation in mm (scene.py works in mm)."""
    out = pose.copy()
    out[:3, 3] = pose[:3, 3] * 1000
    return out


def turned(t):
    """(movement mm, rotation degrees) of a 4x4 rigid transform in metres."""
    return np.linalg.norm(t[:3, 3]) * 1000, np.degrees(np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1)))


class Scanner:
    """Feeds frames one at a time: isolate the object, track it (ICP, if Open3D is present), merge.

    `mode` (scene.MODES): "still" tracks on each frame's piece at the image centre and merges only
    the object above the table plane, which is found once and kept (no flicker); "turntable" tracks
    and merges the object above the kept plane; "free" uses the piece at the image centre for both.
    Measured and dropped for "still": tracking on the whole scene or on the kept object region (a
    large flat table lets ICP slide: synthetic error 4 -> 50 mm), and on the object plus a ring of
    table.

    Options (all on by default; off reproduces the first version, for evaluate.py's baseline):
    clean_depth (prep.clean: "merge" cleans only what is merged and tracks on the uncleaned depth,
    whose edges and steep faces hold flat objects in place; True cleans both), gate_refind
    (re-found poses must be plausible), motion_dt (the next-frame guess follows the real time
    between frames).
    """

    def __init__(self, preset="small", log=print, clean_depth="merge", gate_refind=True, motion_dt=True,
                 mode="still", track_with="joint", target="model", use_features=True, threaded_merge=True):
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
        self.clean_depth, self.gate_refind, self.motion_dt = clean_depth, gate_refind, motion_dt
        from .scene import Scene
        self.scene = Scene(mode, self.rng)
        from .markermap import MarkerMap
        # "joint" (track.solve: shape + markers + motion prior, flat-face check; default), "jointnm"
        # (joint without markers), "icp" (frame-to-frame point-to-plane, Open3D: P2), "markers" (only)
        # Markers are weighted strictly (1 mm) and noisy ones are dropped inside the solve (3 sigma):
        # measured better than loose weighting (3 mm) even with the estimated colour registration,
        # where loose let marker noise pull poses off and the map drift (tilt doubling 16% vs 78%).
        # The marker check also catches sliding: on the tilt scan, poses whose shape fits well but
        # whose markers disagree are the flat-face slides; dropping the check took doubling 7% -> 26%.
        self.marker_sigma, self.marker_accept = 1.0, 2.0
        self.track_with, self.marker_map = track_with, MarkerMap(add_mm=3.0)
        # joint solve against the "model" (cloud so far; default) or the previous "frame". Measured:
        # model + markers halved synthetic error (3.4 -> 1.7 mm, hard path 6.1 -> 1.5); the model
        # WITHOUT markers drifts (synthetic 25 mm): markers keep the model honest.
        self.target = target
        self.model_levels, self.model_at = None, -1
        self.features = None  # features module when OpenCV is available and use_features is on
        if use_features and track_with in ("joint", "jointnm"):
            try:
                from . import features
                self.features = features
            except ImportError:
                pass
        self.prev_feat, self.feature_starts = None, 0  # last tracked frame's ORB; frames started from ORB
        self.cue = {}  # what held the last frame: markers used, ORB start, flat (shape alone slides)
        self.keyframes = []  # merged frames: (index, pose, points mm (camera, 2 mm grid), ORB, markers mm)
        import concurrent.futures
        import threading
        self.cloud_lock = threading.Lock()
        self.merger = concurrent.futures.ThreadPoolExecutor(1) if threaded_merge else None
        self.pending = []
        self.track_info = []  # per tracked frame: (index, info from track.solve); for evaluation
        self.last_t, self.motion_s, self.lost_run = None, None, 0  # time of the last tracked frame, of `motion`
        self.track_log = []  # per frame: (index, status, pose or None); for evaluation
        self.on_merge = None  # optional callback(index, world_xyz_mm, rgb, world_normals, pose)

    def feed(self, frame, index=None):
        """Process one frame set ({'color', 'depth'}); returns True if it was merged into the cloud."""
        stats, fuse = self.stats, self.fuse
        index = stats["frames"] if index is None else index
        stats["frames"] += 1
        if self.clean_depth is True:  # track on cleaned depth as well
            frame = self.cleaned(frame)
        shape = frame["depth"].shape
        all_xyz, all_rgb, all_pix = frame_points(frame, self.stride)  # normals only for merging (below)
        scene = self.scene
        if scene.mode == "turntable":
            scene.observe(all_xyz, all_pix, shape, np.eye(4))
            keep = scene.object_mask(all_xyz, all_pix, shape, np.eye(4))
        else:
            keep = isolate_mask(all_xyz, all_pix, shape, self.rng)[0]
        xyz, rgb, pix = all_xyz[keep], all_rgb[keep], all_pix[keep]
        if len(xyz) < 500 // self.stride ** 2:
            stats["lost"] += 1
            stats["tracking"] = False
            self.lost_run += 1
            self.track_log.append((index, "too few points", None))
            return False
        t = frame.get("t")
        guess = self.motion
        if self.motion_dt and t is not None and self.last_t is not None and self.motion_s:
            guess = scale_motion(self.motion, (t - self.last_t) / self.motion_s)
        status = "first" if self.prev is None else "tracked"
        if self.track_with == "markers":
            if not self.marker_step(frame, guess, index):
                return False
            status = "first" if self.prev is None else "tracked"
            self.prev = True
        elif fuse is not None:
            joint = self.track_with in ("joint", "jointnm")
            if joint:
                from . import track
                levels = track.levels(xyz / 1000)
                marker_pts = self.frame_markers(frame) if self.track_with == "joint" else np.empty((0, 3))
            else:
                levels = fuse.levels(xyz / 1000)
            feat = None
            if joint and self.features is not None:  # colour features: a starting pose for big movements
                feat = self.features.extract(self.features.luma_of(frame), frame["depth"],
                                             None if scene.mode == "still" else pix)
                if self.prev is not None and self.prev_feat is not None:
                    found, _ = self.features.relative_pose(feat, self.prev_feat, self.rng)
                    self.cue = {"features": found is not None and plausible_step(found, xyz)}
                    if self.cue["features"]:
                        guess = found
                        self.feature_starts += 1
            if self.prev is not None:
                if joint:
                    step, ok = self.joint_step(levels, xyz, marker_pts, guess, index)
                else:
                    result = fuse.icp(levels, self.prev, guess)
                    step = result.transformation
                    ok = result.fitness >= fuse.MIN_FITNESS and plausible_step(step, xyz)
                if ok:
                    self.motion = step
                    self.motion_s = (t - self.last_t) if (t is not None and self.last_t is not None) else None
                    self.pose = self.pose @ self.motion
                elif self.relock(*self.relock_source(levels, xyz, all_xyz, all_pix, shape)):
                    stats["relocked"] += 1
                    status = "relocked"
                else:
                    stats["lost"] += 1
                    stats["tracking"] = False
                    self.lost_run += 1
                    self.track_log.append((index, "lost", None))
                    return False
            self.last_t, self.lost_run = t, 0
            self.prev = levels
            stats["tracking"] = True
            if joint and len(marker_pts):
                self.marker_map.add(marker_pts, self.pose)
            self.prev_feat = feat
        self.track_log.append((index, status, self.pose.copy()))
        if scene.mode == "still":  # the object to merge: above the kept table plane, in its region
            to_world = scene_transform(self.pose)  # (every tracked frame: the region grows per frame)
            scene.observe(all_xyz, all_pix, shape, to_world)
            pix = all_pix[scene.object_mask(all_xyz, all_pix, shape, to_world)]
        stats["plane_frames"] += scene.plane is not None
        self.since_merge = getattr(self, "since_merge", 0) + 1
        if self.last_merged is not None:
            move, turn = turned(np.linalg.inv(self.last_merged) @ self.pose)
            if fuse is not None and move < KEY_MOVE_MM and turn < KEY_TURN_DEG and self.since_merge < KEY_EVERY:
                return False
        self.since_merge = 0
        if self.prev is not None and self.prev is not True:  # keyframe for Optimise / continuing a scan
            self.keyframes.append((index, self.pose.copy(), np.asarray(self.prev[-1].points) * 1000,
                                   self.prev_feat, marker_pts if joint else np.empty((0, 3))))
        self.last_merged = self.pose.copy()
        self.poses.append((index, self.pose.copy()))
        stats["merged"] += 1
        job = (frame, index, self.pose.copy(), pix)
        if self.merger is None:
            self.merge(*job)
        else:  # merging runs beside tracking the next frame (numpy and Open3D release the GIL)
            self.pending.append(self.merger.submit(self.merge, *job))
        return True

    def merge(self, frame, index, pose, pix):
        """Merge one tracked frame's object pixels (cleaned, with normals) into the cloud."""
        shape = frame["depth"].shape
        kept = np.zeros(shape, bool)  # merge these pixels (cleaned, if cleaning)
        kept[pix[:, 1], pix[:, 0]] = True
        merge_frame = self.cleaned(frame, pix) if self.clean_depth == "merge" else frame
        cxyz, crgb, cpix, cnrm = frame_points(merge_frame, self.stride, normals=True)
        sel = kept[cpix[:, 1], cpix[:, 0]]
        mxyz, mrgb, mnrm = cxyz[sel], crgb[sel], cnrm[sel]
        world = mxyz @ pose[:3, :3].T + pose[:3, 3] * 1000
        world_nrm = mnrm @ pose[:3, :3].T
        with self.cloud_lock:
            self.cloud.add(world, mrgb, world_nrm)
        if self.on_merge is not None:
            self.on_merge(index, world, mrgb, world_nrm, pose)
        if not self.stats["markers"]:
            color = frame["color"] if "color" in frame else yuyv_to_rgb(frame["yuyv"], *shape[::-1])
            self.stats["markers"] = [[round(x, 1), round(y, 1)] for x, y, *_ in markers.detect_half(color)]

    def flush(self):
        """Wait for merges still running in the background (before reading or saving the cloud)."""
        for job in self.pending:
            job.result()
        self.pending = []

    def joint_step(self, levels, xyz, marker_pts, guess, index):
        """One frame with track.solve (shape + markers + prior), against the previous frame or the
        model (the cloud so far, rebuilt every MODEL_EVERY merges). Returns (step from the previous
        pose, accepted)."""
        from . import track
        use_model = self.target == "model" and self.model_ready()
        ref = np.eye(4) if use_model else self.pose  # the target's frame -> world
        pairs = None
        if len(marker_pts) and len(self.marker_map):
            matched = self.marker_map.match_predicted(marker_pts, self.pose @ guess)
            if matched:
                to_ref = np.linalg.inv(scene_transform(ref))  # world (mm) -> target frame (mm)
                dst = self.marker_map.pos[[m for _, m in matched]] @ to_ref[:3, :3].T + to_ref[:3, 3]
                pairs = (marker_pts[[f for f, _ in matched]], dst)
        radius = float(np.percentile(np.linalg.norm(xyz - xyz.mean(0), axis=1), 90)) if len(xyz) else 60.0
        init = self.pose @ guess if use_model else guess
        found, info = track.solve(levels, self.model_levels if use_model else self.prev, init, prior=init,
                                  marker_pairs=pairs, radius_mm=radius, marker_sigma_mm=self.marker_sigma)
        step = np.linalg.inv(self.pose) @ found if use_model else found
        self.track_info.append((index, info))
        flat = info["degeneracy"] is not None and info["degeneracy"] < track.DEGENERATE and not info["markers_used"]
        self.stats["flat"] = self.stats.get("flat", 0) + flat
        self.cue = {"markers": info["markers_used"], "features": self.cue.get("features", False), "flat": flat}
        ok = (info["fitness"] >= JOINT_MIN_FITNESS and info["rmse_mm"] is not None
              and info["rmse_mm"] <= JOINT_MAX_RMSE_MM and plausible_step(step, xyz)
              and (info["marker_rmse_mm"] is None or info["marker_rmse_mm"] <= self.marker_accept))
        return step, ok

    def model_ready(self):
        """Prepare the model target (levels of the cloud so far, world); False until it has enough points."""
        from . import fuse
        if self.stats["merged"] < 2:
            return False
        if self.model_levels is None or self.stats["merged"] - self.model_at >= MODEL_EVERY:
            self.flush()  # the model must not depend on how far background merges have got (replays match)
            with self.cloud_lock:
                xyz, _, _ = self.cloud.points(min_seen=1)
            if len(xyz) < 500:
                return self.model_levels is not None
            self.model_levels, self.model_at = fuse.levels(xyz / 1000), self.stats["merged"]
        return True

    def frame_markers(self, frame):
        """Camera-frame centres (mm, (N, 3)) of the markers in a frame; uses frame["markers2d"]
        (pre-detected colour pixels) when given."""
        found = frame.get("markers2d")
        if found is None:
            from .features import luma_of
            found = markers.detect_fast(luma_of(frame), frame.get("color"))
        pts = [loc[0] for loc in markers.locate(frame["depth"], found) if loc is not None]
        return np.array(pts).reshape(-1, 3)

    def marker_step(self, frame, guess, index):
        """Marker-only tracking (for measuring what markers alone can do): pose from the marker map."""
        stats, mmap = self.stats, self.marker_map
        pts = self.frame_markers(frame)
        t = frame.get("t")
        if self.prev is not None:
            pairs = mmap.match_predicted(pts, self.pose @ guess) if len(pts) >= 3 else []
            pose = mmap.pose_from(pts, pairs) if len(pairs) >= 3 else mmap.match_global(pts)
            if pose is None:
                stats["lost"] += 1
                stats["tracking"] = False
                self.lost_run += 1
                self.track_log.append((index, "lost", None))
                return False
            self.motion = np.linalg.inv(self.pose) @ pose
            self.motion_s = (t - self.last_t) if (t is not None and self.last_t is not None) else None
            self.pose = pose
        mmap.add(pts, self.pose)
        self.last_t, self.lost_run = t, 0
        stats["tracking"] = True
        return True

    def relock_source(self, levels, xyz, all_xyz, all_pix, shape):
        """Points to re-find with. The cloud holds only the object, so in "still" mode (tracking on
        the whole scene) use the frame's object piece, not the scene."""
        if self.scene.mode != "still":
            return levels, xyz
        obj = isolate_mask(all_xyz, all_pix, shape, self.rng)[0]
        if obj.sum() < 50:
            return levels, xyz
        return self.fuse.levels(all_xyz[obj] / 1000), all_xyz[obj]

    def cleaned(self, frame, pix=None):
        """The frame with prep.clean applied to the pixels tracking samples (every stride-th); with
        `pix`, only inside their bounding box (plus a margin): the merge needs nothing else."""
        s, d = self.stride, frame["depth"]
        out = np.zeros_like(d)
        r0, r1, c0, c1 = 0, d.shape[0], 0, d.shape[1]
        if pix is not None and len(pix):
            m = 8 * s
            c0, c1 = max(0, pix[:, 0].min() - m) // s * s, min(d.shape[1], pix[:, 0].max() + m + 1)
            r0, r1 = max(0, pix[:, 1].min() - m) // s * s, min(d.shape[0], pix[:, 1].max() + m + 1)
        out[r0:r1:s, c0:c1:s] = prep.clean(d[r0:r1:s, c0:c1:s], s, origin=(c0, r0))
        return dict(frame, depth=out)

    def relock(self, levels, xyz):
        """Tracking failed: try to place the frame against the cloud so far, from the last known pose.

        Works when the scanner comes back near where tracking was lost (like the app's "move back").
        With gate_refind, the re-found pose must be a believable step from the last good pose
        (limits widen with the frames lost, up to RELOCK_MAX_SCALE): the first version accepted
        jumps of 120-190 mm, which merged shifted copies of the object.
        """
        if not len(self.cloud):
            return False
        if self.model_merged != self.stats["merged"]:
            xyz, _, _ = self.cloud.points(min_seen=1)
            self.model, self.model_merged = self.fuse.levels(xyz / 1000), self.stats["merged"]
        result = self.fuse.icp(levels, self.model, self.pose)
        if result.fitness < RELOCK_FITNESS:
            return False
        if self.gate_refind:
            jump = np.linalg.inv(self.pose) @ result.transformation  # current camera -> last good camera
            if not plausible_step(jump, xyz, min(self.lost_run + 1, RELOCK_MAX_SCALE)):
                return False
        self.pose, self.motion, self.motion_s = result.transformation, np.eye(4), None
        return True

    def save(self, directory, min_seen=MIN_SEEN):
        self.flush()
        self.trim_table()
        out = save(directory, self.cloud, self.poses, self.stats, self.preset, min_seen)
        self.save_state(directory)
        return out

    def save_state(self, directory):
        """What Optimise and continuing a scan need: per-frame tracking log, keyframes, marker map and
        the cloud's sums (track.csv, keyframes.npz, markers.json, cloud_state.npz)."""
        info = dict(self.track_info)
        with open(directory / "track.csv", "w") as f:
            f.write("index,status,fitness,rmse_mm,degeneracy,markers_used\n")
            for i, status, _ in self.track_log:
                d = info.get(i, {})
                f.write(f"{i},{status},{d.get('fitness', '')},{d.get('rmse_mm', '')},{d.get('degeneracy', '')},"
                        f"{d.get('markers_used', '')}\n")
        kf = self.keyframes
        if kf:
            pts = [k[2] for k in kf]
            feat_pts = [k[3][0] if k[3] is not None else np.empty((0, 3)) for k in kf]
            feat_desc = [k[3][1] if k[3] is not None and k[3][1] is not None else np.empty((0, 32), np.uint8) for k in kf]
            mks = [k[4] for k in kf]
            count = lambda arrays: np.cumsum([0] + [len(a) for a in arrays])
            np.savez_compressed(directory / "keyframes.npz", index=np.array([k[0] for k in kf]),
                                pose=np.array([k[1] for k in kf]), points=np.concatenate(pts), points_at=count(pts),
                                feat_points=np.concatenate(feat_pts), feat_desc=np.concatenate(feat_desc),
                                feat_at=count(feat_pts), markers=np.concatenate(mks), markers_at=count(mks))
        (directory / "markers.json").write_text(json.dumps(
            {"positions_mm": self.marker_map.pos.round(3).tolist(), "sightings": self.marker_map.count.tolist()}))
        c = self.cloud
        np.savez_compressed(directory / "cloud_state.npz", point_mm=c.point_mm, keys=c.keys, xyz=c.xyz, rgb=c.rgb,
                            n=c.n, nrm=c.nrm, seen=c.seen, frames=c.frames, raw_points=c.raw_points)

    def trim_table(self):
        """Drop cubes merged before the table plane was settled: on/below it or outside the region."""
        self.flush()
        scene = self.scene
        if scene.mode != "still" or scene.plane is None or scene.axis is None or not len(self.cloud):
            return
        c = self.cloud.xyz / self.cloud.n[:, None]
        n = scene.plane[0]
        h = scene.height(c)
        flat = c - np.outer((c - scene.axis) @ n, n)
        keep = (h > OBJECT_ABOVE_MM) & (np.linalg.norm(flat - scene.axis, axis=1) <= scene.radius)
        self.cloud.keep(keep)


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
