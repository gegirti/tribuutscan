"""Merge many depth frames into one compact point cloud: the "point distance" rule real scanners use.

    python3 -m tribuutscan.cloud captures/<stamp>-rec [small|medium|large] [--keep-base]
    .venv/bin/python -m tribuutscan.cloud ...      # with Open3D: tracks motion (on the GPU when there is one;
                                                   # TRIBUUTSCAN_GPU=0: on the CPU)
    .venv/bin/python -m tribuutscan.cloud captures/<stamp>-rec --optimise --mesh   # then Optimise, fusion.py

Space is cut into cubes of side `point distance`. Every point that lands in a cube is averaged into
that cube's one point, so the cloud grows with the object's surface, not with scan time. A cube
must be seen in `min_seen` frames to be kept, which drops noise and flying pixels.

Each frame is first cut down to the object: the depth window, then the table or turntable plane
(if the object stands on it), then the connected piece at the image centre.

Writes captures/<stamp>-scan/: cloud.ply (metres, x right, y up, z toward the scanner),
poses.txt (merged frames), frame_poses.txt (every tracked frame), labels.npz, session.json (with the
camera model's source and fingerprint, and `sources`: the recordings the frame indices point into,
so a scan can be fused again from its raw frames).
"""
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import frames as fr
from . import markers
from . import pointcloud as pc
from . import prep
from .calib import CAMERA, CLOSE
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
MIN_SEEDS = 20  # region-core points (scene.Scene.core) that place the tracking source instead of the centre
OUTPUT_CELL_MM = 3.0  # connectivity of the saved cloud: cubes further apart than this are not connected

KEY_MOVE_MM, KEY_TURN_DEG = 3.0, 2.0  # merge a frame after this much motion since the last one...
KEY_EVERY = 10  # ...or every 10th tracked frame anyway (a still scanner keeps averaging the view)
MERGE_MIN_S = 0.25  # ...into the point cloud at most one per this much frame time (the GPU tracker's volume takes
    # every such frame: capping it too made synthetic tracking 0.35 -> 2.6 mm worse): at 10-30 fps nearly every
    # frame moved 3 mm, and the merges (point cloud + keyframe ORB, ~190 ms each) took the tracking thread's CPU
MAX_STEP_MM, MAX_STEP_DEG = 30.0, 15.0  # larger jumps between tracked frames are tracking failures
RELOCK_FITNESS = 0.6  # after a loss, a frame must match this share of its points to the cloud so far
RELOCK_MAX_SCALE = 2.0  # a re-found pose may be at most this many normal steps from the last good one
GUESS_MAX_S = 0.3  # the motion guess extrapolates the last motion over at most this long (lost frames:
                   # it ran on and on, 7 -> 177 mm at the object over 20 lost frames of the hand scan)
REFIND_KEYFRAMES, REFIND_TRIES = 16, 3  # refind_anywhere: keyframes compared, best ones tried...
REFIND_MIN_INLIERS = 50  # ...and the ORB inliers a pose found anywhere needs (the hand scan's had 141-245)...
REFIND_MIN_AWAY = 25  # ...this many of them on the still scene around the object, not on it: the synthetic plate
                      # (the same markers on both faces, a plain table) was re-found mirrored, 180 degrees off
REFIND_MM_S, REFIND_DEG_S, REFIND_SLACK = 300.0, 60.0, (50.0, 15.0)  # ...and must be reachable in the time lost:
    # a full circle round the object in 6 s is a fast walk; a faster turn only delays the re-find. The plate's
    # mirrored re-find turned 136 degrees in 1.35 s (truly ~45); the real ones 18-31 degrees after 19-27 s
REFIND_HOLD = 2  # frames tracked after a pose found anywhere before merging resumes
SHAPE_EVERY_S = 0.25  # refind_shape: at most one try per this much frame time while lost (~50-150 ms each)
MARKER_CHECK_MM, MARKER_CHECK_DEG, MARKER_CHECK_MIN = 5.0, 2.0, 3  # marker_check: a solve this far from the markers' own
    # placement is redone from it, and must then hold this many of them
TURN_AXIS_MIN_DEG = 20.0  # turn_axis: poses turned at least this far from the first one count
GAP_S = 0.5  # the first frame tracked after this long without one (lost frames, or none came) is checked by shape:
    # its solve began at the last good pose, and a turn about a vertical axis hides in horizontal faces (the
    # synthetic block on a platform, 3 s left out: "tracked" 37 degrees off)
ROOM_EVERY_S, ROOM_GRID_PX, ROOM_POINTS = 1.0, 8, 5000  # check_room: once per second of frame time, on the room's
    # pixels on an 8-pixel grid (the same pixels in every frame: a still room's samples stay where they were), at most
    # ROOM_POINTS of a keyframe's looked for in a later one. At every keyframe on the full stride-4 room it cost the
    # live worker 10-25% of its frame rate on the hand-held hand scan
ROOM_SHIFT_MM, ROOM_MIN, ROOM_TOL_MM, ROOM_CLEAR, ROOM_VERDICTS, ROOM_SETTLED = 10.0, 300, 4.0, 0.5, 3, 5
    # points the tracked motion moves ROOM_SHIFT_MM or more count (ROOM_MIN of them for a verdict); a point is where a
    # hypothesis puts it within ROOM_TOL_MM; a verdict needs ROOM_CLEAR of the points and twice the other
    # hypothesis's share; ROOM_VERDICTS "still" ones set object_turns (the room stands while the object turns: a
    # platform, the scanner standing); ROOM_SETTLED "moved" ones and no "still" end the test (a hand-held scan: each
    # check costs ~80 ms, ~4.5 ms a frame over a scan)
MARKER_IDLE_FRAMES, MARKER_IDLE_EVERY = 30, 8  # no markers seen for this long: look on every 8th frame only (a
    # search costs ~10 ms; at every 4th after 15 it was still 2.6 ms of a hand-held frame)
GPU_MAX_RMSE_MM, GPU_RMSE_Z_MM = 1.0, 450.0  # the GPU solve's residual limit, growing with the piece's distance
    # squared beyond 450 mm (depth noise does; long range reaches 720 mm: 2.6 mm there). synthhard's jerk frame
    # was taken 10 mm off with 1.36 mm, fused, and tracking stayed ~4 mm off; good frames on every dataset (hand
    # at 33-50 cm included): at most 0.46 mm. (The fitness says how much of the view the model covers so far,
    # not how well it fits: gating fusion on it starved the synthetic boxes' models from frame 1 on.)
FLOOR_MAX_DEG = 2.5  # a taken pose tilts the kept base at most this much from the frame's own table...
FLOOR_OTHER_DEG, FLOOR_MIN_SPAN_MM = 15.0, 150.0  # ...a plane further off, or narrower, says nothing
OBJECT_ABOVE_MM = 3.0  # (scene.py) the object starts this far above the table
JOINT_MIN_FITNESS, JOINT_MAX_RMSE_MM = 0.3, 2.0  # accepting a joint-solve pose (markers: see Scanner)
MODEL_EVERY = 5  # the CPU tracker's model: its levels are rebuilt after this many merges
LABEL_STEP = 8  # per-frame labels (and the viewer's object mask) at 1/8 of the frame size (160 x 100)
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

    def subset(self, mask):
        """A new cloud with only the cubes in `mask` (this one is left as it is)."""
        out = VoxelCloud(self.point_mm)
        for name in ("keys", "xyz", "rgb", "n", "nrm", "seen"):
            setattr(out, name, getattr(self, name)[mask])
        out.frames, out.raw_points = self.frames, self.raw_points
        return out

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


def frame_points(frame, stride=1, normals=False, colour=True):
    """Camera-frame points (mm; x right, y down, z forward), colours, and pixel coords of one frame.

    `stride` takes every n-th pixel in both directions (more points than cubes is wasted work).
    The frame holds either decoded "color" or the raw "yuyv" buffer (then only the needed
    pixels are converted, which is much cheaper live). With `normals`, also returns unit normals
    from the neighbouring pixels, facing the camera (zero where a neighbour has no depth). Without
    `colour` the colours are None (tracking needs none: the lookup took 6 of a frame's 7 ms).
    """
    if stride == 4 and "depth4" in frame:  # a live frame: its every-4th pixel decoded alone (depth.LazyFrame)
        ds, (h, w) = frame["depth4"], frame["shape"]
    else:
        d = frame["depth"]
        h, w = d.shape
        ds = d[::stride, ::stride]
    valid = ds > 0
    v, u = np.nonzero(valid)
    z = pc.depth_to_mm(ds[v, u])
    v, u = v * stride, u * stride
    xyz = CAMERA.backproject(u, v, z)
    rgb = None
    if colour:
        src_x, src_y = pc.color_source(u, v, w, h, z)  # only the needed pixels; parallax-aware when calibrated
        rgb = frame["color"][src_y, src_x] if "color" in frame else yuyv_pixels(frame["yuyv"], w, src_x, src_y)
    if not normals:
        return xyz, rgb, np.stack([u, v], 1)
    return xyz, rgb, np.stack([u, v], 1), pixel_normals(ds, stride, v // stride, u // stride, xyz)


def frame_shape(frame):
    """(height, width) of a frame set, without decoding a live frame's depth (depth.LazyFrame)."""
    return tuple(frame["shape"]) if "shape" in frame else frame["depth"].shape


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


def support_plane(xyz, rng, pix=None, shape=None, iterations=60, sample=1500, tries=3,
                  min_share=PLANE_MIN_SHARE, min_cm2=PLANE_MIN_CM2, accept=None):
    """The table/turntable plane (normal n, offset c with n.p + c = 0, n toward the scanner), or None.

    A plane counts only if it holds `min_share` of the points AND most other points lie on the
    scanner's side of it (the object stands on it) AND the scanner is not looking along it (an
    object's side wall seen edge-on is not a table). With `pix`/`shape` given, it must also run out
    of the picture: a table does, a flat lid or face in the middle of the view does not. It must
    cover `min_cm2`: a plane slicing a curved surface only catches thin bands. Finally `accept(n, c)`,
    if given, must agree (scene.py: the object at the centre stands on it).
    A plane that fails is set aside and the next largest plane is tried, up to `tries` planes.
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
        if plane is None or inliers.sum() < min_share * len(pts):
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
                and plane_area_cm2(xyz, n, c) >= min_cm2 and (accept is None or accept(n, c)):
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


def connected_piece(xyz, seeds, cell_mm=CLUSTER_CELL_MM):
    """Mask of the points connected (through cells of `cell_mm`) to any seed point (bool mask)."""
    keys = pack(np.floor(xyz / cell_mm).astype(np.int64))
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


def isolate_mask(xyz, pix, shape, rng, seeds=None, known=None, plane=None):
    """Mask of the object's points (support plane dropped, piece at the centre kept), and that plane
    ((n, c), or None). With `seeds` (bool mask: where the object should be, scene.Scene.core), the piece
    connected to the seeds left after the plane cut is kept instead, when there are MIN_SEEDS of them.
    `known`: a function giving a plane to cut by when the frame shows none of its own (the kept base
    where the pose guess puts it: without a cut the piece took in the whole floor, and the hand scan's
    frames 46-49 matched the model at 10-28%). `plane`: the table, already found (scene.Scene.
    table_in_camera): no search of the frame's own (support_plane, 7 ms a frame)."""
    keep = np.ones(len(xyz), bool)
    if plane is None:
        plane = support_plane(xyz, rng, pix, shape)
    cut = plane if plane is not None or known is None else known()
    if cut is not None:
        keep = np.abs(xyz @ cut[0] + cut[1]) >= PLANE_TOL_MM
    if keep.any():
        idx = np.nonzero(keep)[0]
        if seeds is not None and seeds[idx].sum() >= MIN_SEEDS:
            piece = connected_piece(xyz[idx], seeds[idx])
        else:
            piece = centre_piece(xyz[idx], pix[idx], shape)
        keep[idx[~piece]] = False
    return keep, plane


def table_normal(xyz, plane):
    """(unit normal, least squares, facing the camera; narrower in-plane extent in mm, +-2 sigma) of the
    points within PLANE_TOL_MM of a frame's table `plane`, or None."""
    if plane is None:
        return None
    pts = xyz[np.abs(xyz @ plane[0] + plane[1]) < PLANE_TOL_MM]
    if len(pts) < 200:
        return None
    pts = pts[::max(1, len(pts) // 5000)]
    centre = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - centre).T))
    n = v[:, 0] if v[:, 0] @ centre < 0 else -v[:, 0]
    return n, 4 * float(np.sqrt(max(w[1], 0.0)))


def cleaned_frame(frame, stride, pix=None):
    """The frame with prep.clean applied to every `stride`-th pixel; with `pix`, only inside their
    bounding box (plus a margin): a merge needs nothing else."""
    s, d = stride, frame["depth"]
    out = np.zeros_like(d)
    r0, r1, c0, c1 = 0, d.shape[0], 0, d.shape[1]
    if pix is not None and len(pix):
        m = 8 * s
        c0, c1 = max(0, pix[:, 0].min() - m) // s * s, min(d.shape[1], pix[:, 0].max() + m + 1)
        r0, r1 = max(0, pix[:, 1].min() - m) // s * s, min(d.shape[0], pix[:, 1].max() + m + 1)
    out[r0:r1:s, c0:c1:s] = prep.clean(d[r0:r1:s, c0:c1:s], s, origin=(c0, r0))
    return dict(frame, depth=out)


def merge_points(frame, pix, step, stride, clean=True):
    """A merge's camera points (mm), colours and camera normals: the frame's pixels inside the
    step x step blocks of `pix`, every `stride`-th, cleaned first (prep.clean) when `clean`."""
    h, w = frame["depth"].shape
    kept = np.zeros(((h + step - 1) // step, (w + step - 1) // step), bool)  # merge these blocks
    kept[pix[:, 1] // step, pix[:, 0] // step] = True
    cxyz, crgb, cpix, cnrm = frame_points(cleaned_frame(frame, stride, pix) if clean else frame, stride, normals=True)
    sel = kept[cpix[:, 1] // step, cpix[:, 0] // step]
    return cxyz[sel], crgb[sel], cnrm[sel]


def marker_list(frame):
    """The markers in a frame's colour image, as stats["markers"] lists them."""
    h, w = frame["depth"].shape
    color = frame["color"] if "color" in frame else yuyv_to_rgb(frame["yuyv"], w, h)
    return [[round(x, 1), round(y, 1)] for x, y, *_ in markers.detect_half(color)]


def isolate(xyz, rgb, pix, shape, rng):
    """Cut a frame down to the object: drop the support plane, keep the piece at the centre."""
    keep, plane = isolate_mask(xyz, pix, shape, rng)
    return xyz[keep], rgb[keep], plane is not None


def fits_better(a, b):
    """Does the solve info `a` fit the frame clearly better than `b`: more of it matched, or as much with a
    clearly smaller residual?"""
    if a["rmse_mm"] is None:
        return False
    if b["rmse_mm"] is None:
        return True
    return a["fitness"] >= b["fitness"] + 0.05 or (a["fitness"] >= b["fitness"] - 0.02 and a["rmse_mm"] <= 0.8 * b["rmse_mm"])


def same_pose(a, b, xyz, mm=3.0, deg=1.5):
    """Do two steps (from the same pose) place the frame's piece within `mm` (at its centre) and `deg`?"""
    d = np.linalg.inv(a) @ b
    c = xyz.mean(0) / 1000
    return np.linalg.norm(d[:3, :3] @ c + d[:3, 3] - c) * 1000 <= mm and turned(d)[1] <= deg


def room_verdict(still_pts, moved_pts, room, tol_mm=ROOM_TOL_MM):
    """Scanner.check_room's verdict: "still" if the room's points are found again where they were in the view
    (still_pts) rather than where the tracked motion would have taken them (moved_pts), "moved" for the
    opposite, None when neither clearly wins. room: this frame's room points (camera mm). Only points the two
    tell apart count: a point that lands on the room as it was says nothing (a plane the motion maps onto
    itself, as a table under a turntable: with it "moved" matched half the points of a still room)."""
    import open3d as o3d
    cloud = lambda pts: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, float)))
    near = lambda a, b: np.asarray(cloud(a).compute_point_cloud_distance(cloud(b))) < tol_mm
    tell = ~near(moved_pts, still_pts) & ~near(still_pts, moved_pts)
    if tell.sum() < ROOM_MIN:
        return None
    still, moved = float(near(still_pts[tell], room).mean()), float(near(moved_pts[tell], room).mean())
    if still >= ROOM_CLEAR and still >= 2 * moved:
        return "still"
    if moved >= ROOM_CLEAR and moved >= 2 * still:
        return "moved"
    return None


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


def view_mask(view, step=LABEL_STEP):
    """What a frame's pixels were, from Scanner.last_view: uint8 at 1/step size, 0 other (background,
    or not seen), 1 base, 2 object; None if the frame was not tracked."""
    if view is None:
        return None
    pix, obj, base, shape = view
    out = np.zeros((shape[0] // step, shape[1] // step), np.uint8)
    for value, mask in ((1, base), (2, obj)):
        p = pix[mask] // step
        out[np.minimum(p[:, 1], out.shape[0] - 1), np.minimum(p[:, 0], out.shape[1] - 1)] = value
    return out


def write_poses(path, poses):
    with open(path, "w") as f:
        for i, p in poses:
            f.write(f"{i} " + " ".join(f"{v:.6f}" for v in p.ravel()) + "\n")


def turn_axis(poses, min_deg=TURN_AXIS_MIN_DEG):
    """The axis a platform turned the object about (unit, world, pointing the way the cameras' up does) over
    the tracked camera poses, or None: turned less than `min_deg`. Each pose relative to the first is a turn
    about it (rotations of 150 degrees and more, where the axis is ill-defined, left out)."""
    if len(poses) < 10:
        return None
    axes = []
    for p in poses[::5]:
        r = (p @ np.linalg.inv(poses[0]))[:3, :3]
        angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
        if np.radians(min_deg) <= angle <= np.radians(150):
            a = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
            axes.append(a / np.linalg.norm(a))
    if not axes:
        return None
    axes = np.array(axes)
    axes[axes @ axes[0] < 0] *= -1
    axis = axes.mean(0)
    axis /= np.linalg.norm(axis)
    up = -np.mean([p[:3, 1] for p in poses], axis=0)  # (a camera's y axis points down)
    return axis if axis @ up >= 0 else -axis


def crop_axes(up):
    """The axes of the 3D page's crop box (rows, in cloud.ply's axes: x right, y up, z toward the first camera)
    for `up` (world: the base's normal or a platform's turn axis): y along it, so the box's faces cut level with
    the table, x the first camera's x laid square to it; without `up` the first camera's own."""
    if up is None:
        return np.eye(3)
    y = FLIP @ up
    x = np.array([1.0, 0.0, 0.0]) - y[0] * y
    if np.linalg.norm(x) < 0.1:  # (up along the camera's x: the scanner held on its side)
        x = np.array([0.0, 0.0, 1.0]) - y[2] * y
    x /= np.linalg.norm(x)
    return np.stack([x, y, np.cross(x, y)])


def saved_crop_axes(directory, session):
    """Scanner.crop_axes of a saved scan, from its scene.json and tracked poses (for sessions saved before
    session.json noted them)."""
    directory = Path(directory)
    scene = json.loads((directory / "scene.json").read_text()) if (directory / "scene.json").exists() else {}
    up = None
    if scene.get("mode", "still") == "still" and scene.get("plane"):
        up = np.array(scene["plane"][0])
    elif session.get("object_turns"):
        path = next((p for p in (directory / "frame_poses.txt", directory / "poses.txt") if p.exists()), None)
        up = None if path is None else turn_axis([p for _, p in read_poses(path)])
    return crop_axes(up)


def read_poses(path):
    """[(index, 4x4 pose)] from a poses.txt-format file."""
    text = Path(path).read_text()
    rows = np.array([line.split() for line in text.splitlines() if line.strip()], float).reshape(-1, 17)
    return [(int(r[0]), r[1:].reshape(4, 4)) for r in rows]


def load_npz(path):
    """{name: array} of an .npz, each read once: indexing the open file decompresses the whole array
    again every time (a slice of it kept that whole copy alive: restore once grew past 8 GB)."""
    with np.load(path) as f:
        return {name: f[name] for name in f.files}


def blend(a, b, t):
    """The rigid transform a fraction t of the way from a (t = 0) to b (t = 1): rotation along the
    shortest arc, translation along a straight line."""
    d = a[:3, :3].T @ b[:3, :3]
    angle = np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))
    r = a[:3, :3]
    if angle > 1e-12:
        w = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) / (2 * np.sin(angle))
        k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        r = r @ (np.eye(3) + np.sin(t * angle) * k + (1 - np.cos(t * angle)) * k @ k)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, (1 - t) * a[:3, 3] + t * b[:3, 3]
    return out


def scene_margin():
    from .scene import REGION_MARGIN_MM
    return REGION_MARGIN_MM


def scene_transform(pose):
    """Camera -> world transform with the translation in mm (scene.py works in mm)."""
    out = pose.copy()
    out[:3, 3] = pose[:3, 3] * 1000
    return out


def object_radius(xyz):
    """The frame piece's size (mm, 90th percentile from its centre): turns this large are weighed as shifts."""
    return float(np.percentile(np.linalg.norm(xyz - xyz.mean(0), axis=1), 90)) if len(xyz) else 60.0


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

    track_with: "gpu" (the joint solve against a ray cast of a GPU volume: gputrack.py; the CPU one when
    there is no WebGPU adapter or TRIBUUTSCAN_GPU=0) or "joint" (track.solve against the cloud so far,
    Open3D on the CPU). Tracking takes the uncleaned depth, whose edges and steep faces hold flat objects
    in place; prep.clean is for merging only. The GPU tracker searches the object / base on every 4th
    pixel (the merge takes every 2nd; the masks cover their blocks) and extracts ORB only for keyframes
    and to retry a frame the motion guess did not hold; the CPU one does both on every frame. Earlier
    variants measured with evaluate.py (no cleanup, ungated re-finds, frame-to-frame ICP, markers alone,
    ...) were removed in Phase 6: docs/history.md has their numbers.
    """

    def __init__(self, preset="small", log=print, mode="still", track_with="joint", remove_base=True,
                 merge_min_s=None, live_view=False, merge_process=False):
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
                      "tracking": True, "shape_refinds": 0, "object_turns": False}
        from .scene import Scene
        self.scene = Scene(mode, self.rng)
        self.stats["mode"] = mode
        from .markermap import MarkerMap
        # "joint" (track.solve: shape + markers + motion prior, flat-face check) or "gpu" (the same solve
        # against a ray cast of a GPU volume: gputrack.py).
        # Markers are weighted strictly (1 mm) and noisy ones are dropped inside the solve (3 sigma):
        # measured better than loose weighting (3 mm) even with the estimated colour registration,
        # where loose let marker noise pull poses off and the map drift (tilt doubling 16% vs 78%).
        # The marker check also catches sliding: on the tilt scan, poses whose shape fits well but
        # whose markers disagree are the flat-face slides; dropping the check took doubling 7% -> 26%.
        self.marker_sigma, self.marker_accept = 1.0, 2.0
        self.gpu, self.gpu_kind = None, None  # gputrack.Tracker (made on the first frame) and its adapter
        if track_with == "gpu":
            from . import gputrack
            self.gpu_kind = gputrack.kind()
            if self.gpu_kind is None:
                log("tracking on the CPU (no WebGPU adapter, or TRIBUUTSCAN_GPU=0)")
                track_with = "joint"
        self.track_with, self.marker_map = track_with, MarkerMap(add_mm=3.0)
        self.stats["tracker"] = track_with
        on_gpu = track_with == "gpu"
        self.seg_stride = 2 * self.stride if on_gpu else self.stride  # the object / base search's grid
        self.orb = "demand" if on_gpu else "frame"  # ORB on keyframes and retries (GPU), or on every frame
        self.feat_pose = None  # pose of the frame prev_feat belongs to (with orb="demand": the last keyframe)
        self.orb_job = None  # orb="demand": the last keyframe's ORB, extracted beside tracking (merge thread)
        # The joint solve runs against the model (the cloud so far) once there is one, else the previous
        # frame. Measured: model + markers halved synthetic error (3.4 -> 1.7 mm, hard path 6.1 -> 1.5);
        # the model WITHOUT markers drifts (synthetic 25 mm): markers keep the model honest.
        self.model_levels, self.model_at = None, -1
        try:
            from . import features
            self.features = features  # colour features (OpenCV): starting poses, re-finds, Optimise's loops
        except ImportError:
            self.features = None
        self.prev_feat, self.feature_starts = None, 0  # last tracked frame's ORB; frames started from ORB
        self.cue = {}  # what held the last frame: markers used, ORB start, flat (shape alone slides)
        self.keyframes = []  # merged frames: (index, pose, points mm (camera, 2 mm grid), ORB, markers mm)
        self.merged_data = {}  # index -> (camera points mm float32, rgb, camera normals float32): re-merging
        self.refind_needed = False  # set by continue_from(): the next frames must find the saved world first
        import concurrent.futures
        import threading
        self.cloud_lock = threading.Lock()
        self.merger = concurrent.futures.ThreadPoolExecutor(1)  # merging beside tracking (or in merge_proc)
        self.pending = []
        self.merge_proc, self.orb_slot = None, None  # merger.MergeProcess (merge_process); the keyframe ORB it owes
        if merge_process:  # the merges (and the keyframes' ORB) in a process of their own: not taken off tracking
            try:
                from .merger import MergeProcess
                self.merge_proc = MergeProcess(self.cloud.point_mm, self.stride, True, self.merged_reply)
            except Exception as e:
                log(f"merging on a thread: the merge process did not start ({e})")
        self.track_info = []  # per tracked frame: (index, info from track.solve); for evaluation
        self.last_t, self.motion_s, self.lost_run = None, None, 0  # time of the last tracked frame, of `motion`
        self.frame_table = None  # this frame's own table: (normal, narrower extent mm), for floor_agrees
        self.hold, self.refind_feat = 0, None  # refind_anywhere's state
        self.shape_model, self.shape_t = None, None  # refind_shape's: (merged count, the model's shape features), last try
        self.object_turns = False  # the room stands still while the object turns (check_room): its features left out
        self.room_ref, self.room_votes, self.room_t = None, {"still": 0, "moved": 0}, None  # check_room's reference,
        # verdicts and the frame time it last looked
        self.merge_t = None  # frame time of the last merge
        self.merge_min_s = MERGE_MIN_S if merge_min_s is None else merge_min_s
        self.no_markers = 0  # frames since markers were last seen (frame_markers)
        self.points_kept = 0  # cubes seen MIN_SEEN times after the last merge (read without the cloud lock)
        self.live_view, self.view_codes = live_view, None  # live_view: each GPU-tracked frame's preview codes
        self.gpu_fused = 0  # frames fused into the GPU tracker's volume
        self.track_log = []  # per frame: (index, status, pose or None); for evaluation
        self.on_merge = None  # optional callback(index, world_xyz_mm, rgb, world_normals, pose)
        self.remove_base = remove_base  # the saved/shown cloud leaves the base out (it is merged either way)
        self.crop = None  # the 3D page's crop box, {"axes", "lo", "hi"} (in_crop): the output keeps what is inside
        self.last_view = None  # (pixels, object mask, base mask, shape) of the last tracked frame, for the viewer
        self.labels = {}  # index -> view_mask of every tracked frame (what fusion takes as object / base)
        self.sources = []  # [{"rec": recording dir, "first_index": scanner index of its frame 0}]

    def feed(self, frame, index=None):
        """Process one frame set ({'color', 'depth'}); returns True if it was merged into the cloud."""
        stats, fuse = self.stats, self.fuse
        index = stats["frames"] if index is None else index
        stats["frames"] += 1
        self.last_view, self.view_codes = None, None
        if self.merge_proc is not None:
            try:
                self.merge_proc.poll()  # merges done meanwhile: counts, markers, keyframes' ORB
            except RuntimeError as e:
                self.merge_lost(e)
        shape = frame_shape(frame)
        seg = self.seg_stride  # the object / base search's grid (merging takes every self.stride-th pixel)
        all_xyz, _, all_pix = frame_points(frame, seg, colour=False)  # normals and colour only for merging
        scene = self.scene
        t = frame.get("t")
        guess = self.motion
        if t is not None and self.last_t is not None and self.motion_s:  # the guess follows the real time
            guess = scale_motion(self.motion, min(t - self.last_t, GUESS_MAX_S) / self.motion_s)
        self.frame_table, table_plane = None, None  # the frame's table (camera): for floor_agrees / the masks
        if scene.mode == "turntable":
            scene.observe(all_xyz, all_pix, shape, np.eye(4))
            keep = scene.object_mask(all_xyz, all_pix, shape, np.eye(4))
        else:  # the frame's piece where the guess puts the object (the middle of the view can miss a narrow one)
            seeds = known = None
            if self.prev is not None:
                to_world = scene_transform(self.pose @ guess)
                seeds = scene.core(all_xyz, to_world)
                if scene.mode == "still" and scene.plane is not None:  # the table: the kept base re-fitted, once
                    table = scene.table_in_camera(all_xyz, to_world)
                    if table[3]:
                        table_plane, self.frame_table = table[:2], (table[0], table[2])
                    else:  # (the guess was off: the frame's own plane is searched, the kept one only if none)
                        known = lambda: table[:2]
            if table_plane is not None:
                keep, _ = isolate_mask(all_xyz, all_pix, shape, self.rng, seeds, plane=table_plane)
            else:
                keep, plane = isolate_mask(all_xyz, all_pix, shape, self.rng, seeds, known)
                self.frame_table = table_normal(all_xyz, plane)
        xyz, pix = all_xyz[keep], all_pix[keep]
        if len(xyz) < 500 // seg ** 2:
            stats["lost"] += 1
            stats["tracking"] = False
            self.lost_run += 1
            self.track_log.append((index, "too few points", None))
            if self.gpu is not None:
                self.gpu_source(frame, xyz, pix)  # (what little there is, in orange over the shape)
                self.show_ghost()
            return False
        status = "first" if self.prev is None else "tracked"
        if fuse is not None:
            if self.track_with == "gpu":
                levels = None  # the GPU matches the frame's own maps (Open3D levels only to re-find)
                self.gpu_source(frame, xyz, pix)
            else:
                from . import track
                levels = track.levels(xyz / 1000)
            marker_pts = self.frame_markers(frame)
            feat = None
            if self.features is not None and self.orb == "frame":  # colour features: a starting pose
                feat = self.features.extract(self.features.luma_of(frame), frame["depth"], self.orb_mask(pix))
                if self.prev is not None and self.prev_feat is not None:
                    found, _ = self.features.relative_pose(feat, self.prev_feat, self.rng)
                    self.cue = {"features": bool(found is not None and plausible_step(found, xyz))}
                    if self.cue["features"]:
                        guess = found
                        self.feature_starts += 1
            if self.refind_needed:  # continuing a saved scan: find this frame in its world first
                from . import session
                found = session.refind(self, xyz, marker_pts, levels)
                if found is None:
                    stats["lost"] += 1
                    stats["tracking"] = False
                    self.track_log.append((index, "lost", None))
                    return False
                self.pose, self.motion, self.motion_s = found, np.eye(4), None
                self.refind_needed, status = False, "re-found"
                self.log(f"continuing: found my place in the saved scan at frame {index}")
            elif self.prev is not None:
                if self.track_with == "gpu":
                    self.cue["features"] = False
                    step, info = self.gpu_step(xyz, marker_pts, guess)
                    ok = self.accepted(info, step, xyz)
                    if not ok and self.features is not None:  # an ORB start, then again
                        feat, retry = self.orb_guess(frame, pix, xyz)
                        if retry is not None:
                            step, info = self.gpu_step(xyz, marker_pts, retry)
                            ok = self.accepted(info, step, xyz)
                            self.feature_starts += 1
                    alt = None
                    if ok and self.after_gap(t):  # a second opinion by shape (GAP_S)
                        alt = self.shape_candidate(xyz, marker_pts, t, every_s=0.0)  # (once per gap)
                        if alt is not None and fits_better(alt[1], info) and not same_pose(alt[0], step, xyz):
                            step, info, status = alt[0], alt[1], "re-found by shape"
                            stats["relocked"] += 1
                            stats["shape_refinds"] += 1
                    if ok and status == "tracked" and not info["markers_used"] and len(marker_pts) >= 3 \
                            and len(self.marker_map) >= 3:  # markers in view, none of them used: do they agree?
                        checked = self.marker_check(xyz, marker_pts, step)
                        if checked is not None:
                            step, info, ok = checked
                            if ok:
                                status, alt = "re-found by markers", True
                                stats["relocked"] += 1
                    self.note(index, info)
                    if self.live_view and alt is not None:  # (the last ray cast was the candidate's: one at the pose)
                        self.view_codes = self.gpu.ghost(self.pose @ step, self.gpu_base(), self.preview_weight())
                    elif self.live_view:
                        self.view_codes = self.gpu.preview(self.gpu_base(), self.preview_weight())
                else:
                    step, ok = self.joint_step(levels, xyz, marker_pts, guess, index)
                if ok and status != "tracked":  # (re-found by shape after a gap: no motion to go on from)
                    self.pose, self.motion, self.motion_s = self.pose @ step, np.eye(4), None
                elif ok:
                    self.motion = step
                    self.motion_s = (t - self.last_t) if (t is not None and self.last_t is not None) else None
                    self.pose = self.pose @ self.motion
                elif self.relock(*self.relock_source(levels, xyz), marker_pts):
                    stats["relocked"] += 1
                    status = "relocked"
                elif self.refind_anywhere(frame, pix, xyz, marker_pts, feat):
                    stats["relocked"] += 1
                    status = "re-found"
                    feat = self.refind_feat
                elif self.refind_shape(xyz, marker_pts, t):
                    stats["relocked"] += 1
                    status = "re-found by shape"
                else:
                    stats["lost"] += 1
                    stats["tracking"] = False
                    self.lost_run += 1
                    self.track_log.append((index, "lost", None))
                    self.show_ghost()
                    return False
            self.last_t, self.lost_run = t, 0
            self.prev = True if levels is None else levels
            stats["tracking"] = True
            if len(marker_pts):
                self.marker_map.add(marker_pts, self.pose)
            if self.orb == "frame" or feat is not None:
                self.prev_feat, self.feat_pose = feat, self.pose.copy()
        self.track_log.append((index, status, self.pose.copy()))
        if scene.mode == "still":  # the object (above the kept base, in its region) and the base under it
            to_world = scene_transform(self.pose)  # (every tracked frame: the region grows per frame)
            scene.observe(all_xyz, all_pix, shape, to_world, keep)
            obj, base = scene.masks(all_xyz, all_pix, shape, to_world, keep, table_plane)
            pix = all_pix[obj | base]
            self.last_view = (all_pix, obj, base, shape)
        else:
            self.last_view = (all_pix, keep, np.zeros(len(keep), bool), shape)
        self.labels[index] = view_mask(self.last_view)
        stats["plane_frames"] += scene.plane is not None
        if status.startswith("re-found") and self.track_with == "gpu":
            self.hold = REFIND_HOLD + 1
        if self.hold:  # a pose found anywhere is merged only once the frames after it have tracked from it
            self.hold -= 1
            return False
        self.since_merge = getattr(self, "since_merge", 0) + 1
        if self.last_merged is not None:
            move, turn = turned(np.linalg.inv(self.last_merged) @ self.pose)
            if fuse is not None and move < KEY_MOVE_MM and turn < KEY_TURN_DEG and self.since_merge < KEY_EVERY:
                return False
        self.since_merge, self.last_merged = 0, self.pose.copy()
        if scene.mode == "still" and fuse is not None and not self.object_turns and not self.room_settled() \
                and (t is None or self.room_t is None or not 0 <= t - self.room_t < ROOM_EVERY_S):
            self.room_t = t
            room = ~(obj | base) & np.all(all_pix % ROOM_GRID_PX == 0, axis=1)  # (neither object nor base)
            self.check_room(all_xyz[room])
        if self.gpu is not None:  # into the volume the next frame is tracked against: every such frame (cheap)
            self.gpu.fuse(frame, self.labels[index], self.pose)
            self.gpu_fused += 1
        if t is not None and self.merge_t is not None and t - self.merge_t < self.merge_min_s:
            return False  # the point cloud and Optimise's keyframes take one every MERGE_MIN_S at most
        self.merge_t, orb_job = t, None
        if self.gpu is not None:  # keyframe for Optimise / continuing a scan
            from .gputrack import grid_points
            self.keyframes.append((index, self.pose.copy(), grid_points(xyz), feat, marker_pts))
            if self.features is not None and feat is None:  # keyframes keep their ORB:
                job = (len(self.keyframes) - 1, frame, self.orb_mask(pix), self.pose.copy())
                if self.merge_proc is not None:  # extracted in the merge process, with this frame's merge
                    orb_job, self.orb_slot = (job[0], job[2]), job[0]
                else:  # extracted in the merge thread, in order (ORB releases the GIL)
                    self.orb_job = self.merger.submit(self.keyframe_orb, *job)
                    self.pending.append(self.orb_job)
        elif self.prev is not None and self.prev is not True:
            self.keyframes.append((index, self.pose.copy(), np.asarray(self.prev[-1].points) * 1000,
                                   self.prev_feat, marker_pts))
        self.poses.append((index, self.pose.copy()))
        stats["merged"] += 1
        job = (frame, index, self.pose.copy(), pix, seg)
        if self.merge_proc is not None:
            if "depth_raw" in frame:  # the scanner's bytes travel, not 4 MB of floats: the merge process decodes them
                job = (type(frame)({k: v for k, v in frame.items() if k not in ("depth", "depth4")}),) + job[1:]
            self.merge_proc.submit(*job, orb_job)
        else:  # merging runs beside tracking the next frame (numpy and Open3D release the GIL)
            self.pending.append(self.merger.submit(self.merge, *job))
        return True

    def add_source(self, rec, first_index):
        """Frames from `rec` (a Recorder directory) are fed from scanner index `first_index` on."""
        self.sources.append({"rec": str(rec), "first_index": int(first_index)})

    def frame_poses(self):
        """[(index, pose)] of every tracked frame (a scan saved before these were kept: its merged ones)."""
        return [(i, p) for i, _, p in self.track_log if p is not None] or list(self.poses)

    def turn_axis(self, min_deg=TURN_AXIS_MIN_DEG):
        """The axis a platform turned the object about (turn_axis), or None: not a platform scan (object_turns).
        With no base found, the printable solid is cut square to it (fusion.printing_planes)."""
        return turn_axis([p for _, p in self.frame_poses()], min_deg) if self.object_turns else None

    def merge(self, frame, index, pose, pix, step=1):
        """Merge one tracked frame's object pixels (cleaned, with normals) into the cloud; `pix` sampled every
        `step`-th pixel, each standing for its step x step block (merge_points)."""
        mxyz, mrgb, mnrm = merge_points(frame, pix, step, self.stride)
        self.merged_data[index] = (mxyz.astype(np.float32), mrgb, mnrm.astype(np.float32))
        world = mxyz @ pose[:3, :3].T + pose[:3, 3] * 1000
        world_nrm = mnrm @ pose[:3, :3].T
        with self.cloud_lock:
            self.cloud.add(world, mrgb, world_nrm)
            self.points_kept = int((self.cloud.seen >= MIN_SEEN).sum())
        if self.on_merge is not None:
            self.on_merge(index, world, mrgb, world_nrm, pose)
        if not self.stats["markers"]:
            self.stats["markers"] = marker_list(frame)

    def remerge(self, new_poses, point_mm=None):
        """Merge every merged frame again with corrected poses (dict index -> pose), e.g. after
        optimise.optimise, optionally at another point distance; also moves the current pose and
        the marker map along."""
        self.flush()
        old = dict(self.poses)
        if not old:
            return
        last = self.poses[-1][0]
        fix = new_poses.get(last, old[last]) @ np.linalg.inv(old[last])
        self.pose, self.last_merged = fix @ self.pose, fix @ self.last_merged
        cloud = VoxelCloud(point_mm or self.cloud.point_mm)
        for index, _ in self.poses:
            pose = new_poses.get(index, old[index])
            pts, rgb, nrm = self.merged_data[index]
            cloud.add(pts.astype(np.float64) @ pose[:3, :3].T + pose[:3, 3] * 1000, rgb,
                      nrm.astype(np.float64) @ pose[:3, :3].T)
        with self.cloud_lock:
            self.cloud = cloud
            self.points_kept = int((cloud.seen >= MIN_SEEN).sum())
        # every tracked frame gets a correction blended from the merged frames on either side of it, within
        # one source recording (held constant instead, the corrections jumped at each merged frame and
        # the mesh fused from every frame came out blurred: p90 0.55 mm against 0.34 from merged frames)
        fixes = {i: new_poses.get(i, old[i]) @ np.linalg.inv(old[i]) for i, _ in self.poses}
        keyed = sorted(fixes)
        starts = sorted(src["first_index"] for src in self.sources)
        part = lambda i: bisect.bisect_right(starts, i)
        log = []
        for i, status, p in self.track_log:
            if p is not None:
                k = bisect.bisect_left(keyed, i)
                if k < len(keyed) and (keyed[k] == i or k == 0):
                    fix = fixes[keyed[k]]
                elif k == len(keyed) or part(keyed[k - 1]) != part(keyed[k]):
                    fix = fixes[keyed[k - 1]]
                else:
                    a, b = keyed[k - 1], keyed[k]
                    fix = blend(fixes[a], fixes[b], (i - a) / (b - a))
                p = fix @ p
            log.append((i, status, p))
        self.track_log = log
        self.poses = [(i, new_poses.get(i, p)) for i, p in self.poses]
        self.keyframes = [(i, new_poses.get(i, p), *rest) for i, p, *rest in self.keyframes]
        from .markermap import MarkerMap
        self.marker_map = MarkerMap(add_mm=self.marker_map.add_mm)
        for _, pose, _, _, mk in self.keyframes:
            if len(mk):
                self.marker_map.add(mk, pose)
        self.model_levels = None  # rebuilt from the corrected cloud

    @classmethod
    def restore(cls, directory, log=print, use_mode=True):
        """A Scanner rebuilt from a saved session (keyframes, merged frames, scene), for Optimise. With
        `use_mode` the camera takes the scan's depth mode: what is saved again notes it (a long-range scan
        re-saved by the 3D page's Process under the close law no longer matched its camera for the mesh)."""
        directory = Path(directory)
        session = json.loads((directory / "session.json").read_text())
        if use_mode:
            CAMERA.set_mode(session.get("depth_mode") or CLOSE)
        sc = cls(session.get("preset", "small"), log=log, mode=session.get("mode", "still"),
                 remove_base=session.get("remove_base", True))
        kf = load_npz(directory / "keyframes.npz")
        at = lambda name: kf[name + "_at"]
        for k, index in enumerate(kf["index"]):
            pts = kf["points"][at("points")[k]:at("points")[k + 1]]
            fp = kf["feat_points"][at("feat")[k]:at("feat")[k + 1]]
            fd = kf["feat_desc"][at("feat")[k]:at("feat")[k + 1]]
            mk = kf["markers"][at("markers")[k]:at("markers")[k + 1]]
            sc.keyframes.append((int(index), kf["pose"][k], pts, (fp, fd) if len(fd) else None, mk))
            sc.poses.append((int(index), kf["pose"][k]))
        m = load_npz(directory / "merged.npz")
        for k, index in enumerate(m["index"]):
            a, b = m["at"][k], m["at"][k + 1]
            sc.merged_data[int(index)] = (m["points"][a:b], m["rgb"][a:b], m["normals"][a:b])
        sc.pose = sc.last_merged = sc.poses[-1][1]
        sc.stats.update({k: session[k] for k in ("frames", "merged", "lost", "relocked", "shape_refinds") if k in session})
        sc.object_turns = sc.stats["object_turns"] = bool(session.get("object_turns", False))
        sc.crop = session.get("crop")
        sc.sources = list(session.get("sources", []))
        if (directory / "frame_poses.txt").exists():  # every tracked frame (scans saved before 2026-11-01: none)
            status = {}
            if (directory / "track.csv").exists():
                for line in (directory / "track.csv").read_text().splitlines()[1:]:
                    i, s = line.split(",")[:2]
                    status[int(i)] = s
            sc.track_log = [(i, status.get(i, "tracked"), p) for i, p in read_poses(directory / "frame_poses.txt")]
        if (directory / "labels.npz").exists():
            lab = load_npz(directory / "labels.npz")
            sc.labels = {int(i): m for i, m in zip(lab["index"], lab["labels"])}
        sc.scene.load(json.loads((directory / "scene.json").read_text()))
        sc.remerge({})
        return sc

    @classmethod
    def continue_from(cls, directory, log=print):
        """A Scanner that goes on with a saved scan: restored, and re-finding its place first."""
        sc = cls.restore(directory, log=log, use_mode=False)  # (the scanner's own mode goes on)
        sc.refind_needed = True
        sc.crop = None  # (it would hide what is scanned next)
        sc.prev, sc.prev_feat = None, None
        return sc

    def flush(self):
        """Wait for merges still running in the background (before reading or saving the cloud). With a
        merge process, the cloud and the merged frames come back here and the process ends (later
        merges go to the thread)."""
        for job in self.pending:
            job.result()
        self.pending = []
        if self.merge_proc is not None:
            from .merger import CLOUD_ARRAYS
            try:
                arrays, merged = self.merge_proc.state()
            except RuntimeError as e:
                self.merge_lost(e)
                return
            self.merge_proc = None
            cloud = VoxelCloud(self.cloud.point_mm)
            for name in CLOUD_ARRAYS:
                setattr(cloud, name, arrays[name])
            cloud.frames, cloud.raw_points = arrays["frames"], arrays["raw_points"]
            with self.cloud_lock:
                self.cloud = cloud
            self.merged_data.update(merged)

    def sync_merges(self):
        """Wait for the merges sent so far, wherever they run (the cloud stays where it is)."""
        if self.merge_proc is not None:
            try:
                self.merge_proc.sync()
                return
            except RuntimeError as e:
                self.merge_lost(e)
        self.flush()

    def merge_lost(self, error):
        """The merge process went away mid-scan: what it merged is lost to the point cloud (not to the mesh,
        which is fused again from the recording with the poses kept here); merging goes on in the thread."""
        lost = self.merge_proc.sent
        self.merge_proc, self.orb_slot = None, None
        self.log(f"merge process lost ({error}): {lost} merged frames are missing from the point cloud; "
                 "merging goes on beside tracking")

    def merged_reply(self, reply):
        """A merge done in the merge process: the cloud's counts, the markers, the keyframe's ORB."""
        _, index, kept, raw, found, slot, feat = reply
        self.points_kept, self.cloud.raw_points = kept, raw
        if found and not self.stats["markers"]:
            self.stats["markers"] = found
        if slot is not None:
            k = self.keyframes[slot]
            self.keyframes[slot] = (k[0], k[1], k[2], feat, k[4])
            self.prev_feat, self.feat_pose = feat, k[1].copy()

    def cloud_points(self, min_seen=1):
        """(xyz mm, rgb, seen) of the cloud with every merge sent so far."""
        if self.merge_proc is not None:
            try:
                return self.merge_proc.points(min_seen)
            except RuntimeError as e:
                self.merge_lost(e)
        with self.cloud_lock:
            return self.cloud.points(min_seen)

    def live_points(self, min_seen=1):
        """What the 3D page shows while scanning: the cloud's points inside the object's region (output_mask,
        no connectivity). With a merge process, its latest snapshot: a newer one is asked for, not waited on
        (this runs on the page's thread)."""
        if self.merge_proc is not None:
            self.merge_proc.ask_points(min_seen)
            pts = self.merge_proc.points_cache
            if pts is None:
                return np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty(0, np.int32)
            xyz, rgb, seen = pts
        else:
            with self.cloud_lock:
                xyz, rgb, seen = self.cloud.points(min_seen)
        keep = self.output_mask(xyz, connect=False)
        return xyz[keep], rgb[keep], seen[keep]

    def joint_step(self, levels, xyz, marker_pts, guess, index):
        """One frame with track.solve (shape + markers + prior), against the model (the cloud so far,
        rebuilt every MODEL_EVERY merges), or the previous frame until there is one. Returns (step from the
        previous pose, accepted)."""
        from . import track
        use_model = self.model_ready()
        ref = np.eye(4) if use_model else self.pose  # the target's frame -> world
        pairs = self.marker_pairs(marker_pts, guess, ref)
        init = self.pose @ guess if use_model else guess
        found, info = track.solve(levels, self.model_levels if use_model else self.prev, init, prior=init,
                                  marker_pairs=pairs, radius_mm=object_radius(xyz), marker_sigma_mm=self.marker_sigma)
        step = np.linalg.inv(self.pose) @ found if use_model else found
        return step, self.judge(index, info, step, xyz)

    def gpu_step(self, xyz, marker_pts, guess):
        """One frame on the GPU (gputrack: against a ray cast of the volume of merged frames, the base left
        out by the kept plane as model_points does), with joint_step's marker pairs and prior; returns
        (step from the previous pose, info) for `accepted` and `note`."""
        init = self.pose @ guess
        to_world = scene_transform(init)
        found, info = self.gpu.solve(init, init, self.marker_pairs(marker_pts, guess, np.eye(4)), object_radius(xyz),
                                     xyz.mean(0) @ to_world[:3, :3].T + to_world[:3, 3], self.gpu_base(),
                                     self.marker_sigma)
        return np.linalg.inv(self.pose) @ found, info

    def gpu_base(self):
        """The kept base for the GPU (n, c: the model is what lies OBJECT_ABOVE_MM above it), or None."""
        if self.scene.mode == "still" and self.scene.plane is not None:
            n, c = self.scene.plane
            return n, c - OBJECT_ABOVE_MM
        return None

    def preview_weight(self):
        """How often the live view wants a surface seen before it shows it as done: as often as the mesh
        (fusion.MIN_WEIGHT merged frames), scaled by the GPU volume's frames per merged one (it takes
        every keyframe, the point cloud and the mesh one per MERGE_MIN_S)."""
        from .fusion import MIN_WEIGHT
        return MIN_WEIGHT * max(1.0, self.gpu_fused / max(self.stats["merged"], 1))

    def keyframe_orb(self, slot, frame, mask_pixels, pose):
        """A keyframe's ORB (Optimise's loop candidates, the next ORB start), into its keyframe entry."""
        feat = self.features.extract(self.features.luma_of(frame), frame["depth"], mask_pixels)
        k = self.keyframes[slot]
        self.keyframes[slot] = (k[0], k[1], k[2], feat, k[4])
        self.prev_feat, self.feat_pose = feat, pose

    def orb_guess(self, frame, pix, xyz):
        """(this frame's ORB, a step from the previous pose from matching it with the last frame that has
        ORB, or None)."""
        if self.orb_job is not None:
            self.orb_job.result()  # the last keyframe's ORB first (replays stay deterministic)
            self.orb_job = None
        if self.merge_proc is not None and self.orb_slot is not None:
            self.sync_merges()  # (the same, from the merge process: its replies come in order)
            self.orb_slot = None
        feat = self.features.extract(self.features.luma_of(frame), self.depth_of(frame), self.orb_mask(pix))
        if self.prev_feat is None or self.feat_pose is None:
            return feat, None
        found, _ = self.features.relative_pose(feat, self.prev_feat, self.rng)
        if found is None:
            return feat, None
        step = np.linalg.inv(self.pose) @ self.feat_pose @ found
        self.cue["features"] = bool(plausible_step(step, xyz))
        return feat, (step if self.cue["features"] else None)

    def gpu_source(self, frame, xyz, pix):
        """Hand this frame's piece to the GPU tracker; the first frame makes it, its volume around this piece."""
        if self.gpu is None:
            from . import gputrack
            to_world = scene_transform(self.pose)
            self.gpu = gputrack.Tracker(CAMERA, frame_shape(frame)[::-1],
                                        xyz.mean(0) @ to_world[:3, :3].T + to_world[:3, 3], self.gpu_kind)
            self.log(f"tracking on the GPU: {self.gpu.engine.device_name}")
        self.gpu.source(frame, pix, self.seg_stride, float(np.percentile(xyz[:, 2], 2)) if len(xyz) else None)

    def close(self):
        """Give the GPU tracker's memory back (after the last frame; the cloud and everything saved stay)."""
        if self.gpu is not None:
            self.stats["gpu_ms"] = self.gpu.report()
            self.gpu.close()
            self.gpu = None

    def marker_pairs(self, marker_pts, guess, ref):
        """(this frame's markers mm, their map positions in the target frame mm) for the markers matched
        at the predicted pose, or None; ref: the target's frame -> world."""
        if not len(marker_pts) or not len(self.marker_map):
            return None
        matched = self.marker_map.match_predicted(marker_pts, self.pose @ guess)
        if not matched:
            return None
        to_ref = np.linalg.inv(scene_transform(ref))  # world (mm) -> target frame (mm)
        dst = self.marker_map.pos[[m for _, m in matched]] @ to_ref[:3, :3].T + to_ref[:3, 3]
        return marker_pts[[f for f, _ in matched]], dst

    def judge(self, index, info, step, xyz):
        """Note a solve's info (flat-face hint, cues) and say whether its pose is taken."""
        self.note(index, info)
        return self.accepted(info, step, xyz)

    def note(self, index, info):
        """Keep a frame's solve info (evaluation, track.csv) and the cues the viewer shows."""
        from . import track
        self.track_info.append((index, info))
        flat = bool(info["degeneracy"] is not None and info["degeneracy"] < track.DEGENERATE and not info["markers_used"])
        self.stats["flat"] = self.stats.get("flat", 0) + flat
        self.cue = {"markers": int(info["markers_used"]), "features": bool(self.cue.get("features", False)),
                    "flat": flat}  # plain Python values: the page receives them as JSON

    def accepted(self, info, step, xyz):
        """Is a solve's pose taken: its checks (solve_ok), a believable step, the table as flat as it was."""
        return self.solve_ok(info, xyz) and plausible_step(step, xyz) and self.floor_agrees(step)

    def floor_agrees(self, step):
        """In "still" mode, does the frame's own table (frame_table, when broad enough to tell) lie as flat as
        the kept base with the camera at self.pose @ step? Their normals within FLOOR_MAX_DEG; a plane more
        than FLOOR_OTHER_DEG off is another surface and says nothing, and so does a narrow strip (a table
        rim around an object filling the view fixes the tilt across it poorly). Tilt only: an object's flat
        top lies parallel to the table. Hand scan (long range): good poses within 0.75 degrees, poses that
        slid along the thin fingers 3.4-8.2; marker scan: its table rim 80 mm across read 1.6-3.7."""
        from .scene import plane_to
        scene, own = self.scene, self.frame_table
        if scene.mode != "still" or scene.plane is None or own is None or own[1] < FLOOR_MIN_SPAN_MM:
            return True
        n = plane_to(scene.plane, np.linalg.inv(scene_transform(self.pose @ step)))[0]
        tilt = np.degrees(np.arccos(np.clip(abs(n @ own[0]), -1, 1)))
        return tilt <= FLOOR_MAX_DEG or tilt > FLOOR_OTHER_DEG

    def solve_ok(self, info, xyz=None):
        """Did a solve fit: enough overlap, a small residual, markers agreeing. On the GPU the residual
        must stay within GPU_MAX_RMSE_MM at the piece's distance (`xyz`)."""
        limit = JOINT_MAX_RMSE_MM
        if self.gpu is not None and xyz is not None and len(xyz):
            limit = GPU_MAX_RMSE_MM * max(1.0, float(np.median(xyz[:, 2])) / GPU_RMSE_Z_MM) ** 2
        return (info["fitness"] >= JOINT_MIN_FITNESS and info["rmse_mm"] is not None
                and info["rmse_mm"] <= limit
                and (info["marker_rmse_mm"] is None or info["marker_rmse_mm"] <= self.marker_accept))

    def model_ready(self):
        """Prepare the model target (levels of the cloud so far, world); False until it has enough points."""
        from . import fuse
        if self.stats["merged"] < 2:
            return False
        if self.model_levels is None or self.stats["merged"] - self.model_at >= MODEL_EVERY:
            self.sync_merges()  # the model must not depend on how far background merges have got (replays match)
            xyz = self.model_points()
            if len(xyz) < 500:
                return self.model_levels is not None
            self.model_levels, self.model_at = fuse.levels(xyz / 1000), self.stats["merged"]
        return True

    def depth_of(self, frame):
        """A frame's full depth: as decoded, or for a live frame (depth.LazyFrame) read back from the GPU
        that unpacked it (~2 ms), else decoded on first use. Tracking's thread only (the GPU's buffer)."""
        if "depth" not in frame and self.gpu is not None:
            depth = self.gpu.depth(frame)
            if depth is not None:
                frame["depth"] = depth
        return frame["depth"]

    def frame_markers(self, frame):
        """Camera-frame centres (mm, (N, 3)) of the markers in a frame; uses frame["markers2d"]
        (pre-detected colour pixels) when given. While none were seen for MARKER_IDLE_FRAMES frames,
        markers are looked for only on every MARKER_IDLE_EVERY-th (the search took 7-12 ms a frame
        live, on a scan without markers)."""
        found = frame.get("markers2d")
        if found is None:
            if self.no_markers >= MARKER_IDLE_FRAMES and self.stats["frames"] % MARKER_IDLE_EVERY:
                return np.empty((0, 3))
            from .features import luma_of
            found = markers.detect_fast(luma_of(frame), frame.get("color"))
        if not len(found):  # (nothing to place: a live frame's depth stays packed)
            self.no_markers += 1
            return np.empty((0, 3))
        pts = [loc[0] for loc in markers.locate(self.depth_of(frame), found) if loc is not None]
        self.no_markers = 0 if pts else self.no_markers + 1
        return np.array(pts).reshape(-1, 3)

    def refind_anywhere(self, frame, pix, xyz, marker_pts, feat=None):
        """Lost, and not back near the last good pose (relock): find the frame anywhere in the scan by
        matching its ORB with keyframes' (the whole image in "still" mode, where the scene stands still),
        the best REFIND_TRIES of up to REFIND_KEYFRAMES spread over the scan, each candidate with
        REFIND_MIN_INLIERS (REFIND_MIN_AWAY of them on the still scene, off the object), checked by the
        tracking solve (solve_ok, RELOCK_FITNESS, floor_agrees) and by the time lost (reachable). GPU tracker
        only. The ORB extracted here is kept in self.refind_feat."""
        self.refind_feat = feat
        kfs = [k for k in self.keyframes if k[3] is not None and k[3][1] is not None]
        if self.gpu is None or self.features is None or not kfs:
            return False
        if feat is None:
            feat = self.refind_feat = self.features.extract(self.features.luma_of(frame), self.depth_of(frame),
                                                            self.orb_mask(pix))
        pick = sorted(set(np.linspace(0, len(kfs) - 1, min(REFIND_KEYFRAMES, len(kfs))).round().astype(int))
                      | set(range(max(0, len(kfs) - 4), len(kfs))))
        counts = sorted(((len(self.features.good_matches(feat, kfs[j][3])), j) for j in pick), reverse=True)
        for count, j in counts[:REFIND_TRIES]:
            if count < REFIND_MIN_INLIERS:
                break
            pts = []
            found, inliers = self.features.relative_pose(feat, kfs[j][3], self.rng, pts)
            if found is None or inliers < REFIND_MIN_INLIERS:
                continue
            pose = kfs[j][1] @ found
            if self.scene.mode == "still" and not self.object_turns \
                    and self.scene.away(pts[0], scene_transform(pose)).sum() < REFIND_MIN_AWAY:
                continue  # held by the object alone: a symmetric one matches itself turned round
            step, info = self.gpu_step(xyz, marker_pts, np.linalg.inv(self.pose) @ pose)
            if self.solve_ok(info, xyz) and info["fitness"] >= RELOCK_FITNESS and self.floor_agrees(step) \
                    and self.reachable(step, xyz, frame.get("t")):
                self.pose, self.motion, self.motion_s = self.pose @ step, np.eye(4), None
                return True
        return False

    def reachable(self, step, xyz, t):
        """Could a hand-held scanner have moved by `step` (from the last good pose, judged at the object's
        centre as plausible_step) in the time since that pose? REFIND_MM_S / REFIND_DEG_S plus a slack;
        True without time stamps. The synthetic plate, symmetric with the same markers on both faces, was
        otherwise re-found 180 degrees round after 1 s."""
        if t is None or self.last_t is None:
            return True
        dt = max(t - self.last_t, 0.0)
        c = xyz.mean(0) / 1000
        moved_mm = np.linalg.norm(step[:3, :3] @ c + step[:3, 3] - c) * 1000
        return (moved_mm <= REFIND_MM_S * dt + REFIND_SLACK[0]
                and turned(step)[1] <= REFIND_DEG_S * dt + REFIND_SLACK[1])

    def orb_mask(self, pix):
        """Where colour features are taken: the whole image in "still" mode, where the room stands for the
        scanner's motion, unless check_room found it standing while the object turns; else the object's pixels."""
        return None if self.scene.mode == "still" and not self.object_turns else pix

    def refind_shape(self, xyz, marker_pts, t):
        """Lost, and neither back near the last good pose (relock) nor found by colour (refind_anywhere): fit
        the frame's piece to the object scanned so far, anywhere on it, by shape (session.shape_match, as
        continuing a scan does), refined and checked by the tracking solve: solve_ok, RELOCK_FITNESS, the
        shape pinning the pose (an edge-on hand "fits" anywhere: fitness 1.00 at contradictory angles on the
        first platform scan, so a flat-face view is refused), floor_agrees and reachable. It works where the
        view overlaps what is scanned (the live view shows the scan so far while lost: show_ghost). At most
        one try per SHAPE_EVERY_S of frame time. GPU tracker only."""
        found = self.shape_candidate(xyz, marker_pts, t)
        if found is None:
            return False
        self.pose, self.motion, self.motion_s = self.pose @ found[0], np.eye(4), None
        self.stats["shape_refinds"] += 1
        return True

    def shape_candidate(self, xyz, marker_pts, t, every_s=SHAPE_EVERY_S):
        """refind_shape's search: (step from the last good pose, its solve info) placing the frame's piece on the
        object scanned so far by shape and passing the checks, or None (also between tries: every_s)."""
        if self.gpu is None or self.fuse is None or not self.stats["merged"] or len(xyz) < 200:
            return None
        if t is not None and self.shape_t is not None and 0 <= t - self.shape_t < every_s:
            return None
        self.shape_t = t
        from . import session, track
        if self.shape_model is None or self.shape_model[0] != self.stats["merged"]:
            self.sync_merges()
            model = self.model_points(min_seen=2)
            if len(model) < 200:
                return None
            self.shape_model = (self.stats["merged"], session.shape_features(model))
        found = session.shape_match(session.shape_features(xyz), self.shape_model[1])
        if found is None:
            return None
        step, info = self.gpu_step(xyz, marker_pts, np.linalg.inv(self.pose) @ found)
        if not (self.solve_ok(info, xyz) and info["fitness"] >= RELOCK_FITNESS
                and (info["degeneracy"] or 0.0) >= track.DEGENERATE and self.floor_agrees(step)
                and self.reachable(step, xyz, t)):
            return None
        return step, info

    def marker_check(self, xyz, marker_pts, step):
        """An accepted solve used none of the >= 3 markers the frame shows: are they where it put them? Placed by
        the marker map alone (match_global), they either agree with the solve (None: keep it), or the solve is
        done again from their pose with them (step, info, True when it passes and holds MARKER_CHECK_MIN of them),
        or the frame is not taken (step, info, False). synthhard, coming back from beyond the depth window: the
        motion guess carried on (~14 mm a frame) after the scanner stopped, the box's flat faces did not resist,
        and 8-9 markers in view matched nothing at the sliding pose (ATE 0.47 -> 122 mm on other random draws)."""
        found = self.marker_map.match_global(marker_pts)
        if found is None:
            return None
        start = np.linalg.inv(self.pose) @ found
        if same_pose(start, step, xyz, mm=MARKER_CHECK_MM, deg=MARKER_CHECK_DEG):
            return None
        step2, info2 = self.gpu_step(xyz, marker_pts, start)
        ok = self.solve_ok(info2, xyz) and info2["markers_used"] >= MARKER_CHECK_MIN and self.floor_agrees(step2)
        return step2, info2, ok

    def after_gap(self, t):
        """Is this the first frame after GAP_S without a tracked one (or after lost frames, without stamps)?"""
        if t is None or self.last_t is None:
            return self.lost_run > 0
        return t - self.last_t > GAP_S

    def check_room(self, room):
        """Does the room (this keyframe's points that are neither object nor base, camera mm) stand still in
        the view while tracking says the view of the object moved: the object turns (a platform) and the
        scanner stands? Then the room's colour features say "nothing moved" and must not be used: the first
        platform scan was re-found from the shelves behind the hand and Optimise closed 102 loops on them.
        Compared with the room of an earlier keyframe once the tracked motion moves ROOM_MIN of its points
        ROOM_SHIFT_MM (room_verdict); ROOM_VERDICTS "still" verdicts (and more than "moved" ones) set
        object_turns for the rest of the scan. `room`: on the ROOM_GRID_PX pixel grid (feed)."""
        query = room[::max(1, len(room) // ROOM_POINTS)]  # (thinned only where it is looked for: found again in the
        if self.room_ref is None:                          # whole room, a still one is not lost between samples)
            self.room_ref = (self.pose.copy(), query)
            return
        ref_pose, ref_room = self.room_ref
        move = np.linalg.inv(self.pose) @ ref_pose  # the reference camera -> this camera (metres)
        moved = ref_room @ move[:3, :3].T + move[:3, 3] * 1000
        far = np.linalg.norm(moved - ref_room, axis=1) >= ROOM_SHIFT_MM
        if far.sum() < ROOM_MIN or len(room) < ROOM_MIN:
            return
        self.room_ref = (self.pose.copy(), query)
        verdict = room_verdict(ref_room[far], moved[far], room)
        if verdict is None:
            return
        self.room_votes[verdict] += 1
        if verdict == "still" and self.room_votes["still"] >= ROOM_VERDICTS \
                and self.room_votes["still"] > self.room_votes["moved"]:
            self.object_turns = self.stats["object_turns"] = True
            self.log("the object turns and the scanner stands (a platform): the room's colour features are left out")

    def room_settled(self):
        """Has check_room settled on "the scanner moves" (ROOM_SETTLED verdicts, none "still")?"""
        return self.room_votes["moved"] >= ROOM_SETTLED and not self.room_votes["still"]

    def show_ghost(self):
        """While lost, the live view shows the scan so far where it was last seen (the last good pose, over
        the whole view), with this frame's piece in orange where it does not match: turn (or move) back until
        they line up, and relock or refind_shape takes over."""
        if self.live_view and self.gpu is not None and self.stats["merged"]:
            self.view_codes = self.gpu.ghost(self.pose, self.gpu_base(), self.preview_weight())

    def relock_source(self, levels, xyz):
        """Levels to re-find with: the frame's object piece, as tracked (the cloud holds only the object)."""
        return (self.fuse.levels(xyz / 1000) if levels is None else levels), xyz

    def relock(self, levels, xyz, marker_pts=None):
        """Tracking failed: try to place the frame against the cloud so far, from the last known pose.

        Works when the scanner comes back near where tracking was lost (like the app's "move back").
        The re-found pose must be a believable step from the last good pose
        (limits widen with the frames lost, up to RELOCK_MAX_SCALE): the first version accepted
        jumps of 120-190 mm, which merged shifted copies of the object. On the GPU the pose found must
        also pass the tracking solve's own checks (markers among them, which Open3D's ICP does not
        have): synth replayed with the region-seeded source re-found a box 8.3 mm off and merged it.
        """
        if not self.stats["merged"]:
            return False
        if self.model_merged != self.stats["merged"]:
            self.sync_merges()  # the model must not depend on how far background merges have got (replays match)
            self.model, self.model_merged = self.fuse.levels(self.model_points() / 1000), self.stats["merged"]
        result = self.fuse.icp(levels, self.model, self.pose)
        if result.fitness < RELOCK_FITNESS:
            return False
        found = result.transformation
        if self.gpu is not None:
            step, info = self.gpu_step(xyz, np.empty((0, 3)) if marker_pts is None else marker_pts,
                                       np.linalg.inv(self.pose) @ found)
            if not self.solve_ok(info, xyz):
                return False
            found = self.pose @ step
        jump = np.linalg.inv(self.pose) @ found  # current camera -> last good camera
        if not plausible_step(jump, xyz, min(self.lost_run + 1, RELOCK_MAX_SCALE)):
            return False
        if not self.floor_agrees(jump):
            return False
        self.pose, self.motion, self.motion_s = found, np.eye(4), None
        return True

    def model_points(self, min_seen=1):
        """The cloud so far without its base: what tracking matches against (the object only, as it
        was before the base was kept in the cloud)."""
        xyz = self.cloud_points(min_seen)[0]
        scene = self.scene
        if scene.mode == "still" and scene.plane is not None:
            xyz = xyz[scene.height(xyz) > OBJECT_ABOVE_MM]
        return xyz

    def crop_axes(self):
        """The axes of the 3D page's crop box (crop_axes): along the base's normal, or the platform's turn axis."""
        scene = self.scene
        return crop_axes(scene.plane[0] if scene.mode == "still" and scene.plane is not None else self.turn_axis())

    def in_crop(self, xyz):
        """Which points (world mm) lie inside the crop box: `crop` {"axes": crop_axes rows, "lo", "hi": mm
        along them}; all of them without one."""
        if self.crop is None:
            return np.ones(len(xyz), bool)
        q = xyz @ (np.asarray(self.crop["axes"]) @ FLIP).T
        return np.all((q >= self.crop["lo"]) & (q <= self.crop["hi"]), axis=1)

    def output_mask(self, xyz, connect=True):
        """Which cubes (world centres, mm) are shown and saved: inside the object's region (the
        cylinder on the base, or the ball when no base was found), above the base when remove_base,
        inside the crop box when there is one, and (`connect`) only what is connected to the object's
        core, which drops floating leftovers."""
        scene, keep = self.scene, self.in_crop(xyz)
        if scene.mode != "still" or not len(xyz):
            return keep
        if scene.plane is not None and scene.axis is not None:
            n, h = scene.plane[0], scene.height(xyz)
            r = np.linalg.norm(xyz - np.outer((xyz - scene.axis) @ n, n) - scene.axis, axis=1)
            keep &= (r <= scene.radius) & (h <= scene.top) & (h > (OBJECT_ABOVE_MM if self.remove_base else -PLANE_TOL_MM))
            core = keep & (h > OBJECT_ABOVE_MM) & (r <= max(scene.radius - scene_margin(), scene.radius / 2))
        elif scene.ball is not None:
            centre, radius, _ = scene.ball
            d = np.linalg.norm(xyz - centre, axis=1)
            keep &= d <= radius
            core = keep & (d <= radius / 2)
        else:
            return keep
        if connect and core.any() and keep.sum() > core.sum():
            idx = np.nonzero(keep)[0]
            piece = connected_piece(xyz[idx], core[idx], max(OUTPUT_CELL_MM, 3 * self.cloud.point_mm))
            keep[idx[~piece]] = False
        return keep

    def output_cloud(self, connect=True):
        """The cloud as shown and saved (see output_mask); the merged cloud itself is left whole."""
        with self.cloud_lock:
            cloud = self.cloud
            if not len(cloud):
                return cloud
            return cloud.subset(self.output_mask(cloud.xyz / cloud.n[:, None], connect))

    def save(self, directory, min_seen=MIN_SEEN):
        self.flush()
        self.trim_table()
        self.stats["remove_base"], self.stats["base_found"] = self.remove_base, self.scene.plane is not None
        self.stats["sources"] = self.sources
        self.stats["crop"], self.stats["crop_axes"] = self.crop, self.crop_axes().round(6).tolist()
        out = save(directory, self.output_cloud(), self.poses, self.stats, self.preset, min_seen)
        self.save_state(directory)
        return out

    def save_state(self, directory):
        """What Optimise, continuing a scan and fusing it again need: per-frame tracking log and poses,
        labels, keyframes, marker map and the cloud's sums (track.csv, frame_poses.txt, labels.npz,
        keyframes.npz, markers.json, cloud_state.npz, merged.npz, scene.json)."""
        write_poses(directory / "frame_poses.txt", self.frame_poses())
        tracked = [i for i, _ in self.frame_poses() if self.labels.get(i) is not None]
        if tracked:
            np.savez_compressed(directory / "labels.npz", index=np.array(tracked),
                                labels=np.stack([self.labels[i] for i in tracked]), step=LABEL_STEP)
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
        if self.merged_data:
            order = [i for i, _ in self.poses if i in self.merged_data]
            parts = [self.merged_data[i] for i in order]
            np.savez_compressed(directory / "merged.npz", index=np.array(order),
                                at=np.cumsum([0] + [len(p[0]) for p in parts]),
                                points=np.concatenate([p[0] for p in parts]), rgb=np.concatenate([p[1] for p in parts]),
                                normals=np.concatenate([p[2] for p in parts]))
        (directory / "scene.json").write_text(json.dumps(self.scene.state()))

    def trim_table(self):
        """Drop cubes merged before the table plane was settled: below it or outside the region (the
        base itself stays; output_mask leaves it out when remove_base)."""
        self.flush()
        scene = self.scene
        if scene.mode != "still" or scene.plane is None or scene.axis is None or not len(self.cloud):
            return
        c = self.cloud.xyz / self.cloud.n[:, None]
        n = scene.plane[0]
        h = scene.height(c)
        flat = c - np.outer((c - scene.axis) @ n, n)
        keep = (h > -PLANE_TOL_MM) & (np.linalg.norm(flat - scene.axis, axis=1) <= scene.radius)
        with self.cloud_lock:
            self.cloud.keep(keep)


def scan(directory, preset="small", log=print, remove_base=True, track_with="gpu"):
    """Merge a recording; returns the Scanner (its cloud, poses and stats; save() writes the result).
    Tracks on the GPU (gputrack) when there is one, else on the CPU (track.solve)."""
    scanner = Scanner(preset, log, remove_base=remove_base, track_with=track_with)
    scanner.add_source(Path(directory).resolve(), 0)
    try:
        for i, frame in enumerate(fr.read_recording(directory)):
            if scanner.feed(frame, i) and scanner.stats["merged"] % 10 == 1:
                log(f"frame {i:4d}: merged {scanner.stats['merged']:3d}, cubes {len(scanner.cloud):7d}, "
                    f"raw points so far {scanner.cloud.raw_points:9d}")
    finally:
        scanner.close()  # the GPU's memory back before Optimise and the mesh
    scanner.flush()
    return scanner


def ply_arrays(cloud, poses, min_seen=MIN_SEEN):
    """What cloud.ply holds: (xyz metres y/z flipped for viewers, rgb, normals, seen, min_seen used)."""
    xyz, rgb, seen = cloud.points(min_seen)
    if not len(xyz) and len(cloud):  # too short a scan for min_seen: keep what there is, and say so
        min_seen = 1
        xyz, rgb, seen = cloud.points(min_seen)
    normals = cloud.normals(xyz) if len(xyz) else np.empty((0, 3))
    if poses:  # face the normals toward the cameras that saw the points
        centres = np.array([p[:3, 3] for _, p in poses]) * 1000
        flip = np.einsum("ij,ij->i", normals, centres.mean(0) - xyz) < 0
        normals[flip] *= -1
    return (xyz @ FLIP.T / 1000).astype(np.float32), rgb, normals @ FLIP.T, seen, min_seen


def save(directory, cloud, poses, stats, preset, min_seen=MIN_SEEN):
    """Write cloud.ply, poses.txt and session.json; returns (path, number of points)."""
    directory.mkdir(parents=True, exist_ok=True)
    xyz, rgb, normals, seen, min_seen = ply_arrays(cloud, poses, min_seen)
    out = directory / "cloud.ply"
    pc.write_ply(out, xyz, rgb, normals=normals, seen=seen)
    write_poses(directory / "poses.txt", poses)
    (directory / "session.json").write_text(json.dumps({
        "preset": preset, "point_distance_mm": cloud.point_mm, "min_seen": min_seen, "mode": stats.get("mode"),
        "raw_points": cloud.raw_points, "cubes": len(cloud), "points_kept": len(xyz), **stats,
        "depth_mode": CAMERA.mode,
        "camera": {"source": CAMERA.source, "fingerprint": CAMERA.fingerprint()},
        "note": ("distances rest on the estimated depth law and field of view" if CAMERA.estimate
                 else f"camera: {CAMERA.source} calibration"),
    }, indent=2))
    return out, len(xyz)


def main():
    directory = Path(sys.argv[1])
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    preset = args[0] if args else "small"
    t0 = time.time()
    scanner = scan(directory, preset, remove_base="--keep-base" not in sys.argv)
    out_dir = directory.parent / (directory.name.removesuffix("-rec") + "-scan")
    out, kept = scanner.save(out_dir)
    cloud, stats = scanner.cloud, scanner.stats
    raw_mb = cloud.raw_points * 15 / 1e6  # float32 xyz + rgb per raw point
    base = ("base found, " + ("removed" if scanner.remove_base else "kept")) if stats["base_found"] else "no base found"
    print(f"{stats['frames']} frames, {stats['merged']} merged, {stats['lost']} lost or skipped, "
          f"support plane in {stats['plane_frames']}; {base}")
    print(f"raw points {cloud.raw_points} (~{raw_mb:.0f} MB) -> {len(cloud)} cubes at {cloud.point_mm} mm "
          f"-> {kept} kept (seen >= {MIN_SEEN}, object only), {out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s")
    print(f"-> {out}")
    if "--optimise" in sys.argv:  # as a live scan does after Stop
        from . import optimise
        print(optimise.process(scanner, out_dir, "optimise"))
    if "--mesh" in sys.argv:
        from . import fusion
        report = fusion.make_mesh(out_dir, scanner=scanner)
        print(f"mesh: {report['triangles']:,} triangles from {report['frames']} frames, "
              f"{time.time() - t0:.0f} s in all -> {out_dir / 'mesh.ply'}, mesh.stl")


if __name__ == "__main__":
    main()
