"""Shared parts of the GPU volume engine (gpu_wgpu.py; an OpenCL one was measured against it, bakeoff.py):
layout, inputs, numpy references.

Layout, the same in every engine: the scan volume is a dense grid of bricks (BRICK^3 voxels of
voxel_mm; brick (0, 0, 0) starts at `origin`), each with a u32 in `index`: EMPTY until a frame touches
it, then its slot in a pool of `capacity` bricks. A voxel holds one u32 of TSDF (int16, normalised by
the truncation) and weight (uint16), and one u32 of colour (8-bit r, g, b). Voxel (i, j, k) sits at
origin + (i, j, k) * voxel_mm, Open3D's convention: with `origin` on the brick lattice the voxels are
Open3D's (fusion.Engine), so the two can be compared voxel by voxel.

Portable by construction (they hold on WebGPU and on OpenCL, on any GPU and on the CPU fallbacks
llvmpipe and PoCL): no float or 64-bit atomics (one thread per voxel; sums reduced per work-group and
finished on the CPU in float64), no FP64, 32-bit indices, at most 8 storage buffers of at most 128 MB
per kernel (WebGPU's default limits; llvmpipe allows no more).

A frame comes in as depth in mm on a pinhole grid (0 = none), colour (8-bit RGB) on the same grid and a
pose camera -> world with its translation in mm. Kernels: vertex/normal maps, allocate (bricks a frame
touches) + integrate, ray cast, ICP sums. The numpy functions here define what each must return.
"""
import struct

import numpy as np

VOXEL_MM, TRUNC_VOXELS, BRICK = 0.5, 3.0, 8
CAPACITY = 65536  # bricks in the pool: 128 MB per voxel buffer, the binding limit (Phase 1 marker mesh: 10.5k)
MAX_VISIBLE = 65535  # bricks one frame may touch (a dispatch's limit per dimension)
ALLOC_STRIDE = 4  # bricks are allocated from every 4th pixel each way, at 4 depths within the truncation
                  # along its ray (Open3D's DepthTouch, down_factor 4: the same bricks as Phase 1)
EMPTY, PENDING, FULL = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD  # index entries that are not slots
TUKEY_MM = 3.0  # ICP: residuals beyond this get no weight (track.TUKEY_MM)
NORMAL_GATE = 0.866  # ICP: frame and model normals within 30 degrees
RAY_MIN_WEIGHT = 1.0  # ray cast: voxels seen at least this often
RAY_STEPS = 1024
ICP_GROUPS = 256  # work-groups of the ICP sums; each writes ICP_VALUES partial sums
ICP_VALUES = 32  # 21 of the 6x6 J^T W J (upper triangle, row by row), 6 of J^T W r, then: pairs used, sum w,
                 # sum r^2 of those pairs, frame pixels taking part, pixels matched to the model (within reach)
PARAMS_BYTES = 336


class Layout:
    """Where the volume is: `origin` (world mm, put on the brick lattice), `dims` bricks per axis."""

    def __init__(self, lo_mm, hi_mm, voxel_mm=VOXEL_MM, trunc_voxels=TRUNC_VOXELS, capacity=CAPACITY):
        self.voxel_mm, self.trunc_mm, self.capacity = float(voxel_mm), float(voxel_mm * trunc_voxels), int(capacity)
        self.brick_mm = BRICK * self.voxel_mm
        lo = np.floor(np.asarray(lo_mm, float) / self.brick_mm)
        hi = np.ceil(np.asarray(hi_mm, float) / self.brick_mm)
        self.origin = lo * self.brick_mm
        self.dims = (hi - lo).astype(np.int64)
        if np.prod(self.dims) >= 2 ** 31:
            raise ValueError(f"volume of {self.dims} bricks is too big for 32-bit indices")

    @property
    def bricks(self):
        return int(np.prod(self.dims))


def pack_colour(rgb):
    """(H, W, 3) colour (0-255, any dtype) -> (H, W) u32 r | g << 8 | b << 16."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = rgb
    return out.view("<u4")[..., 0]


def unpack_colour(c):
    c = np.asarray(c, np.uint32)
    return np.stack([c & 255, (c >> 8) & 255, (c >> 16) & 255], -1).astype(np.uint8)


def unpack_tsdf(packed):
    """u32 voxels -> (tsdf normalised float32, weight uint16)."""
    p = np.asarray(packed, np.uint32)
    return (p & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.float32) / 32767, (p >> 16).astype(np.uint16)


def pose_mm(pose_m):
    """A pose with its translation in metres (fusion, cloud) -> the same in mm."""
    out = np.array(pose_m, np.float64)
    out[:3, 3] *= 1000
    return out


def params(layout, K, size, pose=np.eye(4), ref_inv=np.eye(4), centre=(0, 0, 0), frame=0, far_mm=1000.0,
           near_mm=50.0, min_weight=RAY_MIN_WEIGHT, limit=MAX_VISIBLE, tukey=TUKEY_MM, gate=NORMAL_GATE,
           area=0.0, reach_mm=1e9, normal_step=1, stride=1, base=(0.0, 0.0, 0.0, 1.0)):
    """The parameter block every kernel reads (PARAMS_BYTES; matrices column by column). ICP (icp_sums):
    tukey, gate, area (pixel footprint scale), reach_mm, stride, base; maps and model normals:
    normal_step; ray cast: min_weight."""
    pose = np.asarray(pose, np.float64)
    mats = [pose, np.linalg.inv(pose), np.asarray(ref_inv, np.float64)]
    w, h = size
    out = b"".join(np.ascontiguousarray(m.T, np.float32).tobytes() for m in mats)
    out += struct.pack("<4f", K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    out += struct.pack("<4f", *layout.origin, layout.voxel_mm)
    out += struct.pack("<4f", layout.trunc_mm, near_mm, far_mm, tukey)
    out += struct.pack("<4f", *centre, gate)
    out += struct.pack("<4f", min_weight, 0.8, RAY_STEPS, 0)
    out += struct.pack("<4I", *layout.dims, layout.capacity)
    out += struct.pack("<4I", w, h, frame, limit)
    out += struct.pack("<4f", area, reach_mm, normal_step, stride)
    out += struct.pack("<4f", *base)
    return out


# --- numpy references -------------------------------------------------------------------------------

def rays(K, size):
    """(H, W, 3) pixel rays (x/z, y/z, 1) of the pinhole K."""
    w, h = size
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)


def maps(depth, K, step=1):
    """Vertex and normal maps of a depth image (mm): (H, W, 4) each, w = 1 where valid. Normals from the
    four neighbours `step` pixels away (cross product of the horizontal and vertical central
    differences, as cloud.pixel_normals), facing the camera; none at the border or next to a pixel
    without depth."""
    h, w = depth.shape
    k = step
    d = np.asarray(depth, np.float64)
    v = rays(K, (w, h)) * d[..., None]
    ok = d > 0
    vmap = np.concatenate([np.where(ok[..., None], v, 0), ok[..., None].astype(np.float64)], -1)
    nmap = np.zeros((h, w, 4))
    inner = np.zeros_like(ok)
    inner[k:-k, k:-k] = ok[k:-k, k:-k] & ok[k:-k, 2 * k:] & ok[k:-k, :-2 * k] & ok[2 * k:, k:-k] & ok[:-2 * k, k:-k]
    dx = np.zeros_like(v)
    dy = np.zeros_like(v)
    dx[k:-k, k:-k] = v[k:-k, 2 * k:] - v[k:-k, :-2 * k]
    dy[k:-k, k:-k] = v[2 * k:, k:-k] - v[:-2 * k, k:-k]
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1)
    good = inner & (length > 0)
    n[good] /= length[good, None]
    flip = (n * v).sum(-1) > 0
    n[flip] *= -1
    nmap[good, :3] = n[good]
    nmap[good, 3] = 1
    return vmap, nmap


def icp_sums(vcur, ncur, pose, vref, nref, ref_inv, K, centre, tukey=TUKEY_MM, gate=NORMAL_GATE, area=0.0,
             reach_mm=1e9, stride=1, base=(0.0, 0.0, 0.0, 1.0)):
    """The ICP sums (ICP_VALUES, float64) of a frame's maps (camera) against a ray-cast model's maps
    (world, seen from the reference camera; world -> reference camera = ref_inv), the frame placed at
    `pose` (camera -> world, mm). Every `stride`-th frame pixel each way takes part; each is matched to
    the model pixel it projects to, if that model point is above `base` (n . q + c > 0: the base
    plane, raised by what counts as base) and within `reach_mm`. The residual is the model normal's
    distance r = n . (p - q), used inside `tukey` with normals agreeing within `gate` (cos) and
    weighted (1 - (r / tukey)^2)^2 (1 - (|p - q| / reach)^2)^2, times the pixel's footprint
    z^2 * area when `area` > 0 (so a surface weighs the same seen near or far, densely or sparsely
    sampled). J = ((p - centre) x n, n): a small twist (rotation about `centre`, then translation)
    moves r by J . twist."""
    h, w = vcur.shape[:2]
    sel = np.zeros((h, w), bool)
    sel[::stride, ::stride] = True
    ok = (vcur[..., 3] > 0) & (ncur[..., 3] > 0) & sel
    pc, nc = vcur[ok][:, :3], ncur[ok]
    R, t = pose[:3, :3], pose[:3, 3]
    p = pc @ R.T + t
    q = p @ ref_inv[:3, :3].T + ref_inv[:3, 3]
    front = q[:, 2] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(q[:, 0] / q[:, 2] * K[0, 0] + K[0, 2] + 0.5)
        v = np.floor(q[:, 1] / q[:, 2] * K[1, 1] + K[1, 2] + 0.5)
    inside = front & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    j = np.where(inside, v * w + u, 0).astype(np.int64)
    qr, nr = vref.reshape(-1, 4)[j], nref.reshape(-1, 4)[j]
    base = np.asarray(base, np.float64)
    reach = ((p - qr[:, :3]) ** 2).sum(1) / reach_mm ** 2
    matched = inside & (qr[:, 3] > 0) & (nr[:, 3] > 0) & (qr[:, :3] @ base[:3] + base[3] > 0) & (reach < 1)
    ncw = nc[:, :3] @ R.T
    keep = matched & ((ncw * nr[:, :3]).sum(1) >= gate)
    r = (nr[:, :3] * (p - qr[:, :3])).sum(1)
    keep &= np.abs(r) < tukey
    wgt = (1 - (r[keep] / tukey) ** 2) ** 2 * (1 - reach[keep]) ** 2
    if area > 0:
        wgt = wgt * pc[keep, 2] ** 2 * area
    p, n, r = p[keep], nr[keep, :3], r[keep]
    J = np.concatenate([np.cross(p - np.asarray(centre), n), n], 1)
    A = (J[:, :, None] * J[:, None, :] * wgt[:, None, None]).sum(0)
    out = np.zeros(ICP_VALUES)
    out[:21] = A[np.triu_indices(6)]
    out[21:27] = (J * (wgt * r)[:, None]).sum(0)
    out[27:] = len(r), wgt.sum(), (r * r).sum(), len(pc), matched.sum()
    return out


def solve_icp(sums):
    """The twist (rotation vector rad, translation mm) that the sums ask for (one Gauss-Newton step)."""
    A = np.zeros((6, 6))
    A[np.triu_indices(6)] = sums[:21]
    A = A + np.triu(A, 1).T
    return np.linalg.solve(A + 1e-9 * np.eye(6), -sums[21:27])


def reduce_icp(partials):
    """Partial sums from the work-groups (ICP_GROUPS x ICP_VALUES float32) -> totals in float64."""
    return np.asarray(partials, np.float32).reshape(-1, ICP_VALUES).astype(np.float64).sum(0)


# --- marching cubes -----------------------------------------------------------------------------------
# Corner c of a cube is (c & 1, c >> 1 & 1, c >> 2 & 1) voxels from its first corner; edge e joins
# corners EDGES[e] = (a, b), b = a plus one step along EDGE_AXIS[e].
EDGES = [(a, a | 1 << axis) for axis in range(3) for a in range(8) if not a >> axis & 1]
EDGE_AXIS = [axis for axis in range(3) for a in range(8) if not a >> axis & 1]
FACES = [  # corners of each face in order around it
    (0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6)]


def triangle_table():
    """Triangles (as edge triples) for each of the 256 inside/outside patterns (bit c set: corner c is
    inside, TSDF < 0), made rather than typed: on each face the crossed edges are paired (where a face
    has four, the pairing that keeps its inside corners apart: the same choice from both cubes that
    share the face, so the surface has no cracks), the pairs are chained into loops round the inside
    corners, each loop is fanned into triangles that face the outside (positive TSDF)."""
    corner = np.array([(c & 1, c >> 1 & 1, c >> 2 & 1) for c in range(8)], float)
    edge_of = {frozenset(e): k for k, e in enumerate(EDGES)}
    table = []
    for pattern in range(256):
        inside = [bool(pattern >> c & 1) for c in range(8)]
        links = {}
        for face in FACES:
            crossed = [edge_of[frozenset((face[i], face[(i + 1) % 4]))] for i in range(4)
                       if inside[face[i]] != inside[face[(i + 1) % 4]]]
            if len(crossed) == 4:  # inside corners on a diagonal: cut each one off on its own
                start = 0 if inside[face[0]] else 1  # the face edges are in order round the face
                order = crossed[start:] + crossed[:start]
                pairs = [(order[3], order[0]), (order[1], order[2])]
            else:
                pairs = [tuple(crossed)] if crossed else []
            for a, b in pairs:
                links.setdefault(a, []).append(b)
                links.setdefault(b, []).append(a)
        tris, seen = [], set()
        for first in sorted(links):
            if first in seen:
                continue
            loop, prev, cur = [first], None, first
            while True:
                nxt = [e for e in links[cur] if e != prev and (e not in loop or e == first and len(loop) > 2)]
                nxt = nxt[0]
                if nxt == first:
                    break
                loop.append(nxt)
                prev, cur = cur, nxt
            seen.update(loop)
            mid = [(corner[EDGES[e][0]] + corner[EDGES[e][1]]) / 2 for e in loop]
            outward = sum((corner[EDGES[e][1]] - corner[EDGES[e][0]]) * (1 if inside[EDGES[e][0]] else -1) for e in loop)
            normal = sum(np.cross(mid[i] - mid[0], mid[i + 1] - mid[0]) for i in range(1, len(loop) - 1))
            if normal @ outward < 0:
                loop = loop[::-1]
            tris += [(loop[0], loop[i], loop[i + 1]) for i in range(1, len(loop) - 1)]
        table.append(tris)
    return table


def packed_triangle_table():
    """The table as one u32 array: 257 offsets (into the array, in edges), then the edge indices."""
    table = triangle_table()
    offsets, edges = [257], []
    for tris in table:
        edges += [e for t in tris for e in t]
        offsets.append(257 + len(edges))
    return np.array(offsets + edges, np.uint32)


def assemble_mesh(raw, layout):
    """Triangles from the marching-cubes kernel -> (vertices world mm, triangles, colours). Each triangle
    vertex comes as 5 u32: voxel i, j, k * 4 + edge axis, t (f32 along the edge), colour; the same
    edge from neighbouring cubes is one vertex."""
    v = np.asarray(raw, np.uint32).reshape(-1, 5)
    i, j = v[:, 0].astype(np.int64), v[:, 1].astype(np.int64)
    k, axis = (v[:, 2] >> 2).astype(np.int64), (v[:, 2] & 3).astype(np.int64)
    n = layout.dims * BRICK + 1
    keys = ((i * n[1] + j) * n[2] + k) * 3 + axis
    uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    t = v[first, 3].view(np.float32).astype(np.float64)
    ijk = np.stack([i[first], j[first], k[first]], 1).astype(np.float64)
    ijk[np.arange(len(uniq)), axis[first]] += t
    vertices = layout.origin + ijk * layout.voxel_mm
    return vertices, inverse.reshape(-1, 3).astype(np.int64), unpack_colour(v[first, 4])
