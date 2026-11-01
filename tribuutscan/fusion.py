"""Fuse a scan's depth frames into one surface (a truncated signed distance volume, TSDF) and mesh it.

    .venv/bin/python -m tribuutscan.fusion captures/<stamp>-scan [--all] [--voxel 0.5] [--min-weight 4] [--force]

Writes mesh.ply (metres, y up, z toward the scanner, vertex colour: like cloud.ply) and mesh.stl (mm,
same axes, for slicers) into the scan. The frames come from the scan's source recordings
(session.json "sources"), each at its tracked pose and only over its own object/base pixels
(labels.npz). By default only the frames the cloud merged (the keyframes, whose poses Optimise
corrects): on the marker scan they gave p90 0.34 mm against Creality's mesh, every tracked frame 0.53
(frame_poses.txt, with corrections blended between keyframes; 0.55 held constant); --all takes them all. A scan is fused only under the camera model it was tracked with (session.json "camera"),
unless --force. The base and the surroundings are cut at extraction with the cloud's own rules
(Scanner.output_mask): the volume cannot drop what it took in, and the base may be found late.

`Engine` is the volume itself and knows nothing about the scanner: depth in mm on a pinhole grid,
colour on the same grid, a camera -> world pose. This one is Open3D's VoxelBlockGrid on the CPU (the
reference a GPU engine must match). Depth is fused as float mm (14-bit steps kept), after the lens
distortion is taken out by a fixed nearest-pixel lookup (the board model has some; the factory
model's depth is already pinhole) and after the merge's cleaning (prep.clean).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c

from . import calib
from . import cloud as cl
from . import frames as fr
from . import pointcloud as pc
from . import prep
from .calib import CAMERA
from .image import yuyv_pixels

VOXEL_MM = 0.5  # close range: a pixel covers 0.26 mm at 25 cm
TRUNC_VOXELS = 3.0  # truncation 1.5 mm: thinner than a 1.8 mm plate, so both its faces stay apart
MIN_WEIGHT = 4  # a surface must be seen in at least this many frames (marker scan: 3 left more noise at the
#                edges, p90 0.62 against 0.55 mm; 6 trimmed more for 0.50)
MIN_PIXELS = 100  # frames with less depth than this (after masking and cleaning) are skipped
PIECE_SHARE = 0.01  # pieces with less area than this share of the largest piece are crumbs: dropped
BLOCK = 8  # voxels per block side (4 mm blocks at 0.5 mm: little empty space stored)
FLIP = np.diag([1.0, -1.0, -1.0])  # camera axes -> y up, z toward the scanner (cloud.FLIP)


class Engine:
    """TSDF volume on Open3D's VoxelBlockGrid (CPU). Depth (H, W) float mm on the pinhole grid K, 0 where
    none; colour (H, W, 3) on the same grid; pose camera -> world (4x4, translation in metres)."""
    name = "open3d-cpu"

    def __init__(self, K, voxel_mm=VOXEL_MM, trunc_voxels=TRUNC_VOXELS, far_mm=1000.0, blocks=40000):
        self.K = o3c.Tensor(np.asarray(K, float), o3c.float64)
        self.voxel_mm, self.trunc, self.far_m = voxel_mm, float(trunc_voxels), far_mm / 1000
        self.grid = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight", "color"), (o3c.float32, o3c.uint16, o3c.uint16),
                                                  (1, 1, 3), voxel_mm / 1000, BLOCK, blocks, o3c.Device("CPU:0"))
        self.frames, self.empty = 0, 0

    def integrate(self, depth_mm, rgb, pose):
        """Fuse one frame; a frame without depth is counted in `empty` (the volume refuses it)."""
        if np.count_nonzero(depth_mm) < MIN_PIXELS:
            self.empty += 1
            return
        depth = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth_mm, np.float32)))
        colour = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb, np.float32)))
        extrinsic = o3c.Tensor(np.linalg.inv(pose))  # world -> camera
        blocks = self.grid.compute_unique_block_coordinates(depth, self.K, extrinsic, 1000.0, self.far_m, self.trunc)
        self.grid.integrate(blocks, depth, colour, self.K, self.K, extrinsic, 1000.0, self.far_m, self.trunc)
        self.frames += 1

    def mesh(self, min_weight=MIN_WEIGHT):
        """(vertices (N, 3) world mm, triangles (M, 3), colours (N, 3) uint8) of the surface seen in at
        least `min_weight` frames (Open3D keeps weights strictly above its threshold)."""
        m = self.grid.extract_triangle_mesh(weight_threshold=float(min_weight) - 0.5)
        if "positions" not in m.vertex or not len(m.vertex.positions):
            return np.empty((0, 3)), np.empty((0, 3), int), np.empty((0, 3), np.uint8)
        vertices = m.vertex.positions.numpy().astype(np.float64) * 1000
        colours = np.clip(np.round(m.vertex.colors.numpy()), 0, 255).astype(np.uint8)
        return vertices, m.triangle.indices.numpy().astype(np.int64), colours

    def blocks(self):
        return self.grid.hashmap().size()


class Pinhole:
    """Fixed nearest-pixel lookup from the depth camera's (distorted) image onto a pinhole grid with
    the same fx, fy, cx, cy; nothing to do for an undistorted model."""

    def __init__(self, camera=CAMERA):
        self.K = camera.intrinsic_matrix()
        self.index = None
        if np.any(camera.dist):
            vv, uu = np.mgrid[0:camera.height, 0:camera.width]
            xd, yd = calib.distort((uu - camera.cx) / camera.fx, (vv - camera.cy) / camera.fy, camera.dist)
            u = np.round(xd * camera.fx + camera.cx).astype(int)
            v = np.round(yd * camera.fy + camera.cy).astype(int)
            inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
            self.index = np.where(inside, v * camera.width + u, -1).ravel()

    def __call__(self, image):
        if self.index is None:
            return image
        flat = image.reshape(len(self.index), -1)
        out = flat[np.maximum(self.index, 0)]
        out[self.index < 0] = 0
        return out.reshape(image.shape)


def frame_images(frame, mask=None):
    """(depth mm float32, colour float32) on the sensor grid for one frame set: cleaned as for merging,
    only inside `mask` (H, W bool) when given, colour looked up per pixel at that pixel's distance."""
    d = frame["depth"]
    h, w = d.shape
    raw = np.zeros_like(d)
    r0, r1, c0, c1 = 0, h, 0, w
    if mask is not None:  # clean only around the mask (8 px margin: cleaning looks at neighbours)
        rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
        if not len(rows):
            return np.zeros((h, w), np.float32), np.zeros((h, w, 3), np.float32)
        r0, r1, c0, c1 = max(rows[0] - 8, 0), min(rows[-1] + 9, h), max(cols[0] - 8, 0), min(cols[-1] + 9, w)
    raw[r0:r1, c0:c1] = prep.clean(d[r0:r1, c0:c1], 1, origin=(c0, r0))
    if mask is not None:
        raw = np.where(mask, raw, 0)
    v, u = np.nonzero(raw > 0)
    z = CAMERA.depth_mm(raw[v, u])
    depth = np.zeros((h, w), np.float32)
    depth[v, u] = z
    rgb = np.zeros((h, w, 3), np.float32)
    if len(u):
        src_x, src_y = pc.color_source(u, v, w, h, z)
        rgb[v, u] = frame["color"][src_y, src_x] if "color" in frame else yuyv_pixels(frame["yuyv"], w, src_x, src_y)
    return depth, rgb


def label_mask(labels, step, shape):
    """Full-size pixel mask (object or base) from a frame's 1/step labels, grown by one cell."""
    keep = labels > 0
    grown = keep.copy()
    grown[1:] |= keep[:-1]
    grown[:-1] |= keep[1:]
    grown[:, 1:] |= grown[:, :-1].copy()
    grown[:, :-1] |= grown[:, 1:].copy()
    return np.repeat(np.repeat(grown, step, 0), step, 1)[:shape[0], :shape[1]]


def scan_frames(scanner, which="merged", check=None):
    """(frame, pose, mask) for a scan's tracked frames ("all") or its merged ones, read from the source
    recordings it notes (session.json "sources"); `check()` runs after the first frame is read (it set
    the recording's depth mode)."""
    poses = dict(scanner.poses if which == "merged" else scanner.frame_poses())
    if not scanner.sources:
        raise ValueError("this scan does not note its recordings (saved before 2026-11-01, or without "
                         "keep raw frames): it cannot be fused again")
    for k, source in enumerate(scanner.sources):
        end = scanner.sources[k + 1]["first_index"] if k + 1 < len(scanner.sources) else None
        rec = Path(source["rec"])
        if not rec.exists():
            raise FileNotFoundError(f"recording {rec} is gone")
        for i, frame in enumerate(fr.read_recording(rec)):
            if k == 0 and i == 0 and check is not None:
                check()
            index = source["first_index"] + i
            if end is not None and index >= end:
                break
            if index in poses:
                labels = scanner.labels.get(index)
                mask = None if labels is None else label_mask(labels, cl.LABEL_STEP, frame["depth"].shape)
                yield frame, poses[index], mask


def check_camera(directory, force=False):
    """The scan's camera (session.json) against the one in use now; raises on a mismatch unless forced."""
    noted = json.loads((Path(directory) / "session.json").read_text()).get("camera")
    if noted and noted.get("fingerprint") != CAMERA.fingerprint() and not force:
        raise ValueError(f"the scan was tracked with the {noted.get('source')} camera ({noted.get('fingerprint')}), "
                         f"this is the {CAMERA.source} one ({CAMERA.fingerprint()}): set TRIBUUTSCAN_CAMERA, "
                         "or force")


def cut(vertices, triangles, keep):
    """Triangles whose three vertices are kept, with the unused vertices dropped; returns (vertices
    index map, triangles)."""
    tri = triangles[keep[triangles].all(1)]
    used = np.unique(tri)
    new = np.full(len(vertices), -1)
    new[used] = np.arange(len(used))
    return used, new[tri]


def fuse(items, voxel_mm=VOXEL_MM, trunc_voxels=TRUNC_VOXELS, log=print):
    """An Engine with every (frame, pose, mask) of `items` fused, and ms per frame by step."""
    engine, pinhole = None, Pinhole()
    timing = {"read": 0.0, "prepare": 0.0, "integrate": 0.0}
    t = time.perf_counter()
    for frame, pose, mask in items:
        if engine is None:  # after the first frame: its recording set the depth mode (the window)
            engine = Engine(pinhole.K, voxel_mm, trunc_voxels, CAMERA.window_mm[1] + 20)
        t1 = time.perf_counter()
        depth, rgb = frame_images(frame, mask)
        depth, rgb = pinhole(depth), pinhole(rgb)
        t2 = time.perf_counter()
        engine.integrate(depth, rgb, pose)
        t3 = time.perf_counter()
        for key, dt in (("read", t1 - t), ("prepare", t2 - t1), ("integrate", t3 - t2)):
            timing[key] += dt
        if engine.frames and engine.frames % 100 == 0:
            log(f"fused {engine.frames} frames, {engine.blocks()} blocks")
        t = time.perf_counter()
    if engine is None:
        raise ValueError("no tracked frames to fuse")
    return engine, {k: round(1000 * v / (engine.frames + engine.empty), 1) for k, v in timing.items()}


def without_crumbs(vertices, triangles, share=PIECE_SHARE):
    """Keep-mask over triangles: pieces (connected by triangles) of at least `share` of the largest
    piece's area. On the marker scan 571 of 577 pieces were crumbs, 1.1% of the area."""
    if not len(triangles):
        return np.ones(0, bool)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles.astype(np.int32)))
    ids, _, areas = mesh.cluster_connected_triangles()
    areas = np.asarray(areas)
    return (areas >= share * areas.max())[np.asarray(ids)]


def extract(engine, keep=None, min_weight=MIN_WEIGHT):
    """The engine's mesh (vertices world mm, triangles, colours), cut to `keep` (world vertices -> bool
    mask) and without crumbs."""
    vertices, triangles, colours = engine.mesh(min_weight)
    if keep is not None:
        used, triangles = cut(vertices, triangles, keep(vertices))
        vertices, colours = vertices[used], colours[used]
    triangles = triangles[without_crumbs(vertices, triangles)]
    used = np.unique(triangles)
    return vertices[used], np.searchsorted(used, triangles), colours[used]


def fuse_scan(directory, which="merged", voxel_mm=VOXEL_MM, trunc_voxels=TRUNC_VOXELS, min_weight=MIN_WEIGHT, force=False,
              log=print, scanner=None):
    """Fuse a saved scan (`scanner`: the same scan already in memory, saves restoring it); returns
    (vertices world mm, triangles, colours, report). Nothing is written."""
    t0 = time.time()
    if scanner is None:
        scanner = cl.Scanner.restore(directory, log=lambda m: None)
    else:
        scanner.flush()
    engine, ms = fuse(scan_frames(scanner, which, lambda: check_camera(directory, force)), voxel_mm, trunc_voxels, log)
    t1 = time.time()
    vertices, triangles, colours = extract(engine, scanner.output_mask, min_weight)
    report = {"frames": engine.frames, "empty_frames": engine.empty, "which": which, "engine": engine.name, "voxel_mm": voxel_mm,
              "trunc_mm": voxel_mm * trunc_voxels, "min_weight": min_weight, "blocks": engine.blocks(),
              "vertices": int(len(vertices)), "triangles": int(len(triangles)), "remove_base": scanner.remove_base,
              "camera": {"source": CAMERA.source, "fingerprint": CAMERA.fingerprint()}, "ms_per_frame": ms,
              "fuse_s": round(t1 - t0, 1), "extract_s": round(time.time() - t1, 1)}
    return vertices, triangles, colours, report


def write_mesh(directory, vertices, triangles, colours):
    """mesh.ply (metres, like cloud.ply, vertex colour and normals) and mesh.stl (mm, for slicers);
    same axes. Written to temporary names first: a page polling for the mesh never reads half a file."""
    directory = Path(directory)
    xyz = vertices @ FLIP.T
    tmp = directory / "mesh.ply.tmp"
    pc.write_mesh_ply(tmp, (xyz / 1000).astype(np.float32), colours, triangles, pc.vertex_normals(xyz, triangles))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector(triangles.astype(np.int32)))
    mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(directory / "mesh.tmp.stl"), mesh)  # Open3D picks the format by the extension
    (directory / "mesh.tmp.stl").replace(directory / "mesh.stl")
    tmp.replace(directory / "mesh.ply")


def make_mesh(directory, which="merged", voxel_mm=VOXEL_MM, min_weight=MIN_WEIGHT, force=False, log=print, scanner=None):
    """Fuse a scan and write mesh.ply, mesh.stl and mesh.json; returns the report."""
    vertices, triangles, colours, report = fuse_scan(directory, which, voxel_mm, min_weight=min_weight, force=force,
                                                     log=log, scanner=scanner)
    write_mesh(directory, vertices, triangles, colours)
    (Path(directory) / "mesh.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    paths, voxel, weight, rest = [], VOXEL_MM, MIN_WEIGHT, iter(sys.argv[1:])
    for a in rest:
        if a == "--voxel":
            voxel = float(next(rest))
        elif a == "--min-weight":
            weight = float(next(rest))
        elif not a.startswith("--"):
            paths.append(Path(a))
    if not paths:
        sys.exit(__doc__)
    report = make_mesh(paths[0], "all" if "--all" in sys.argv else "merged", voxel, weight, "--force" in sys.argv)
    print(json.dumps(report, indent=2))
    print(f"-> {paths[0] / 'mesh.ply'}, mesh.stl")


if __name__ == "__main__":
    main()
