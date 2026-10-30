"""Fuse a recording (captures/<stamp>-rec/) into one coloured mesh. ROUGH prototype.

    .venv/bin/python -m tribuutscan.fuse captures/<stamp>-rec [step]

It uses the ESTIMATED depth law, field of view and colour registration from pointcloud.py, so
sizes and shapes may be off. Tracking is point-to-plane ICP on the depth points (geometry only, so
it needs shape detail in view): a frame-to-frame guess, refined against the model fused so far to
limit drift. Fusion is a TSDF voxel block grid
(Open3D's legacy ScalableTSDFVolume returned empty meshes on this data, so the tensor API is used).
Writes mesh.ply and trajectory.txt into the recording folder. Units: metres, x right, y up,
z toward the scanner (Blender-friendly, right-handed).

Needs Open3D, kept out of the core package:
    python3 -m venv --system-site-packages .venv && .venv/bin/pip install open3d
"""
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c

from . import frames as fr
from . import pointcloud as pc

VOXEL_M = 0.0015  # TSDF voxel
TRUNC_M = 0.006  # TSDF truncation
ICP_LEVELS = ((0.006, 0.03), (0.003, 0.012), (0.002, 0.006))  # coarse to fine: (voxel, max match distance)
MIN_FITNESS = 0.3  # skip a frame whose ICP overlap is below this
MAX_STEP_M, MAX_STEP_DEG = 0.03, 15.0  # reject jumps no hand moves between two used frames
MAX_FIX_M, MAX_FIX_DEG = 0.01, 5.0  # largest correction the model refinement may apply
MODEL_EVERY = 3  # re-extract the model point cloud every N fused frames
MIN_PIECE = 0.02  # drop disconnected mesh pieces smaller than this share of all triangles
FAR_M = 0.40  # ignore anything further (the window ends ~0.32 m, estimate)
MIN_WEIGHT = 3.0  # a surface must be seen in at least this many frames
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # camera (y down, z forward) -> y up, z toward scanner


def intrinsic(w=fr.W, h=fr.H):
    f = (w / 2) / np.tan(np.radians(pc.HFOV_DEG) / 2)
    return o3d.camera.PinholeCameraIntrinsic(w, h, f, f, w / 2, h / 2)


def images(frame):
    """(depth in mm as uint16, registered colour as uint8) for one frame set."""
    d = frame["depth"]
    mm = np.where(d > 0, pc.depth_to_mm(d), 0).astype(np.uint16)
    return mm, np.ascontiguousarray(pc.register_color(frame["color"]))


def icp_clouds(mm, color, intr):
    image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color), o3d.geometry.Image(mm),
        depth_scale=1000.0, depth_trunc=FAR_M, convert_rgb_to_intensity=False)
    return downsample(o3d.geometry.PointCloud.create_from_rgbd_image(image, intr))


def downsample(full):
    """One downsampled cloud with normals per ICP level."""
    clouds = []
    for voxel, _ in ICP_LEVELS:
        cloud = full.voxel_down_sample(voxel)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
        clouds.append(cloud)
    return clouds


def icp(source, target, guess, levels=ICP_LEVELS):
    """Coarse-to-fine point-to-plane ICP; returns the finest level's result."""
    for (_, max_dist), src, tgt in zip(levels, source, target):
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, max_dist, guess, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        guess = result.transformation
    return result


def plausible(t, max_m=MAX_STEP_M, max_deg=MAX_STEP_DEG):
    angle = np.degrees(np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1)))
    return np.linalg.norm(t[:3, 3]) <= max_m and angle <= max_deg


def drop_small_pieces(mesh):
    """Remove floating scraps: connected pieces under MIN_PIECE of the triangles."""
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels, sizes = np.asarray(labels), np.asarray(sizes)
    if len(sizes):
        mesh.remove_triangles_by_mask(sizes[labels] < MIN_PIECE * len(labels))
        mesh.remove_unreferenced_vertices()


def fuse(directory, step=1):
    intr = intrinsic()
    volume = o3d.t.geometry.VoxelBlockGrid(
        ("tsdf", "weight", "color"), (o3c.float32, o3c.float32, o3c.float32), (1, 1, 3),
        VOXEL_M, 16, 50000, o3c.Device("CPU:0"))
    K = o3c.Tensor(intr.intrinsic_matrix, o3c.float64)
    trunc = TRUNC_M / VOXEL_M
    pose, motion, prev, model = np.eye(4), np.eye(4), None, None
    poses, used, skipped = [], 0, 0
    for i, frame in enumerate(fr.read_recording(directory)):
        if i % step:
            continue
        mm, color = images(frame)
        cloud = icp_clouds(mm, color, intr)
        if len(cloud[-1].points) < 500:
            skipped += 1
            continue
        if prev is not None:
            result = icp(cloud, prev, motion)
            if result.fitness < MIN_FITNESS or not plausible(result.transformation):
                skipped += 1
                continue
            motion = result.transformation  # current -> previous; reused as the next guess
            pose = pose @ motion
            if model is not None:  # refine against the fused model (fine levels only)
                fix = icp(cloud[1:], model[1:], pose, ICP_LEVELS[1:])
                if fix.fitness >= MIN_FITNESS and plausible(np.linalg.inv(pose) @ fix.transformation,
                                                            MAX_FIX_M, MAX_FIX_DEG):
                    pose = fix.transformation
        depth, rgb = o3d.t.geometry.Image(o3c.Tensor(mm)), o3d.t.geometry.Image(o3c.Tensor(color))
        extrinsic = o3c.Tensor(np.linalg.inv(pose))
        blocks = volume.compute_unique_block_coordinates(depth, K, extrinsic, 1000.0, FAR_M, trunc)
        volume.integrate(blocks, depth, rgb, K, K, extrinsic, 1000.0, FAR_M, trunc)
        poses.append((i, pose))
        prev, used = cloud, used + 1
        if used % MODEL_EVERY == 0:
            surface = volume.extract_point_cloud(weight_threshold=MIN_WEIGHT).to_legacy()
            model = downsample(surface) if len(surface.points) >= 500 else None
    mesh = volume.extract_triangle_mesh(weight_threshold=MIN_WEIGHT).to_legacy()
    mesh.transform(FLIP)
    drop_small_pieces(mesh)
    mesh.compute_vertex_normals()
    return mesh, poses, used, skipped


def main():
    directory = Path(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t0 = time.time()
    mesh, poses, used, skipped = fuse(directory, step)
    o3d.io.write_triangle_mesh(str(directory / "mesh.ply"), mesh)
    with open(directory / "trajectory.txt", "w") as f:
        for i, pose in poses:
            f.write(f"{i} " + " ".join(f"{v:.6f}" for v in pose.ravel()) + "\n")
    travel = np.linalg.norm(poses[-1][1][:3, 3]) * 100 if poses else 0
    print(f"{used} frames fused, {skipped} skipped, {time.time() - t0:.0f} s")
    print(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles -> {directory / 'mesh.ply'}")
    print(f"estimated scanner travel start->end: {travel:.1f} cm (scale is an estimate)")


if __name__ == "__main__":
    main()
