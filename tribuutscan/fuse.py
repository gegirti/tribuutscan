"""Open3D ICP helpers that tracking uses (cloud.Scanner, track.py): per-level clouds and
coarse-to-fine point-to-plane ICP. Needs Open3D (.venv).

This was the first meshing prototype (frame-to-frame ICP odometry + a TSDF of 1 mm depth steps). Meshes
now come from fusion.py, which fuses each frame at the pose tracking gave it; only these helpers stay.
"""
import numpy as np
import open3d as o3d

ICP_LEVELS = ((0.006, 0.03), (0.003, 0.012), (0.002, 0.006))  # coarse to fine: (voxel, max match distance), metres
MIN_FITNESS = 0.3  # a frame whose ICP overlap is below this is not taken as tracked


def levels(full):
    """Downsample a cloud (Open3D, or an (N, 3) array in metres) once per ICP level, with normals."""
    if isinstance(full, np.ndarray):
        full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full))
    clouds = []
    for voxel, _ in ICP_LEVELS:
        cloud = full.voxel_down_sample(voxel)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
        clouds.append(cloud)
    return clouds


def icp(source, target, guess):
    """Coarse-to-fine point-to-plane ICP; returns the finest level's result."""
    for (_, max_dist), src, tgt in zip(ICP_LEVELS, source, target):
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, max_dist, guess, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        guess = result.transformation
    return result
