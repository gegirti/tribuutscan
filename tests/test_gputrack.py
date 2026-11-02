import numpy as np

from tribuutscan import gpu


def tracker_or_none():
    """gputrack when wgpu and an adapter are there (else these tests pass by)."""
    try:
        from tribuutscan import gputrack
    except ImportError:
        return None
    return gputrack if gputrack.kind() else None


def scene_depth(pose, K, size):
    """Depth (mm) of three balls in front of a tilted wall, seen from `pose` (camera -> world, mm): a
    scene that pins all six directions of motion."""
    d = gpu.rays(K, size) @ pose[:3, :3].T
    o = pose[:3, 3]
    best = np.full(d.shape[:2], np.inf)
    for c, r in (((0, 0, 250), 30.0), ((45, -25, 265), 14.0), ((-40, 30, 255), 18.0)):
        oc = o - np.array(c, float)
        a, b, cc = (d * d).sum(-1), 2 * (d @ oc), oc @ oc - r * r
        disc = b * b - 4 * a * cc
        with np.errstate(invalid="ignore"):
            s = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
        best = np.minimum(best, s)
    n, c0 = np.array([0.15, -0.1, -1.0]) / np.linalg.norm([0.15, -0.1, -1.0]), 300.0  # wall n . x + c0 = 0
    with np.errstate(divide="ignore"):
        s = -(o @ n + c0) / (d @ n)
    best = np.minimum(best, np.where(s > 0, s, np.inf))
    return np.where(np.isfinite(best), best, 0.0)  # along the rays (x/z, y/z, 1): this is z


def test_the_gpu_solve_finds_a_known_motion():
    """One frame fused, the next seen from 1.5 mm and 1 degree away: the solve (projective ICP on the GPU,
    the 6x6 on the CPU) finds the second pose from the first, on the GPU and on the CPU fallback."""
    gputrack = tracker_or_none()
    if gputrack is None:
        return
    from tribuutscan import calib, track
    w, h = 320, 200
    spec = {"size": [w, h], "depth": {"fx": 300.0, "fy": 300.0, "cx": 159.5, "cy": 99.5, "dist": [0, 0, 0, 0, 0]},
            "law": {"a_mm": 289570.0, "b": 873.4}, "window_mm": [150.0, 330.0],
            "colour_from_depth": {"ax": 1.0, "bx": 0.0, "ay": 1.0, "by": 0.0}}
    cam = calib.Camera(spec)
    K = cam.intrinsic_matrix()
    first = np.eye(4)
    move = track.exp_se3(np.r_[np.radians([0.6, -0.5, 0.5]), [1.0, -0.8, 0.7]])
    second = first @ move
    raw = lambda pose: np.where(scene_depth(gpu.pose_mm(pose), K, (w, h)) > 0,
                                cam.raw_from_mm(np.maximum(scene_depth(gpu.pose_mm(pose), K, (w, h)), 1)), 0).astype(np.float32)
    for kind in ("gpu", "cpu"):
        try:
            t = gputrack.Tracker(cam, (w, h), (0, 0, 260), kind)
        except KeyError:  # no adapter of this kind
            continue
        t.fuse({"depth": raw(first)}, None, first)
        frame = {"depth": raw(second)}
        t.source(frame, np.argwhere(np.ones((h // 2, w // 2), bool))[:, ::-1] * 2, 2)
        found, info = t.solve(first, centre_mm=(0, 0, 255), radius_mm=60)
        err = np.linalg.inv(second) @ found
        assert np.linalg.norm(err[:3, 3]) * 1000 < 0.05, (kind, err[:3, 3] * 1000)
        assert np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))) < 0.02
        assert info["fitness"] > 0.8 and info["rmse_mm"] < 0.1 and info["degeneracy"] > 0.001
        t.close()


def test_keyframe_points_are_one_per_cell():
    gputrack = tracker_or_none()
    if gputrack is None:
        return
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 10, (5000, 3))
    pts = gputrack.grid_points(xyz, 2.0)
    assert len(pts) == 125 and len(np.unique(np.floor(pts / 2), axis=0)) == 125
