import numpy as np

try:
    from tribuutscan import fuse, track
except ImportError:  # needs Open3D (.venv)
    track = None


def box_points(rng, n=6000):
    """Points on three faces of a 60 x 40 x 30 mm box corner, ~250 mm away (metres)."""
    a = np.c_[rng.uniform(0, 60, n), rng.uniform(0, 40, n), np.zeros(n)]
    b = np.c_[rng.uniform(0, 60, n), np.zeros(n), rng.uniform(0, 30, n)]
    c = np.c_[np.zeros(n), rng.uniform(0, 40, n), rng.uniform(0, 30, n)]
    return (np.r_[a, b, c] + [-30, -20, 250]) / 1000


def plane_points(rng, n=15000):
    return (np.c_[rng.uniform(-60, 60, (n, 2)), np.zeros(n)] + [0, 0, 250]) / 1000


def move(t, pts):
    return pts @ t[:3, :3].T + t[:3, 3]


def small_motion(mm=(4.0, -3.0, 2.0), deg=(1.0, -0.5, 0.8)):
    return track.exp_se3(np.r_[np.radians(deg), mm])


def test_recovers_motion_on_a_box():
    if track is None:
        return
    rng = np.random.default_rng(1)
    target = box_points(rng)
    truth = small_motion()
    source = move(np.linalg.inv(truth), target) + rng.normal(0, 0.0003, target.shape)
    t, info = track.solve(track.levels(source), track.levels(target), np.eye(4))
    surface = np.linalg.norm(move(np.linalg.inv(truth) @ t, target) - target, axis=1) * 1000
    assert surface.max() < 0.3  # the object lines up (the camera position swings more: 25 cm lever arm)
    assert info["degeneracy"] > track.DEGENERATE


def test_flat_face_is_flagged_and_markers_hold_it():
    if track is None:
        return
    rng = np.random.default_rng(2)
    target = plane_points(rng)
    truth = track.exp_se3(np.r_[0, 0, 0, 6.0, 4.0, 0])  # slide along the plane: invisible to shape
    source = move(np.linalg.inv(truth), target)
    _, info = track.solve(fuse.levels(source), fuse.levels(target), np.eye(4))
    assert info["degeneracy"] < track.DEGENERATE
    markers_dst = np.array([[-40, -30, 250.0], [35, -20, 250], [10, 40, 250]])
    markers_src = move(np.linalg.inv(truth), markers_dst / 1000) * 1000
    t, info = track.solve(fuse.levels(source), fuse.levels(target), np.eye(4),
                          marker_pairs=(markers_src, markers_dst))
    assert np.linalg.norm(t[:3, 3] - truth[:3, 3]) * 1000 < 0.5 and info["markers_used"] == 3


def test_outliers_do_not_drag_the_solve():
    if track is None:
        return
    rng = np.random.default_rng(3)
    target = box_points(rng)
    truth = small_motion()
    source = move(np.linalg.inv(truth), target)
    junk = rng.uniform(-0.04, 0.04, (len(source) // 5, 3)) + [0, 0, 0.25]  # 20% clutter
    t, _ = track.solve(track.levels(np.r_[source, junk]), track.levels(target), np.eye(4))
    assert np.linalg.norm(t[:3, 3] - truth[:3, 3]) * 1000 < 0.5


def test_features_recover_a_known_move():
    try:
        from tribuutscan import features
    except ImportError:  # needs OpenCV (.venv)
        return
    rng = np.random.default_rng(5)
    pts = rng.uniform(-50, 50, (300, 3)) + [0, 0, 250]
    desc = rng.integers(0, 256, (300, 32), dtype=np.uint8)
    truth = small_motion() if track is not None else np.eye(4)
    moved = (pts - truth[:3, 3] * 1000) @ truth[:3, :3]  # the same points seen from the moved camera
    t, inliers = features.relative_pose((moved, desc), (pts, desc), rng)
    assert t is not None and inliers > 250
    assert np.abs(t - truth).max() < 1e-6
