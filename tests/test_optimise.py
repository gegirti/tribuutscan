import numpy as np

try:
    from tribuutscan import optimise, track
except ImportError:  # needs Open3D (.venv)
    optimise = None


def look_at(eye, target=np.zeros(3)):
    f = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, np.cross(f, x), f], 1)
    m[:3, 3] = eye / 1000
    return m


def test_loop_closure_pulls_drift_back():
    if optimise is None:
        return
    rng = np.random.default_rng(9)
    faces = [np.c_[rng.uniform(-40, 40, 3000), rng.uniform(-30, 30, 3000), np.full(3000, 25.0)],
             np.c_[np.full(3000, 40.0), rng.uniform(-30, 30, 3000), rng.uniform(-25, 25, 3000)],
             np.c_[np.full(3000, -40.0), rng.uniform(-30, 30, 3000), rng.uniform(-25, 25, 3000)],
             np.c_[rng.uniform(-40, 40, 3000), np.full(3000, 30.0), rng.uniform(-25, 25, 3000)],
             np.c_[rng.uniform(-40, 40, 3000), np.full(3000, -30.0), rng.uniform(-25, 25, 3000)]]
    box = np.concatenate(faces)
    markers = np.array([[-20, 10, 25.0], [15, -12, 25], [30, 20, 25], [40, 5, -10], [-5, 30, 8]])
    angles = np.radians(np.arange(0, 360, 30))
    truth = [look_at(np.array([250 * np.cos(a), 250 * np.sin(a), 120.0])) for a in angles]
    drift = track.exp_se3(np.r_[0.0, np.radians(0.25), np.radians(0.1), 0.8, -0.5, 0.3])
    est = [truth[0]]
    for a, b in zip(truth, truth[1:]):
        est.append(est[-1] @ np.linalg.inv(a) @ b @ drift)
    to_cam = lambda pose, pts: (pts - pose[:3, 3] * 1000) @ pose[:3, :3]
    kfs = [{"pose": e, "points": to_cam(t, box), "feat": None, "markers": to_cam(t, markers)}
           for t, e in zip(truth, est)]
    before = np.linalg.norm(est[-1][:3, 3] - truth[-1][:3, 3]) * 1000
    poses, report = optimise.optimise_poses(kfs, rng, log=lambda m: None)
    after = np.linalg.norm(poses[-1][:3, 3] - truth[-1][:3, 3]) * 1000
    assert report["loops_kept"] >= 1
    assert after < before / 2


def test_object_features_leave_the_room_out():
    """For a scan whose room stood while the object turned, Optimise matches keyframes' colour features on the
    object only (its ball here: no base was found), not the room's."""
    if optimise is None:
        return
    from tribuutscan.scene import Scene
    scene = Scene("still")
    scene.ball = (np.array([0.0, 0.0, 300.0]), 80.0, 80.0)
    pts = np.array([[0.0, 10.0, 290.0], [30.0, -20.0, 320.0], [250.0, 0.0, 600.0], [-200.0, 50.0, 550.0]])
    desc = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    kept = optimise.object_features((pts, desc), scene, np.eye(4))
    assert len(kept[0]) == 2 and np.array_equal(kept[1], desc[:2])
    assert optimise.object_features(None, scene, np.eye(4)) is None
