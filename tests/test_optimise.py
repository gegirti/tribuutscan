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


def test_scan_pieces_are_joined_into_one_world_and_a_crumb_is_left_out():
    """Two pieces of one scan (cloud.Scanner.start_piece) see the same object from overlapping views, the
    second in a world 20 mm and 12 degrees off the first; join_pieces finds that by shape, checks it with the
    joint solve and moves the second piece's frames into the first's world. A third piece of 2 keyframes is
    noise: left out, its frames gone from the scan."""
    if optimise is None:
        return
    from tribuutscan import cloud as cl
    rng = np.random.default_rng(4)
    faces = [np.c_[rng.uniform(-40, 40, 2500), rng.uniform(-30, 30, 2500), np.full(2500, 25.0)],
             np.c_[np.full(2500, 40.0), rng.uniform(-30, 30, 2500), rng.uniform(-25, 25, 2500)],
             np.c_[rng.uniform(-40, 40, 2500), np.full(2500, 30.0), rng.uniform(-25, 25, 2500)],
             np.c_[rng.uniform(-15, 5, 800), rng.uniform(-10, 10, 800), np.full(800, 40.0)],  # a step on the top
             np.c_[np.full(800, 5.0), rng.uniform(-10, 10, 800), rng.uniform(25, 40, 800)]]
    shape = np.concatenate(faces)
    to_cam = lambda pose, pts: (pts - pose[:3, 3] * 1000) @ pose[:3, :3]
    off = track.exp_se3(np.r_[0.0, 0.0, np.radians(12), 0.020, -0.008, 0.004])  # piece 2's world -> piece 1's
    s = cl.Scanner("small", log=lambda m: None)
    s.pieces = [{"first_index": 0}, {"first_index": 20}, {"first_index": 40}]
    s.piece_maps = [type(s.marker_map)(add_mm=3.0), type(s.marker_map)(add_mm=3.0)]
    s.piece_scenes = [s.scene, s.scene]
    truth = {}
    for i, a in [(i, np.radians(10 * i)) for i in range(10)] + [(20 + i, np.radians(40 + 10 * i)) for i in range(10)] \
            + [(40, np.radians(5)), (41, np.radians(6))]:
        true = look_at(np.array([250 * np.cos(a), 250 * np.sin(a), 150.0]))
        pose = true if i < 20 or i >= 40 else np.linalg.inv(off) @ true  # (as piece 2 tracked it)
        pts = to_cam(true, shape)
        s.keyframes.append((i, pose, pts, None, np.zeros((0, 3))))
        s.poses.append((i, pose))
        s.merged_data[i] = (pts.astype(np.float32), np.zeros((len(pts), 3), np.uint8), np.zeros((len(pts), 3), np.float32))
        s.track_log.append((i, "tracked", pose))
        truth[i] = true
    report = optimise.join_pieces(s, log=lambda m: None)
    assert report["joined"][2]["by"] == "shape" and report["left_out"] == [3] and report["anchor"] == 1
    for i, pose in s.poses:  # piece 2's frames now where they truly were
        assert np.linalg.norm(pose[:3, 3] - truth[i][:3, 3]) * 1000 < 1.5, i
    assert all(i < 40 for i, _ in s.poses) and all(i not in s.merged_data for i in (40, 41))
    assert [st for i, st, _ in s.track_log if i >= 40] == ["left out", "left out"]
    assert s.left_pieces[0]["piece"] == 3 and len(s.left_pieces[0]["xyz"])
    assert [p["joined"] for p in s.pieces] == [True, True, False] and len(s.piece_joins) == 1
    assert optimise.join_pieces(s, log=lambda m: None) is None  # joined once
