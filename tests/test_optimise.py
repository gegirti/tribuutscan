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


def box_faces(rng, n=2500):
    """A box's faces (mm): top z 25, sides x +-40, y +-30, and a step on the top."""
    return [np.c_[rng.uniform(-40, 40, n), rng.uniform(-30, 30, n), np.full(n, 25.0)],
            np.c_[np.full(n, 40.0), rng.uniform(-30, 30, n), rng.uniform(-25, 25, n)],
            np.c_[rng.uniform(-40, 40, n), np.full(n, 30.0), rng.uniform(-25, 25, n)],
            np.c_[np.full(n, -40.0), rng.uniform(-30, 30, n), rng.uniform(-25, 25, n)],
            np.c_[rng.uniform(-40, 40, n), np.full(n, -30.0), rng.uniform(-25, 25, n)],
            np.c_[rng.uniform(-15, 5, 800), rng.uniform(-10, 10, 800), np.full(800, 40.0)],
            np.c_[np.full(800, 5.0), rng.uniform(-10, 10, 800), rng.uniform(25, 40, 800)]]


def test_a_join_is_judged_both_ways():
    """A piece that saw much more of the object than the one it joins (the 2026-11-06 scan's 45 degree pass against
    the level view) fails the gate onto it, as the share of its points that find a partner is small; the other
    way the first lies on it whole. join_solve takes that, inverted. The box seen from one side also fits turned
    180 degrees (the other way passes there too): with both candidates passing, join_pick refuses the join."""
    if optimise is None:
        return
    rng = np.random.default_rng(5)
    faces = box_faces(rng)
    seen_first = np.concatenate([faces[0], faces[2], faces[5], faces[6]])  # the top, one side, the step
    seen_more = np.concatenate(faces + [faces[1] + [120.0, 0, 0]])  # every face, and a second box beside it
    off = track.exp_se3(np.r_[0.0, np.radians(2), np.radians(3), 0.004, -0.003, 0.002])  # piece 2's world -> 1's
    src = (seen_more - off[:3, 3] * 1000) @ off[:3, :3]  # (as piece 2 has it)
    ls, ld = track.levels(src / 1000), track.levels(seen_first / 1000)
    start = off @ track.exp_se3(np.r_[np.radians(0.5), 0.0, 0.0, 0.001, 0.001, 0.0])  # a candidate, a little off
    _, one_way = track.solve(ls, ld, start)
    assert not optimise.join_ok(one_way) and one_way["fitness"] < optimise.JOIN_MIN_FITNESS
    t, info, back = optimise.join_solve(ls, ld, start)
    assert back and optimise.join_ok(info)
    moved = src @ t[:3, :3].T + t[:3, 3] * 1000
    assert np.abs(moved - seen_more).max() < 1.0  # piece 2's points where they truly are
    right = (t, info, "shape", back)
    turned = optimise.join_solve(ls, ld, off @ track.exp_se3(np.r_[0.0, 0.0, np.pi, 0.0, 0.0, 0.0]))
    assert turned is not None and np.abs(src @ turned[0][:3, :3].T + turned[0][:3, 3] * 1000 - seen_more).max() > 50
    assert optimise.join_pick([right], src) is right
    assert optimise.join_pick([right, (*turned[:2], "features", turned[2])], src) is None
    nudged = (track.exp_se3(np.r_[0.0, 0.0, np.radians(1), 0.002, 0.0, 0.0]) @ t, dict(info, fitness=0.5), "shape", True)
    assert optimise.join_pick([right, nudged], src) is right  # (within JOIN_AGREE: the best one)


def test_the_scan_takes_its_base_from_a_joined_piece():
    """The scan's own piece never found a base (level at long range: no depth on the board); a piece that joins it
    did. join_pieces moves that piece's base and region into the scan's world (Scene.adopt_base): the table at the
    same height, the region round the object, then grown over what the pieces saw."""
    if optimise is None:
        return
    from tribuutscan import cloud as cl
    from tribuutscan.scene import Scene
    rng = np.random.default_rng(4)
    shape = np.concatenate(box_faces(rng)[:3] + box_faces(rng)[5:])
    to_cam = lambda pose, pts: (pts - pose[:3, 3] * 1000) @ pose[:3, :3]
    off = track.exp_se3(np.r_[0.0, 0.0, np.radians(12), 0.020, -0.008, 0.004])  # piece 2's world -> piece 1's
    s = cl.Scanner("small", log=lambda m: None)
    s.pieces = [{"first_index": 0}, {"first_index": 20}]
    s.piece_maps = [type(s.marker_map)(add_mm=3.0)]
    second = Scene("still")  # the table at z = -25 (h = z + 25) in the true world, as piece 2 saw it
    r, t = off[:3, :3], off[:3, 3] * 1000
    second.plane = (r.T @ [0, 0, 1.0], 25.0 + np.array([0, 0, 1.0]) @ t)
    second.axis, second.radius, second.top = r.T @ (np.array([0, 0, -25.0]) - t), 60.0, 70.0
    s.piece_scenes = [s.scene]
    s.scene = second
    for i, a in [(i, np.radians(10 * i)) for i in range(10)] + [(20 + i, np.radians(40 + 10 * i)) for i in range(10)]:
        true = look_at(np.array([250 * np.cos(a), 250 * np.sin(a), 150.0]))
        pose = true if i < 20 else np.linalg.inv(off) @ true
        pts = to_cam(true, shape)
        s.keyframes.append((i, pose, pts, None, np.zeros((0, 3))))
        s.poses.append((i, pose))
        s.merged_data[i] = (pts.astype(np.float32), np.zeros((len(pts), 3), np.uint8), np.zeros((len(pts), 3), np.float32))
        s.track_log.append((i, "tracked", pose))
    report = optimise.join_pieces(s, log=lambda m: None)
    assert report["joined"][2] and report["base_from"] == 2
    n, c = s.scene.plane
    assert np.degrees(np.arccos(min(1.0, n @ [0, 0, 1.0]))) < 1.0 and abs(c - 25.0) < 1.5
    assert np.linalg.norm(s.scene.axis[:2]) < 2.0 and abs(s.scene.axis[2] + 25.0) < 1.5
    assert s.scene.radius >= 60.0 and s.scene.top >= 65.0  # (grown over the box: its corners lie 50 mm out)
