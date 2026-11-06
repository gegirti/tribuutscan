import numpy as np

from tribuutscan import cloud as cl


def test_same_frame_twice_does_not_grow():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0, 50, (5000, 3))
    rgb = rng.integers(0, 256, (5000, 3), dtype=np.uint8)
    c = cl.VoxelCloud(1.0)
    c.add(xyz, rgb)
    n = len(c)
    c.add(xyz, rgb)
    assert len(c) == n
    assert (c.seen == 2).all()


def test_average_and_min_seen():
    c = cl.VoxelCloud(1.0)
    rgb = np.full((2, 3), 100, np.uint8)
    c.add(np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]), rgb)  # same cube
    c.add(np.array([[0.6, 0.6, 0.6], [5.5, 5.5, 5.5]]), rgb)  # same cube again + a new one
    xyz, _, seen = c.points(min_seen=2)
    assert len(xyz) == 1 and seen[0] == 2
    assert np.allclose(xyz[0], 0.4)  # mean of 0.2, 0.4, 0.6
    assert len(c.points(min_seen=1)[0]) == 2


def test_negative_coordinates_pack():
    cells = np.array([[-3, 7, -1000], [0, 0, 0], [12345, -54321, 1]])
    assert (cl.unpack(cl.pack(cells)) == cells).all()


def test_isolate_drops_table_keeps_centre_object():
    rng = np.random.default_rng(4)
    # camera looks down +z; table at z = 300 mm, a 40 mm box on it at the centre, a can off to the side
    table = np.c_[rng.uniform(-250, 250, (20000, 2)), np.full(20000, 300.0)]  # runs out of view
    box = np.c_[rng.uniform(-20, 20, (4000, 2)), np.full(4000, 260.0)]  # box top, 40 mm above table
    box_side = np.c_[np.full(2000, 20.0), rng.uniform(-20, 20, 2000), rng.uniform(260, 297, 2000)]
    can = np.c_[rng.uniform(90, 110, (1000, 2)), np.full(1000, 250.0)]
    xyz = np.concatenate([table, box, box_side, can])
    f, w, h = 1000.0, 1280, 800
    pix = np.c_[xyz[:, 0] * f / xyz[:, 2] + w / 2, xyz[:, 1] * f / xyz[:, 2] + h / 2]
    out, _, had_plane = cl.isolate(xyz, np.zeros((len(xyz), 3), np.uint8), pix, (h, w), rng)
    assert had_plane
    assert (np.abs(out[:, 2] - 300) > 3).all()  # table gone
    assert np.abs(out[:, :2]).max() <= 20.5  # only the box: the can is not connected
    assert len(out) > 5000


def test_object_face_is_not_taken_for_a_table():
    rng = np.random.default_rng(5)
    # a big flat face nearest the scanner, with the object's sides behind it (no table in view)
    face = np.c_[rng.uniform(-60, 60, (10000, 2)), np.full(10000, 250.0)]
    sides = np.c_[np.full(4000, 60.0), rng.uniform(-60, 60, 4000), rng.uniform(254, 300, 4000)]
    assert cl.support_plane(np.concatenate([face, sides]), rng) is None


def test_flat_lid_in_the_middle_is_not_a_table():
    rng = np.random.default_rng(6)
    # a flat lid at the centre with a handle arching above it (toward the scanner), nothing reaching the edge
    lid = np.c_[rng.uniform(-60, 60, (8000, 2)), np.full(8000, 280.0)]  # 144 cm2: passes the area rule
    t = rng.uniform(0, np.pi, 3000)
    handle = np.c_[70 * np.cos(t), rng.uniform(-5, 5, 3000), 280 - 40 * np.sin(t)]
    xyz = np.concatenate([lid, handle])
    f, w, h = 1000.0, 1280, 800
    pix = np.c_[xyz[:, 0] * f / xyz[:, 2] + w / 2, xyz[:, 1] * f / xyz[:, 2] + h / 2]
    assert cl.support_plane(xyz, rng, pix, (h, w)) is None
    assert cl.support_plane(xyz, rng) is not None  # without the edge rule it would be taken


def test_normals_point_out_of_a_sphere():
    rng = np.random.default_rng(7)
    d = rng.normal(size=(20000, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    normals = cl.VoxelCloud(0.5).normals(d * 50)
    assert np.median(np.abs(np.sum(normals * d, 1))) > 0.99


def test_scale_motion_halves_and_doubles():
    t = np.eye(4)
    a = np.radians(10)
    t[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    t[:3, 3] = [0.01, 0, 0]
    half = cl.scale_motion(t, 0.5)
    assert np.allclose(half @ half, cl.scale_motion(t, 1.0), atol=1e-3)
    assert abs(cl.turned(cl.scale_motion(t, 2.0))[1] - 20) < 1e-6


def test_refind_gate_rejects_a_jump():
    xyz = np.array([[0, 0, 250.0], [10, 0, 250], [0, 10, 260]])
    jump = np.eye(4)
    jump[0, 3] = 0.12  # 120 mm: what the first version accepted
    assert not cl.plausible_step(jump, xyz, scale=cl.RELOCK_MAX_SCALE)
    small = np.eye(4)
    small[0, 3] = 0.02
    assert cl.plausible_step(small, xyz)


def test_scene_keeps_the_table_once_found():
    from tribuutscan.scene import Scene
    rng = np.random.default_rng(8)
    # camera looks down +z at a table (z = 300 mm) with a box; then the camera moves so only a
    # small strip of table stays in view: the kept plane must still remove it
    table = np.c_[rng.uniform(-250, 250, (20000, 2)), np.full(20000, 300.0)]
    box = np.c_[rng.uniform(-20, 20, (4000, 2)), np.full(4000, 260.0)]
    side = np.c_[np.full(2000, 20.0), rng.uniform(-20, 20, 2000), rng.uniform(260, 297, 2000)]
    xyz = np.concatenate([table, box, side])
    f, w, h = 1000.0, 1280, 800
    pix = np.c_[xyz[:, 0] * f / xyz[:, 2] + w / 2, xyz[:, 1] * f / xyz[:, 2] + h / 2]
    scene = Scene("still", rng)
    for _ in range(3):
        scene.observe(xyz, pix, (h, w), np.eye(4))
    assert scene.plane is not None
    small = np.concatenate([table[np.abs(table[:, 0] - 40) < 10][:300], box, side])  # under 80 cm2 of table
    spix = np.c_[small[:, 0] * f / small[:, 2] + w / 2, small[:, 1] * f / small[:, 2] + h / 2]
    obj = scene.object_mask(small, spix, (h, w), np.eye(4))
    assert not obj[:300].any() and obj[300:].mean() > 0.9  # the table strip is gone, the box stays


def big_object_on_a_rim_of_table():
    """Camera looks down +z at a box top (z = 255 mm) that fills the view but for a thin strip of table
    (z = 300) at the left and right edges (about 5% of the points), with the box's walls reaching down."""
    f, w, h = 1000.0, 1280, 800
    v, u = np.mgrid[0:h:8, 0:w:8]
    ray = np.stack([(u.ravel() - w / 2) / f, (v.ravel() - h / 2) / f, np.ones(u.size)], 1)
    top = ray * 255.0
    on_box = (np.abs(top[:, 0]) <= 155) & (np.abs(top[:, 1]) <= 100)
    xyz = np.where(on_box[:, None], top, ray * 300.0)
    yy, zz = np.meshgrid(np.arange(-100, 101, 2.0), np.arange(257, 298, 2.0))
    walls = np.concatenate([np.c_[np.full(yy.size, sx), yy.ravel(), zz.ravel()] for sx in (-155.0, 155.0)])
    xyz = np.concatenate([xyz, walls])
    pix = np.c_[xyz[:, 0] * f / xyz[:, 2] + w / 2, xyz[:, 1] * f / xyz[:, 2] + h / 2]
    return xyz, pix, (h, w)


def test_base_is_found_under_an_object_that_fills_the_view():
    from tribuutscan.scene import Scene
    rng = np.random.default_rng(9)
    xyz, pix, shape = big_object_on_a_rim_of_table()
    table = np.abs(xyz[:, 2] - 300) < 1
    assert 0.03 < table.mean() < 0.08  # a thin rim, like the marker part on the desk (7-9%)
    assert cl.support_plane(xyz, rng, pix, shape) is None  # tracking's per-frame test: too little table
    scene = Scene("still", rng)
    for _ in range(3):
        scene.observe(xyz, pix, shape, np.eye(4))
    assert scene.plane is not None and abs(abs(scene.plane[1]) - 300) < 3  # the table, not the box top


def test_a_wall_behind_the_object_is_not_its_base():
    from tribuutscan.scene import Scene
    rng = np.random.default_rng(10)
    # a wall at z = 400 mm fills the view; a box floats in front of it (no table), 120 mm away
    wall = np.c_[rng.uniform(-300, 300, (20000, 2)), np.full(20000, 400.0)]
    box = np.c_[rng.uniform(-40, 40, (6000, 2)), np.full(6000, 250.0)]
    side = np.c_[np.full(2000, 40.0), rng.uniform(-40, 40, 2000), rng.uniform(250, 280, 2000)]
    xyz = np.concatenate([wall, box, side])
    f, w, h = 1000.0, 1280, 800
    pix = np.c_[xyz[:, 0] * f / xyz[:, 2] + w / 2, xyz[:, 1] * f / xyz[:, 2] + h / 2]
    assert cl.support_plane(xyz, rng, pix, (h, w)) is not None  # without the contact rule: taken
    scene = Scene("still", rng)
    for _ in range(4):
        scene.observe(xyz, pix, (h, w), np.eye(4))
    assert scene.plane is None


def sloped_face_on_a_table(rng):
    """Camera looks down +z at a rim of table (z = 300 mm, kept plane known; 100-140 mm from the
    axis) and an object's face rising gently from it (6 degrees, within 80 mm): near the table, the
    face outnumbers the table 30 to 1, as on the marker scan."""
    from tribuutscan.scene import Scene
    xy = rng.uniform(-140, 140, (16000, 2))
    r = np.hypot(xy[:, 0], xy[:, 1])
    table = np.c_[xy[(r > 100) & (r < 140)][:400], np.full(400, 300.0)]
    fx, fy = rng.uniform(-80, 80, (2, 20000))
    face = np.c_[fx, fy, 300 - (fy + 80) * np.tan(np.radians(6))]
    scene = Scene("still", rng)
    scene.plane = (np.array([0.0, 0.0, -1.0]), 300.0)  # n toward the scanner: height = 300 - z
    return scene, table, face


def test_the_base_is_refitted_beside_the_object_not_on_it():
    rng = np.random.default_rng(11)
    scene, table, face = sloped_face_on_a_table(rng)
    xyz = np.concatenate([table, face])
    n, c = scene.plane
    for band in (10, 3):  # the re-fit before 2026-11-01: everything near the table, twice
        near = xyz[np.abs(xyz @ n + c) < band]
        n = np.linalg.svd(near - near.mean(0), full_matrices=False)[2][2]
        n, c = n * np.sign(-n[2]), -(n * np.sign(-n[2])) @ near.mean(0)
    assert np.degrees(np.arccos(abs(n[2]))) > 5  # it ends up on the face
    scene.axis, scene.radius, scene.top = np.array([0.0, 0.0, 300.0]), 135.0, 100.0  # the face reaches 113 mm
    n, c = scene.plane_in_camera(xyz, np.eye(4))
    assert np.degrees(np.arccos(abs(n[2]))) < 0.1 and abs(abs(c) - 300) < 0.1


def test_a_tipped_refit_falls_back_to_the_kept_plane():
    rng = np.random.default_rng(12)
    scene, _, face = sloped_face_on_a_table(rng)  # first view: no region yet, and no table in sight
    n, c = scene.plane_in_camera(face, np.eye(4))
    assert np.allclose(n, scene.plane[0]) and c == scene.plane[1]


def test_the_region_grows_toward_what_the_scanner_is_aimed_at():
    """A neighbour touching the object at the edge of the view does not pull the region out; the
    same neighbour would, if growth followed everything connected (the marker scan's 317 mm region)."""
    from tribuutscan import scene as sc
    rng = np.random.default_rng(13)
    f, w, h = 1000.0, 1280, 800
    proj = lambda p: np.c_[p[:, 0] * f / p[:, 2] + w / 2, p[:, 1] * f / p[:, 2] + h / 2]
    table = np.c_[rng.uniform(-250, 250, (20000, 2)), np.full(20000, 300.0)]
    top = np.c_[rng.uniform(-60, 60, (8000, 2)), np.full(8000, 260.0)]
    sides = np.concatenate([np.c_[np.full(1500, x), rng.uniform(-60, 60, 1500), rng.uniform(260, 297, 1500)]
                            for x in (-60.0, 60.0)])
    obj = np.concatenate([top, sides])
    nb = np.c_[rng.uniform(60, 166, 4000), rng.uniform(-20, 20, 4000), np.full(4000, 280.0)]  # touching, low
    nb_side = np.c_[np.full(1000, 166.0), rng.uniform(-20, 20, 1000), rng.uniform(280, 297, 1000)]
    first = np.concatenate([table, obj])
    later = np.concatenate([table, obj, nb, nb_side])
    old = sc.GROW_WINDOW
    try:
        for window, grows in ((sc.GROW_WINDOW, False), (1.0, True)):
            sc.GROW_WINDOW = window
            scene = sc.Scene("still", np.random.default_rng(0))
            for _ in range(3):
                scene.observe(first, proj(first), (h, w), np.eye(4))
            scene.masks(first, proj(first), (h, w), np.eye(4))
            start = scene.radius
            for _ in range(10):
                mask = scene.masks(later, proj(later), (h, w), np.eye(4))[0]
            far = later[:, 0] > start + 10
            assert (scene.radius > start + 30) == grows, (window, start, scene.radius)
            assert mask[len(table):len(first)].mean() > 0.95  # the object stays whole
            if not grows:
                assert not mask[far].any()
    finally:
        sc.GROW_WINDOW = old


def test_the_region_does_not_grow_along_what_lies_on_the_table():
    """A sheet lying on the table (cardboard, a cable: 4 mm up) touching the object's foot, right in the
    middle of the view: the region stays (the long-range hand scan's grew 98 -> 552 mm along such)."""
    from tribuutscan import scene as sc
    rng = np.random.default_rng(14)
    f, w, h = 1000.0, 1280, 800
    proj = lambda p: np.c_[p[:, 0] * f / p[:, 2] + w / 2, p[:, 1] * f / p[:, 2] + h / 2]
    table = np.c_[rng.uniform(-250, 250, (20000, 2)), np.full(20000, 300.0)]
    top = np.c_[rng.uniform(-60, 60, (8000, 2)), np.full(8000, 260.0)]
    sides = np.concatenate([np.c_[np.full(1500, x), rng.uniform(-60, 60, 1500), rng.uniform(260, 297, 1500)]
                            for x in (-60.0, 60.0)])
    obj = np.concatenate([top, sides])
    sheet = np.c_[rng.uniform(60, 200, 6000), rng.uniform(-40, 40, 6000), np.full(6000, 296.0)]
    first, later = np.concatenate([table, obj]), np.concatenate([table, obj, sheet])
    old = sc.GROW_WINDOW
    try:
        sc.GROW_WINDOW = 1.0  # the whole view counts as aimed at
        scene = sc.Scene("still", np.random.default_rng(0))
        for _ in range(3):
            scene.observe(first, proj(first), (h, w), np.eye(4))
        scene.masks(first, proj(first), (h, w), np.eye(4))
        start = scene.radius
        for _ in range(10):
            scene.masks(later, proj(later), (h, w), np.eye(4))
        assert scene.radius < start + 1, (start, scene.radius)
    finally:
        sc.GROW_WINDOW = old


def _off_centre_scene():
    """Camera looking down at a table (z = 300 mm), a 60 mm box off to the side, a crumb (5 points 5 mm up)
    where the middle of the view hits; a Scene that knows the table and the box's region."""
    from tribuutscan import scene as sc
    rng = np.random.default_rng(15)
    f, w, h = 1000.0, 1280, 800
    table = np.c_[rng.uniform(-250, 250, (20000, 2)), np.full(20000, 300.0)]
    top = np.c_[rng.uniform(100, 160, 3000), rng.uniform(-30, 30, 3000), np.full(3000, 240.0)]
    side = np.c_[np.full(2000, 100.0), rng.uniform(-30, 30, 2000), rng.uniform(240, 297, 2000)]
    crumb = np.c_[rng.uniform(-1, 1, (5, 2)), np.full(5, 295.0)]
    xyz = np.concatenate([table, top, side, crumb])
    pix = np.c_[xyz[:, 0] * f / xyz[:, 2] + w / 2, xyz[:, 1] * f / xyz[:, 2] + h / 2].astype(int)
    scene = sc.Scene("still", rng)
    scene.plane = (np.array([0.0, 0.0, -1.0]), 300.0)  # up out of the table, toward the scanner
    scene.axis, scene.radius, scene.top = np.array([130.0, 0.0, 300.0]), 60.0, 80.0
    return xyz, pix, (h, w), scene, slice(len(table), len(table) + 5000)


def test_the_tracking_source_follows_the_region_when_the_centre_misses():
    xyz, pix, shape, scene, box = _off_centre_scene()
    centre, plane = cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0))
    assert plane is not None and centre.sum() <= 5  # the middle of the view: a crumb
    seeds = scene.core(xyz, np.eye(4))
    keep, _ = cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0), seeds)
    assert keep[box].mean() > 0.99 and keep.sum() <= box.stop - box.start + 2  # the box, nothing else
    assert cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0), seeds & False)[0].sum() <= 5  # no seeds


def test_the_aim_takes_the_piece_where_the_viewer_had_the_object():
    """The viewer's aim (a click, or the piece it followed) gives the depth pixel the first frame's piece is
    taken at: the box off to the side, where the middle of the view hits a crumb (the dark-hand scene,
    2026-11-05: the centre took the shelf behind the hand)."""
    xyz, pix, shape, scene, box = _off_centre_scene()
    rng = np.random.default_rng(0)
    assert cl.isolate_mask(xyz, pix, shape, rng)[0].sum() <= 5  # the centre: the crumb
    at = tuple(np.median(pix[box], axis=0))  # a pixel on the box
    keep, _ = cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0), at=at)
    assert keep[box].mean() > 0.99 and keep.sum() <= box.stop - box.start + 2
    assert (cl.centre_piece(xyz, pix, shape, at=at) == cl.centre_piece(xyz, pix, shape, at=at)).all()
    s = cl.Scanner("small", log=lambda m: None, aim=at)
    assert s.aim == at and s.scene.prefer_piece and s.stats["aim"] == list(at)
    assert not cl.Scanner("small", log=lambda m: None).scene.prefer_piece


def test_a_replay_takes_the_aim_its_live_scan_had():
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        assert cl.recording_aim(tmp) is None  # no info.json
        (Path(tmp) / "info.json").write_text(json.dumps({"note": "frames a live scan processed"}))
        assert cl.recording_aim(tmp) is None
        (Path(tmp) / "info.json").write_text(json.dumps({"aim": [412.0, 380.0], "ir": {"auto": True}}))
        assert cl.recording_aim(tmp) == (412.0, 380.0)


def test_without_a_base_the_source_follows_the_ball_when_the_centre_misses():
    """No base known (the platform scans: a white top without depth): the object's ball seeds the tracking
    source, where the middle of the view hits only a crumb."""
    xyz, pix, shape, scene, box = _off_centre_scene()
    scene.plane = scene.axis = None
    scene.ball = (np.array([130.0, 0.0, 270.0]), 90.0, 60.0)  # (centre, radius, start radius) around the box
    seeds = scene.core(xyz, np.eye(4))
    assert seeds[box].mean() > 0.6  # the box's points near the ball's centre (table ones too: the plane cut drops them)
    keep, _ = cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0), seeds)
    assert keep[box].mean() > 0.99 and keep.sum() <= box.stop - box.start + 2
    scene.ball = None
    assert scene.core(xyz, np.eye(4)) is None


def test_a_base_seen_late_is_found_and_the_region_fitted_on_tracking_s_piece():
    """A platform seen with depth only after VOTE_LIMIT frames (looking down at it), the middle of the view on
    something 50 mm above it: the base is still looked for (every VOTE_LATE_EVERY-th frame), stands_on takes
    tracking's piece, and the region is fitted on that piece, not on what the middle of the view hits."""
    from tribuutscan import scene as sc
    xyz, pix, shape, scene, box = _off_centre_scene()
    xyz = xyz.copy()
    xyz[-5:, 2] = 250.0  # the crumb, now 50 mm up: standing on nothing
    f, (h, w) = 1000.0, shape
    pix = np.c_[xyz[:, 0] * f / xyz[:, 2] + w / 2, xyz[:, 1] * f / xyz[:, 2] + h / 2].astype(int)
    tracked = np.zeros(len(xyz), bool)
    tracked[box] = True
    found = []
    for piece in (None, tracked):
        s = sc.Scene("still", np.random.default_rng(3))
        s.seen_frames = sc.VOTE_LIMIT
        for _ in range(40):
            s.observe(xyz, pix, shape, np.eye(4), piece)
        found.append(s)
    assert found[0].plane is None  # the middle of the view alone: no base
    late = found[1]
    assert late.plane is not None and late.late_plane and abs(late.plane[0] @ [0, 0, -1]) > 0.99
    obj, base = late.masks(xyz, pix, shape, np.eye(4), piece=tracked)
    assert np.linalg.norm(late.axis[:2] - [130.0, 0.0]) < 15  # round the box, not the crumb at the centre
    assert obj[box].mean() > 0.9 and base[:box.start].any() and not obj[-5:].any()


def test_the_base_cuts_the_source_when_the_frame_shows_no_table_of_its_own():
    xyz, pix, shape, scene, box = _off_centre_scene()
    near = (np.abs(xyz[:, 0] - 130) < 45) & (np.abs(xyz[:, 1]) < 45)  # a frame showing too little table to fit
    xyz, pix = xyz[near], pix[near]
    assert cl.support_plane(xyz, np.random.default_rng(0), pix, shape) is None
    seeds = scene.core(xyz, np.eye(4))
    whole = cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0), seeds)[0]
    cut = cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0), seeds,
                          lambda: scene.plane_in_camera(xyz, np.eye(4)))[0]
    on_table = np.abs(xyz[:, 2] - 300) < 3
    assert whole[on_table].mean() > 0.9 and not cut[on_table].any() and cut[~on_table & seeds].all()


def test_the_kept_table_refitted_once_serves_the_cut_and_the_floor_check():
    """Scene.table_in_camera: the kept base re-fitted to the frame (its normal and extent, as the frame's
    own RANSAC plane gives them); not taken when the pose guess puts the band beside the table. Cutting
    by it keeps the same piece as searching the frame's own plane (no support_plane per frame)."""
    xyz, pix, shape, scene, box = _off_centre_scene()
    n, c, span, fitted = scene.table_in_camera(xyz, np.eye(4))
    assert fitted and abs(abs(n[2]) - 1) < 1e-4 and abs(abs(c) - 300) < 0.1 and span > cl.FLOOR_MIN_SPAN_MM
    own = cl.support_plane(xyz, np.random.default_rng(0), pix, shape)
    rn, rspan = cl.table_normal(xyz, own)
    assert abs(abs(rn @ n) - 1) < 1e-4 and abs(rspan - span) < 0.05 * span
    seeds = scene.core(xyz, np.eye(4))
    searched = cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0), seeds)[0]
    given, _ = cl.isolate_mask(xyz, pix, shape, np.random.default_rng(0), seeds, plane=(n, c))
    assert (searched == given).all()
    off = np.eye(4)
    off[2, 3] = 25.0  # a guess 25 mm off along the view: the 10 mm band misses the table
    assert not scene.table_in_camera(xyz, off)[3]


def test_a_pose_that_tilts_the_table_is_not_taken():
    """A broad table must lie as flat as the kept base (the hand scan's slid poses tilted it 3.4-8.2
    degrees); a narrow strip of it (a rim around an object filling the view) says nothing."""
    xyz, pix, shape, scene, box = _off_centre_scene()
    s = cl.Scanner(log=lambda m: None)
    s.scene = scene
    plane = cl.support_plane(xyz, np.random.default_rng(0), pix, shape)
    s.frame_table = cl.table_normal(xyz, plane)
    tilt = lambda deg: np.r_[np.c_[np.array([[1, 0, 0], [0, np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                                             [0, np.sin(np.radians(deg)), np.cos(np.radians(deg))]]), np.zeros(3)],
                             [[0, 0, 0, 1]]]
    assert s.floor_agrees(np.eye(4)) and s.floor_agrees(tilt(1.5))
    assert not s.floor_agrees(tilt(5.0)) and not s.floor_agrees(tilt(-5.0))
    assert s.floor_agrees(tilt(40.0))  # that far off it is another surface
    strip = xyz[(np.abs(xyz[:, 2] - 300) < 3) & (np.abs(xyz[:, 1]) < 30)]
    s.frame_table = cl.table_normal(strip, plane)
    assert s.frame_table[1] < cl.FLOOR_MIN_SPAN_MM and s.floor_agrees(tilt(5.0))


def test_a_pose_found_anywhere_must_be_reachable_in_the_time_lost():
    s = cl.Scanner(log=lambda m: None)
    s.last_t = 10.0
    xyz = np.array([[0.0, 0.0, 300.0]])  # the object 30 cm in front
    turn = lambda deg: np.r_[np.c_[np.array([[np.cos(np.radians(deg)), 0, np.sin(np.radians(deg))], [0, 1, 0],
                                             [-np.sin(np.radians(deg)), 0, np.cos(np.radians(deg))]]),
                                   np.zeros(3)], [[0, 0, 0, 1]]]
    about_object = lambda deg: np.r_[np.c_[np.eye(3), [0, 0, 0.3]], [[0, 0, 0, 1]]] @ turn(deg) @ \
        np.r_[np.c_[np.eye(3), [0, 0, -0.3]], [[0, 0, 0, 1]]]  # the camera walked round the object
    assert s.reachable(about_object(40), xyz, 11.0)  # 40 degrees in 1 s
    assert not s.reachable(about_object(136), xyz, 11.35)  # the plate's mirrored re-find
    assert s.reachable(about_object(136), xyz, 13.0)  # given the time
    assert s.reachable(about_object(170), xyz, None)  # no time stamps: no limit


def test_output_leaves_out_base_background_and_floating_bits():
    s = cl.Scanner("small", log=lambda m: None)
    s.scene.plane, s.scene.axis = (np.array([0.0, 0.0, 1.0]), 0.0), np.zeros(3)  # world: base at z = 0
    s.scene.radius, s.scene.top = 60.0, 100.0
    g = np.arange(-60, 61, 1.5)
    gx, gy = np.meshgrid(g, g)
    disc = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    base = disc[np.hypot(disc[:, 0], disc[:, 1]) <= 55]  # the base under and around the object
    t, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 120), np.arange(4.0, 50.0, 1.5))
    obj = np.c_[25 * np.cos(t.ravel()), 25 * np.sin(t.ravel()), zz.ravel()]  # a can on the base
    fx, fy = np.meshgrid(np.arange(-3, 4, 1.5), np.arange(-3, 4, 1.5))
    floater = np.c_[fx.ravel() + 45, fy.ravel(), np.full(fx.size, 80.0)]  # inside the region, not connected
    table = np.c_[np.linspace(100, 200, 50), np.zeros(50), np.zeros(50)]  # base outside the region
    other = np.c_[np.full(30, 150.0), np.zeros(30), np.linspace(10, 40, 30)]  # another object, outside
    parts = {"base": base, "obj": obj, "floater": floater, "table": table, "other": other}
    s.cloud.add(np.concatenate(list(parts.values())), np.zeros((sum(map(len, parts.values())), 3), np.uint8))
    centre = s.cloud.xyz / s.cloud.n[:, None]
    kind = lambda p: ("obj" if p[2] > 3 and np.hypot(p[0], p[1]) < 30 and p[2] < 60 else "floater" if p[2] > 60 else
                      "base" if np.hypot(p[0], p[1]) <= 56 else "far")
    kinds = np.array([kind(p) for p in centre])
    for remove, wanted in ((True, {"obj"}), (False, {"obj", "base"})):
        s.remove_base = remove
        keep = s.output_mask(centre)
        assert set(kinds[keep]) == wanted, (remove, set(kinds[keep]))
        assert keep[kinds == "obj"].all()


def test_the_crop_box_is_level_with_the_base_and_keeps_what_is_inside():
    s = cl.Scanner("small", log=lambda m: None)
    assert np.allclose(s.crop_axes(), np.eye(3))  # no base, no platform: the first camera's axes
    a = np.radians(35.0)
    up = np.array([0.0, -np.cos(a), -np.sin(a)])  # world: the base's normal, the scanner looking down 35 degrees
    s.scene.plane, s.scene.axis = (up, 0.0), np.zeros(3)
    s.scene.radius, s.scene.top = 60.0, 100.0
    axes = s.crop_axes()
    assert np.allclose(axes @ axes.T, np.eye(3)) and np.isclose(np.linalg.det(axes), 1.0)
    assert np.allclose(axes[1], cl.FLIP @ up) and abs(axes[0, 1]) < 1e-12  # y up from the base, x level
    side = np.cross(up, [1.0, 0.0, 0.0])
    t, h = np.meshgrid(np.linspace(0, 2 * np.pi, 120), np.arange(4.0, 50.0, 1.5))
    ring = np.outer(np.cos(t.ravel()), [1.0, 0.0, 0.0]) + np.outer(np.sin(t.ravel()), side)
    can = 25 * ring + np.outer(h.ravel(), up)  # a can standing on the base
    s.cloud.add(can, np.zeros((len(can), 3), np.uint8))
    centre = s.cloud.xyz / s.cloud.n[:, None]
    height, x = centre @ up, (centre @ cl.FLIP.T) @ axes[0]
    assert s.output_mask(centre).all()
    s.crop = {"axes": axes.tolist(), "lo": [-100.0, 20.0, -100.0], "hi": [0.0, 100.0, 100.0]}
    keep = s.output_mask(centre)
    assert keep.any() and np.all(height[keep] >= 20 - 1e-9) and np.all(x[keep] <= 1e-9)
    assert keep[(height > 21) & (x < -1)].all()  # the half above 20 mm, cut level with the base


def test_a_saved_scan_keeps_what_fusing_again_needs():
    """Every tracked frame's pose, its labels, the source recording and the camera survive save and
    restore, and Optimise's corrections move the frames between merged ones too."""
    try:
        from tribuutscan import synth
    except ImportError:  # needs Open3D (.venv)
        return
    import json
    import tempfile
    from pathlib import Path
    from tribuutscan import calib
    before = calib.CAMERA.spec
    calib.use(calib.ESTIMATES)  # synthetic frames are rendered with it
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec, out = Path(tmp) / "t-rec", Path(tmp) / "t-scan"
            synth.generate(rec, n_frames=24)
            sc = cl.scan(rec, log=lambda *a: None)
            tracked = sc.frame_poses()
            assert len(tracked) == sum(p is not None for _, _, p in sc.track_log) >= len(sc.poses) > 0
            assert set(i for i, _ in tracked) <= set(sc.labels)
            sc.save(out)
            session = json.loads((out / "session.json").read_text())
            assert session["sources"] == [{"rec": str(rec.resolve()), "first_index": 0}]
            assert session["camera"] == {"source": "estimate", "fingerprint": calib.CAMERA.fingerprint()}
            back = cl.Scanner.restore(out, log=lambda *a: None)
            bases = {id(points.base) for points, _, _ in back.merged_data.values()}
            assert len(bases) == 1  # every merged frame is a view of ONE loaded array, not its own full copy
            assert [i for i, _ in back.frame_poses()] == [i for i, _ in tracked]
            assert all(np.allclose(p, q, atol=1e-5) for (_, p), (_, q) in zip(back.frame_poses(), tracked))
            assert back.sources == sc.sources
            assert all(np.array_equal(back.labels[i], sc.labels[i]) for i, _ in tracked)
            shift = np.eye(4)
            shift[0, 3] = 0.005  # every merged frame 5 mm along x
            back.remerge({i: shift @ p for i, p in back.poses})
            moved = [(q @ np.linalg.inv(p))[:3, 3] for (_, p), (_, q) in zip(tracked, back.frame_poses())]
            assert np.allclose(moved, [0.005, 0, 0], atol=1e-5)
            again = cl.Scanner.restore(out, log=lambda *a: None)  # corrections growing along the scan:
            keys = [i for i, _ in again.poses]  # frames between merged ones get them blended, not held
            ramp = {}
            for i, pose in again.poses:
                t = np.eye(4)
                t[0, 3] = 0.0001 * i
                ramp[i] = t @ pose
            again.remerge(ramp)
            between = [(i, p, q) for (i, p), (_, q) in zip(tracked, again.frame_poses()) if keys[0] < i < keys[-1]]
            assert between and all(np.allclose((q @ np.linalg.inv(p))[:3, 3], [0.0001 * i, 0, 0], atol=1e-6)
                                   for i, p, q in between)
    finally:
        calib.use(before)


def test_blend_goes_along_the_shortest_arc():
    a = np.eye(4)
    b = np.eye(4)
    b[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 degrees about z
    b[:3, 3] = [2.0, 0, 0]
    half = cl.blend(a, b, 0.5)
    c = np.cos(np.pi / 4)
    assert np.allclose(half[:3, :3], [[c, -c, 0], [c, c, 0], [0, 0, 1]]) and np.allclose(half[:3, 3], [1, 0, 0])
    assert np.allclose(cl.blend(a, b, 0), a) and np.allclose(cl.blend(a, b, 1), b)


def test_synthetic_sensor_sets_and_their_true_surface():
    """The 14-bit plate set with a fixed per-pixel error: no IR, Y14 depth, and frame 0's points lie on
    true_surface.ply (the first camera's frame is the scan's world)."""
    try:
        from tribuutscan import synth
    except ImportError:  # needs Open3D (.venv)
        return
    import json
    import tempfile
    from pathlib import Path
    from tribuutscan import calib
    from tribuutscan import frames as fr
    before = calib.CAMERA.spec
    calib.use(calib.ESTIMATES)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = synth.generate(Path(tmp) / "p-rec", n_frames=2, kind="plate", pattern_mm=0.144, noise_mm=0.05, bits=14)
            info = json.loads((rec / "info.json").read_text())
            assert info["depth_bytesperline"] == fr.W * 14 // 8 and info["streams"] == ["color", "depth"]
            assert not (rec / "ir.raw").exists() and (rec / "true_surface.ply").exists()
            frame = next(fr.read_recording(rec))
            assert frame["ir"] is None and frame["depth"].dtype == np.float32
            xyz, _, _ = cl.frame_points(frame)
            d = synth.truth_distance(rec, xyz)
            on = d < 5
            assert on.mean() > 0.05 and np.median(d[on]) < 0.3  # the plate and its markers, within the sensor error
            pattern = synth.fixed_pattern(0.144, 0)
            assert abs(pattern.std() * 200.0 ** 2 / calib.CAMERA.a - 0.144) < 1e-9
            assert np.array_equal(pattern, synth.fixed_pattern(0.144, 0))  # fixed: the same every time
    finally:
        calib.use(before)


def test_frames_as_the_live_worker_gets_them_are_tracked_and_merged():
    """Colour left as YUYV bytes (as livescan hands frames over): feed and the merge thread take them,
    on the CPU tracker and on the GPU one (when there is a GPU; else it falls back to the CPU)."""
    try:
        from tribuutscan import synth
    except ImportError:  # needs Open3D (.venv)
        return
    import tempfile
    from pathlib import Path
    from tribuutscan import calib
    from tribuutscan import frames as fr
    before = calib.CAMERA.spec
    calib.use(calib.ESTIMATES)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp) / "t-rec"
            synth.generate(rec, n_frames=12)
            for tracker in ("joint", "gpu"):  # (every keyframe merged: 12 frames are 0.6 s)
                sc = cl.Scanner(log=lambda *a: None, track_with=tracker, live_view=True, merge_min_s=0.0)
                views = []
                for i, frame in enumerate(fr.read_recording(rec, yuyv=True)):
                    assert "yuyv" in frame and "color" not in frame
                    sc.feed(frame, i)
                    views.append(sc.view_codes)
                sc.flush()
                sc.close()
                assert sc.stats["lost"] == 0 and sc.stats["merged"] >= 2 and len(sc.cloud) > 1000, (tracker, sc.stats)
                assert sc.stats["markers"]  # the merge decoded the YUYV colour to look for markers
                if sc.track_with != "gpu":
                    assert all(v is None for v in views)  # the live model view comes from the GPU tracker
                    continue
                assert views[0] is None and all(v is not None and v.shape == (400, 640) for v in views[1:])
                done = [int((v >= 128).sum()) for v in views[1:]]
                thin = [int(((v >= 2) & (v < 128)).sum()) for v in views[1:]]
                # the box: seen once at first (no part seen often enough), most of it done after a few frames
                assert done[0] == 0 and thin[0] > 20000 and done[-1] > 4 * thin[-1] and done[-1] > 30000
    finally:
        calib.use(before)


def test_room_verdict_tells_a_still_room_from_a_moving_one():
    """check_room's verdict on a room (a wall, a floor and a box, camera mm), found again in the next keyframe
    as it was (a platform turned, the scanner stood) or where the tracked motion takes it (the scanner went
    round a still object); a wall sliding along itself tells nothing."""
    try:
        import open3d  # noqa: F401
    except ImportError:  # needs Open3D (.venv)
        return
    from tribuutscan import track
    rng = np.random.default_rng(3)
    u = lambda lo, hi, n=3000: rng.uniform(lo, hi, n)
    room = np.r_[np.c_[u(-300, 300), u(-250, 100), np.full(3000, 600.0)],  # wall
                 np.c_[u(-300, 300), np.full(3000, 100.0), u(350, 600)],  # floor
                 np.c_[np.full(1500, -150.0), u(-60, 100, 1500), u(450, 530, 1500)]]  # a box's side
    centre = np.array([0.0, 0.0, 330.0])  # the object, which the view turned round by 8 degrees
    move = track.exp_se3(np.r_[0.0, np.radians(8.0), 0.0, 0.0, 0.0, 0.0])[:3, :3]
    moved = (room - centre) @ move.T + centre
    noise = lambda pts: pts + rng.normal(0, 0.3, pts.shape)
    assert cl.room_verdict(room, moved, noise(room)) == "still"
    assert cl.room_verdict(room, moved, noise(moved)) == "moved"
    wall = room[:3000]  # the view slid 30 mm along the wall: it looks the same either way (new samples of it)
    resampled = np.c_[u(-300, 300, 30000), u(-250, 100, 30000), np.full(30000, 600.0)]
    assert cl.room_verdict(wall, wall + [30.0, 0, 0], noise(resampled)) is None


def turntable_set(tmp, n_frames):
    from pathlib import Path
    from tribuutscan import synth
    return synth.generate(Path(tmp) / "turn-rec", n_frames=n_frames, every=2, kind="turntable", pattern_mm=0.144,
                          noise_mm=0.05, bits=14)


def test_a_turning_platform_is_noticed_and_a_jump_is_found_again_by_shape():
    """The test block on a turning platform in a still room (synth "turntable"): the room's standing is
    noticed (object_turns: its colour features are left out from then on), and the frame 3 s later (the
    block turned ~38 degrees) is found again by shape, close to the truth. A hand-held scan of the box is
    not taken for a turning platform."""
    try:
        from tribuutscan import synth
        from tribuutscan import gputrack
    except ImportError:  # needs Open3D and wgpu (.venv)
        return
    if not gputrack.kind():
        return
    import tempfile
    from pathlib import Path
    from tribuutscan import calib
    from tribuutscan import frames as fr
    before = calib.CAMERA.spec
    calib.use(calib.ESTIMATES)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = turntable_set(tmp, 160)  # 80 frames at 10 fps, ~107 degrees of turn
            truth = synth.read_truth(rec)
            sc = cl.Scanner(log=lambda *a: None, track_with="gpu")
            turned_at, later = None, None
            for i, frame in enumerate(fr.read_recording(rec)):
                if i < 40:
                    sc.feed(frame, i)
                    if sc.object_turns and turned_at is None:
                        turned_at = i
                elif i == 70:
                    later = frame
            assert turned_at is not None and sc.stats["lost"] == 0, (turned_at, sc.stats)
            # (fed in turn, that frame is "tracked" 37 degrees off: its solve starts at the last pose, and a turn
            # about the vertical axis hides in the block's horizontal faces)
            all_xyz, _, all_pix = cl.frame_points(later, sc.seg_stride, colour=False)
            keep, _ = cl.isolate_mask(all_xyz, all_pix, cl.frame_shape(later), sc.rng)
            xyz, pix = all_xyz[keep], all_pix[keep]
            sc.gpu_source(later, xyz, pix)
            assert sc.refind_shape(xyz, np.empty((0, 3)), later["t"]) and sc.stats["shape_refinds"] == 1
            err = np.linalg.inv(truth[70]) @ sc.pose
            assert np.linalg.norm(err[:3, 3]) * 1000 < 3.0 and np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))) < 1.0
            sc.close()
            box = synth.generate(Path(tmp) / "box-rec", n_frames=40)
            sc = cl.Scanner(log=lambda *a: None, track_with="gpu")
            for i, frame in enumerate(fr.read_recording(box)):
                sc.feed(frame, i)
            sc.close()
            assert not sc.object_turns and sc.room_votes["still"] == 0, sc.room_votes
    finally:
        calib.use(before)


def test_shape_match_finds_a_shape_turned_and_moved():
    """session.shape_pose (shape features + RANSAC, now shape_features / shape_match): the test block's
    surface found again turned by 40 degrees and moved by 60 mm, to within a few mm."""
    try:
        from tribuutscan import session
    except ImportError:  # needs Open3D (.venv)
        return
    from tribuutscan import testblock, track
    v, _ = testblock.build(1.0)
    move = track.exp_se3(np.r_[np.radians([5.0, 40.0, -10.0]), [0.06, -0.02, 0.03]])  # metres
    moved = v @ move[:3, :3].T + move[:3, 3] * 1000
    found = session.shape_pose(v, moved)
    assert found is not None
    err = np.linalg.inv(move) @ found
    # coarse (RANSAC on 3 mm features: 1.7-2.5 mm, 0.9-3.7 degrees over seeds); the tracking solve refines it
    assert np.linalg.norm(err[:3, 3]) * 1000 < 5.0 and np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))) < 6.0


def test_markers_in_view_correct_a_solve_that_used_none():
    """marker_check: a solve accepted 20 mm off (as synthhard's slide was, its 8-9 markers matching nothing at
    the sliding pose) is placed by the markers alone and solved again with them, close to the truth."""
    try:
        from tribuutscan import synth
        from tribuutscan import gputrack
    except ImportError:  # needs Open3D and wgpu (.venv)
        return
    if not gputrack.kind():
        return
    import tempfile
    from pathlib import Path
    from tribuutscan import calib
    from tribuutscan import frames as fr
    before = calib.CAMERA.spec
    calib.use(calib.ESTIMATES)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = synth.generate(Path(tmp) / "m-rec", n_frames=44, every=2)
            truth = synth.read_truth(rec)
            sc = cl.Scanner(log=lambda *a: None, track_with="gpu")
            for i, frame in enumerate(fr.read_recording(rec)):
                if i < 20:
                    sc.feed(frame, i)
                    continue
                all_xyz, _, all_pix = cl.frame_points(frame, sc.seg_stride, colour=False)
                keep, _ = cl.isolate_mask(all_xyz, all_pix, cl.frame_shape(frame), sc.rng)
                xyz, pix = all_xyz[keep], all_pix[keep]
                sc.gpu_source(frame, xyz, pix)
                marker_pts = sc.frame_markers(frame)
                assert len(marker_pts) >= 3
                true_step = np.linalg.inv(sc.pose) @ truth[i]
                wrong = true_step.copy()
                wrong[:3, 3] += [0.02, 0.0, 0.0]
                step, info, ok = sc.marker_check(xyz, marker_pts, wrong)
                err = np.linalg.inv(true_step) @ step
                assert ok and info["markers_used"] >= 3 and np.linalg.norm(err[:3, 3]) * 1000 < 2.0, (ok, info)
                assert sc.marker_check(xyz, marker_pts, true_step) is None  # the markers agree: nothing to do
                break
            sc.close()
    finally:
        calib.use(before)


def test_a_platform_scan_knows_the_axis_it_turned_about():
    """turn_axis: poses turned up to 120 degrees about a vertical line 300 mm in front of the first camera give
    that line's direction, pointing up (the cameras' up: -y); a scan that did not turn on a platform gives none."""
    sc = cl.Scanner(log=lambda *a: None)
    axis, point = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 0.3])
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    for i, deg in enumerate(np.linspace(0, 120, 61)):
        m, a = np.eye(4), np.radians(deg)
        m[:3, :3] = np.eye(3) + np.sin(a) * cross + (1 - np.cos(a)) * cross @ cross  # (Rodrigues: needs no Open3D)
        m[:3, 3] = point - m[:3, :3] @ point
        sc.track_log.append((i, "tracked", m))
    assert sc.turn_axis() is None  # (not noticed as a platform)
    sc.object_turns = True
    found = sc.turn_axis()
    assert found is not None and found @ axis > 0.9999


def _surface_points(boxes, step=2.0):
    """Points every `step` mm on the top and the four sides of boxes (x0, x1, y0, y1, z0, z1), z up from the
    base (z = 0), as a scan's keyframes would hold them, several views deep (each point thrice)."""
    out = []
    for x0, x1, y0, y1, z0, z1 in boxes:
        xs, ys, zs = np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step), np.arange(z0, z1 + 1e-9, step)
        gx, gy = np.meshgrid(xs, ys)
        out.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, z1)])
        for x in (x0, x1):
            gy, gz = np.meshgrid(ys, zs)
            out.append(np.c_[np.full(gy.size, x), gy.ravel(), gz.ravel()])
        for y in (y0, y1):
            gx, gz = np.meshgrid(xs, zs)
            out.append(np.c_[gx.ravel(), np.full(gx.size, y), gz.ravel()])
    return np.tile(np.concatenate(out), (3, 1))


def test_the_base_is_lifted_onto_a_board_the_object_stands_on_and_nowhere_else():
    """lift_base (after Stop): a 300 x 200 mm board 24 mm high under a 200 mm tall object of 60 x 60 mm (the dark
    hand on the calibration board, 2026-11-05) moves the base onto the board; a stepped block, a flat-topped part
    with small bosses and a lone box keep their base."""
    from tribuutscan import scene as sc

    def lifted(boxes):
        s = sc.Scene("still")
        s.plane = (np.array([0.0, 0.0, 1.0]), 0.0)  # z up, the base at z = 0
        s.axis, s.radius, s.top = np.zeros(3), 300.0, 260.0
        return s, s.lift_base(_surface_points(boxes))

    s, h = lifted([(-150, 150, -100, 100, 0, 24), (-30, 30, -30, 30, 24, 224)])  # board, and a tall object on it
    assert h is not None and abs(h - 24) < 1.5
    assert abs(s.plane[1] + h) < 1e-9 and abs(s.axis[2] - h) < 1e-9 and abs(s.top - (260 - h)) < 1e-9
    assert s.lifted_mm == h and sc.Scene("still").lifted_mm == 0.0
    assert lifted([(-100, 100, -75, 75, 0, 10), (-75, 75, -50, 50, 10, 25)])[1] is None  # a step: much stands on it
    assert lifted([(-137, 137, -96, 96, 0, 44), (-20, -10, -5, 5, 44, 49), (30, 40, 10, 20, 44, 49)])[1] is None  # thick
    assert lifted([(-50, 50, -50, 50, 0, 40)])[1] is None  # a box: nothing on its lid


def test_one_odd_marker_does_not_fail_a_pose_the_shape_pins():
    """A solve failed only by its marker residual, with one or two markers, on a shape that pins the pose
    (degeneracy 0.037: the still scans of 2026-11-05, a spurious marker on the board's pattern) is solved again
    without them; a flat face that may slide (degeneracy under track.DEGENERATE) is still refused by its markers,
    and so is a pose that three markers disagree with."""
    s = cl.Scanner("small", log=lambda m: None)
    xyz = np.c_[np.zeros((100, 2)), np.full(100, 330.0)]
    info = {"fitness": 0.99, "rmse_mm": 0.11, "degeneracy": 0.037, "markers_used": 1, "marker_rmse_mm": 4.5}
    assert not s.solve_ok(info, xyz) and s.odd_markers(info, xyz)
    assert not s.odd_markers(dict(info, degeneracy=0.004), xyz)  # a flat face: the markers' word stands
    assert not s.odd_markers(dict(info, markers_used=3), xyz)
    assert not s.odd_markers(dict(info, marker_rmse_mm=1.5), xyz)  # (the markers agree: nothing odd)
    assert not s.odd_markers(dict(info, fitness=0.2), xyz)  # the shape does not hold either


def lump(rng, n, half_mm, z_mm):
    """n points filling a cube of half side half_mm around (0, 0, z_mm): a piece of a frame, camera mm."""
    return rng.uniform(-half_mm, half_mm, (n, 3)) + [0.0, 0.0, z_mm]


def test_a_lost_frame_is_judged_by_how_much_it_looks_like_the_object():
    """While lost, a frame's piece is "object" (about the size, points and distance the object had while tracked),
    "other" (clearly something else: crumbs, a wall, a box far behind) or "unsure" in between; with no look yet,
    "object" as before. The proportions are the 2026-11-06 dev scan's (the hand at ~36 cm, the shelf, the wall)."""
    rng = np.random.default_rng(0)
    s = cl.Scanner("small", log=lambda m: None)
    assert s.likeness(cl.look_of(lump(rng, 50, 10, 460))) == "object"  # (no look yet)
    for _ in range(10):
        s.look.append(cl.look_of(lump(rng, 5000, 70, 360)))
    assert s.likeness(cl.look_of(lump(rng, 4500, 65, 380))) == "object"
    assert s.likeness(cl.look_of(lump(rng, 60, 12, 460))) == "other"  # crumbs on the shelf
    assert s.likeness(cl.look_of(np.c_[rng.uniform(-250, 250, (20000, 2)), np.full(20000, 650.0)])) == "other"  # wall
    assert s.likeness(cl.look_of(lump(rng, 800, 70, 680))) == "other"  # a box as big, 32 cm further away
    assert s.likeness(cl.look_of(lump(rng, 1500, 70, 360))) == "unsure"  # the object half out of view


def test_only_frames_like_the_object_start_a_new_piece():
    """Lost frames of something else (or nothing) never start a piece, however long they go on, and each one starts
    the clock again; PIECE_AFTER_S of frames like the object do, and the new piece is taken where they saw it."""
    rng = np.random.default_rng(1)
    s = cl.Scanner("small", log=lambda m: None)
    s.stats["merged"] = 5
    hand, shelf = lump(rng, 3000, 60, 360), lump(rng, 3000, 60, 500) + [200.0, 0, 0]
    s.last_source = shelf.mean(0)  # (where a junk piece once was: not where the new piece is taken)
    t = 0.0
    for like in ["other"] * 30 + ["unsure"] * 5 + ["object"] * 5 + ["other"] + ["object"] * 8:  # 0.1 s apart
        s.lost_long(t, int(t * 10), like, hand)
        t += 0.1
    assert len(s.pieces) == 1 and s.lost_since is not None
    for _ in range(4):
        s.lost_long(t, int(t * 10), "object", hand)
        t += 0.1
    assert len(s.pieces) == 2 and s.pieces[1]["first_index"] == int((t - 0.1) * 10) + 1
    xyz = np.concatenate([shelf, hand])
    pix = np.c_[np.arange(len(xyz)) % 1280, np.arange(len(xyz)) // 1280].astype(float)
    _, at = s.piece_start_seeds({}, xyz, pix)
    i = int(np.argmin(np.hypot(pix[:, 0] - at[0], pix[:, 1] - at[1])))
    assert i >= len(shelf)  # on the hand


def test_a_click_is_taken_at_the_point_nearest_it():
    """A click (Scanner.pick, or a replay's planned one at its frame) becomes the depth pixel of the frame's point
    nearest it: on the depth image directly, on the colour image where the colour camera sees each point."""
    from tribuutscan.calib import CAMERA
    rng = np.random.default_rng(2)
    pix = np.c_[rng.integers(0, 1280, 3000), rng.integers(0, 800, 3000)].astype(float)
    xyz = np.c_[rng.uniform(-100, 100, (3000, 2)), rng.uniform(300, 400, 3000)]
    s = cl.Scanner("small", log=lambda m: None, picks={7: ("depth", (500.0, 300.0))})
    assert s.take_pick(6, xyz, pix) is None
    u, v = s.take_pick(7, xyz, pix)
    assert np.hypot(u - 500, v - 300) == np.hypot(pix[:, 0] - 500, pix[:, 1] - 300).min()
    cx, cy = CAMERA.colour_px(pix[:, 0], pix[:, 1], xyz[:, 2])
    s.pick("colour", (cx[42], cy[42]))
    assert s.take_pick(8, xyz, pix) == (pix[42, 0], pix[42, 1]) and s.take_pick(9, xyz, pix) is None
    assert s.stats["picks"] == 2


def test_a_new_view_votes_for_the_base_on_every_frame_again():
    """While no base is known, votes come on every frame for VOTE_LIMIT frames, then on every 10th; a view turned
    VOTE_NEW_VIEW_DEG from every view a window began at votes on every frame again for VOTE_WINDOW frames (a scan
    level for minutes before a pass from 45 degrees, which is when the base first shows depth). Turning back to
    a view seen before opens none."""
    from tribuutscan import scene as sc
    calls = []
    real = sc.cl.support_plane
    sc.cl.support_plane = lambda *a, **k: calls.append(1)  # (finds nothing: no base)
    try:
        s = sc.Scene("still")
        xyz, pix = np.zeros((10, 3)), np.zeros((10, 2))
        level, down = np.eye(4), np.eye(4)
        a = np.radians(sc.VOTE_NEW_VIEW_DEG + 5)
        down[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]

        def frames(n, pose):
            before = len(calls)
            for _ in range(n):
                s.observe(xyz, pix, (800, 1280), pose)
            return len(calls) - before
        assert frames(sc.VOTE_LIMIT, level) == sc.VOTE_LIMIT
        assert frames(200, level) == 200 // sc.VOTE_LATE_EVERY
        assert frames(sc.VOTE_WINDOW, down) == sc.VOTE_WINDOW  # a new direction: every frame again
        assert frames(200, down) == 200 // sc.VOTE_LATE_EVERY
        assert frames(200, level) == 200 // sc.VOTE_LATE_EVERY  # (seen before)
    finally:
        sc.cl.support_plane = real
