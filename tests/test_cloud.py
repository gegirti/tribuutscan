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
    table = np.c_[rng_ring := np.linspace(100, 200, 50), np.zeros(50), np.zeros(50)]  # base outside the region
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
