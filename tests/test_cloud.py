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
