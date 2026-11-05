import numpy as np

from tribuutscan import colour


def _scene():
    """A colour image green on the left, light grey on the right; depth points: the object (green, two
    separate parts 60 mm apart, like split fingers) at ~300 mm, the room (grey) at ~600 mm, and a green patch
    far behind the object (the same colour elsewhere in the room)."""
    h, w = 800, 1280
    img = np.full((h, w, 3), 200, np.uint8)
    img[:, :640] = (40, 170, 60)
    rng = np.random.default_rng(2)

    def points(n, u0, u1, z):
        pix = np.c_[rng.integers(u0, u1, n), rng.integers(300, 500, n)]
        xyz = np.c_[(pix[:, 0] - w / 2) * z / 900, (pix[:, 1] - h / 2) * z / 900, np.full(n, z)]
        return xyz, pix

    parts = [points(800, 150, 250, 300.0), points(500, 310, 360, 300.0),  # the object in two pieces
             points(3000, 800, 1200, 600.0), points(300, 450, 520, 1200.0)]  # the room; a green patch far off
    xyz = np.concatenate([p[0] for p in parts])
    pix = np.concatenate([p[1] for p in parts])
    obj = np.zeros(len(xyz), bool)
    obj[:800] = True  # tracking had the first piece only
    return {"color": img, "shape": (h, w)}, xyz, pix, obj


def test_the_object_s_colours_find_all_of_it_and_nothing_far_off():
    frame, xyz, pix, obj = _scene()
    model = colour.ObjectColours()
    assert model.seeds(frame, xyz, pix, 150.0) is None  # nothing learned: silent
    for _ in range(3):
        model.learn(frame, xyz, pix, obj, ~obj)
    assert model.informative()  # (the split piece and the far patch count as room here: some overlap)
    seeds = model.seeds(frame, xyz, pix, 150.0)
    assert seeds[:800].mean() > 0.95 and seeds[800:1300].mean() > 0.95  # both parts, the split one joined
    assert not seeds[1300:].any()  # neither the room nor the green patch 900 mm behind


def test_the_model_stays_silent_when_object_and_room_share_their_colours():
    frame, xyz, pix, obj = _scene()
    frame["color"][:] = 120  # a grey part on a grey table
    model = colour.ObjectColours()
    for _ in range(3):
        model.learn(frame, xyz, pix, obj, ~obj)
    assert not model.informative() and model.score(frame, xyz, pix) is None
    assert model.seeds(frame, xyz, pix, 150.0) is None


def test_near_black_and_blown_out_pixels_say_nothing():
    rgb = np.array([[5, 5, 5], [250, 252, 251], [40, 170, 60]], np.uint8)
    _, usable = colour.bins(rgb)
    assert list(usable) == [False, False, True]
