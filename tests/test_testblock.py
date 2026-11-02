import numpy as np

from tribuutscan import testblock as tb


def closed(triangles):
    """Every edge in exactly two triangles."""
    e = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), 1)
    return bool((np.unique(e, axis=0, return_counts=True)[1] == 2).all())


def volume(vertices, triangles):
    v = vertices[triangles]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6)


def test_marching_cubes_closes_a_sphere_on_its_surface():
    g, lo = 0.5, np.array([-6.0, -6.0, -6.0])
    ax = lo[0] + np.arange(25) * g
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    v, t = tb.marching_cubes((np.sqrt(x ** 2 + y ** 2 + z ** 2) - 4.3).astype(np.float32), lo, g)
    assert closed(t) and np.abs(np.linalg.norm(v, axis=1) - 4.3).max() < 0.01
    assert abs(volume(v, t) - 4 / 3 * np.pi * 4.3 ** 3) < 0.01 * 4 / 3 * np.pi * 4.3 ** 3  # facing out: positive


def test_the_block_is_closed_and_holds_its_volume():
    """A coarse mesh of the block: closed, its volume the sum of the parts' (base 44000, step A 13500,
    step B 6000, ramp 1440, dome cap 360 pi, bosses 768 pi + 72 pi, ribs 540, minus holes 200 pi + 20 pi
    + 54 pi: 68389 mm3), and the vertices near the shape (at sharp rims the distance bends along a grid
    edge, so a vertex interpolated on it is off by up to ~a quarter of the grid step)."""
    v, t = tb.build(0.5)
    want = 44000 + 13500 + 6000 + 1440 + 540 + np.pi * (360 + 768 + 72 - 200 - 20 - 54)
    assert closed(t) and abs(volume(v, t) - want) < 0.003 * want
    assert np.abs(tb.sdf(v.astype(np.float32))).max() < 0.15 and tb.face_error(v, t) < 0.5
