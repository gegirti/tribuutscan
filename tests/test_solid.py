import numpy as np

try:
    from tribuutscan import solid
    import open3d  # noqa: F401  (Poisson)
except ImportError:  # needs Open3D (.venv)
    solid = None


def open_cylinder():
    """A cylinder (radius 20 mm, 40 high, standing on z = 0), meshed from its exact shape, with its bottom
    and a 10 mm patch of its side taken out, as a scan leaves them; (vertices, triangles, sdf)."""
    from tribuutscan import testblock
    sdf = lambda p: testblock.cylinder(p, (0.0, 0.0), 20.0, 0.0, 40.0)
    step = 0.5
    lo = np.array([-25.0, -25.0, -5.0]) + 0.37 * step
    n = (np.array([50.0, 50.0, 50.0]) / step).astype(int) + 1
    grid = np.stack(np.meshgrid(*[lo[k] + np.arange(n[k]) * step for k in range(3)], indexing="ij"), -1)
    v, t = testblock.marching_cubes(sdf(grid.astype(np.float32)), lo, step)
    centre = v[t].mean(1)
    gone = (centre[:, 2] < 0.5) | (np.linalg.norm(centre - [20.0, 0.0, 20.0], axis=1) < 5.0)
    return v, t[~gone], sdf


def test_an_open_cylinder_is_closed_flat_on_its_base_and_true_to_its_shape():
    if solid is None:
        return
    v, t, sdf = open_cylinder()
    assert not solid.edges_closed(t)
    rgb = np.full((len(v), 3), 200, np.uint8)
    V, T, C, report = solid.close(v, t, rgb, plane=(np.array([0.0, 0.0, 1.0]), 0.0))
    assert report["edges_closed"] and solid.edges_closed(T) and C.shape == (len(V), 3)
    assert V[:, 2].min() > -1e-6  # nothing below the base
    tri = V[T]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    bottom = np.abs(tri[:, :, 2]).max(1) < 1e-3
    assert abs(area[bottom].sum() / (np.pi * 20.0 ** 2) - 1) < 0.05  # a flat disc on the plane, the cylinder's bottom
    d = np.abs(sdf(V.astype(np.float32)))
    assert np.median(d) < 0.05 and np.percentile(d, 99.5) < 0.5, (np.median(d), np.percentile(d, 99.5))  # the hole too
    assert 0.1 < report["added_share"] < 0.4  # the bottom (~19% of the area) and the side patch


def test_edges_closed_and_ear_clipping():
    if solid is None:
        return
    tetra = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    assert solid.edges_closed(tetra) and not solid.edges_closed(tetra[:3])
    assert not solid.edges_closed(np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]]))  # one face turned round
    xy = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]], float)  # an L, not convex
    tris = solid.ear_clip(xy, np.arange(6))
    p = xy[tris]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    assert len(tris) == 4 and np.all(area > 0) and abs(area.sum() - 7.0) < 1e-9  # its 7 mm2, in its winding


def test_two_planes_cut_one_after_the_other_each_capped_flat():
    if solid is None:
        return
    v, t, _ = open_cylinder()
    V, T, _, report = solid.close(v, t, None, plane=[(np.array([0.0, 0.0, 1.0]), 0.0), (np.array([-1.0, 0.0, 0.0]), 10.0)])
    assert report["edges_closed"] and report["cut"] == 2 and solid.edges_closed(T)
    assert V[:, 2].min() > -1e-6 and V[:, 0].max() < 10 + 1e-6  # nothing below the base, nothing beyond x = 10
    tri = V[T]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    side = np.abs(tri[:, :, 0] - 10).max(1) < 1e-3
    assert abs(area[side].sum() / (2 * np.sqrt(20.0 ** 2 - 10.0 ** 2) * 40) - 1) < 0.05  # the cylinder's flat chord face
