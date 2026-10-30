import numpy as np

from tribuutscan import markermap as mm


def rigid(rng, max_mm=40.0, max_deg=25.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.eye(4)
    t[:3, :3] = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
    t[:3, 3] = rng.uniform(-max_mm, max_mm, 3) / 1000
    return t


def spaced(rng, n, low, high, scale=(1, 1, 1), offset=(0, 0, 250), min_mm=15.0):
    """n random marker positions at least min_mm apart (real stickers cannot overlap)."""
    pts = []
    while len(pts) < n:
        p = rng.uniform(low, high, 3) * scale + offset
        if all(np.linalg.norm(p - q) >= min_mm for q in pts):
            pts.append(p)
    return np.array(pts)


def built_map(world):
    m = mm.MarkerMap()
    m.add(world, np.eye(4))
    m.add(world, np.eye(4))  # two sightings: they join the map
    return m


def test_global_match_recovers_pose_with_noise_outliers_and_missing():
    rng = np.random.default_rng(3)
    world = spaced(rng, 15, -60, 60, scale=(1, 1, 0.3))
    m = built_map(world)
    assert len(m) == 15
    pose = rigid(rng)  # camera -> world
    seen = world[[1, 4, 7, 11]]  # 4 of 15 markers in view
    cam = (seen - pose[:3, 3] * 1000) @ pose[:3, :3] + rng.normal(0, 0.3, (4, 3))
    cam = np.vstack([cam, [5.0, 5.0, 260.0]])  # one false detection
    found = m.match_global(cam)
    assert found is not None
    # markers land where they belong (what matters on the object); the camera position itself is
    # swung by a few mm of lever arm when the seen markers are nearly coplanar and 25 cm away
    assert np.linalg.norm(mm.apply(found, cam[:4]) - seen, axis=1).max() < 1.0
    assert np.linalg.norm(found[:3, 3] - pose[:3, 3]) * 1000 < 6.0


def test_symmetric_layout_is_refused():
    world = np.array([[0, 0, 250.0], [40, 0, 250], [20, 34.641, 250]])  # equilateral: 3 equally good answers
    assert built_map(world).match_global(world) is None


def test_collinear_markers_are_rejected():
    world = np.array([[0, 0, 250.0], [20, 0, 250], [45, 0, 250], [70, 0, 250]])
    assert built_map(world).match_global(world) is None


def test_predicted_match_and_kabsch():
    rng = np.random.default_rng(4)
    world = spaced(rng, 8, -60, 60)
    m = built_map(world)
    pose = rigid(rng, 20, 10)
    cam = (world - pose[:3, 3] * 1000) @ pose[:3, :3]
    near = pose.copy()
    near[:3, 3] += 0.004  # prediction 4 mm off
    pairs = m.match_predicted(cam, near)
    assert len(pairs) == 8
    est = m.pose_from(cam, pairs)
    assert np.abs(est[:3, 3] - pose[:3, 3]).max() * 1000 < 1e-6
