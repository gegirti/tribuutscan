import numpy as np

from tribuutscan import metrics


def sheet(z, n, rng, count=20000):
    xyz = np.c_[rng.uniform(0, 60, (count, 2)), np.full(count, z)]
    return xyz, np.tile(n, (count, 1))


def test_doubled_sheet_counts_thin_wall_does_not():
    rng = np.random.default_rng(1)
    a, na = sheet(0.0, [0, 0, 1.0], rng)
    b, nb = sheet(4.0, [0, 0, 1.0], rng)  # a shifted copy facing the same way: a pose error
    doubled = metrics.double_surface_ratio(np.r_[a, b], np.r_[na, nb])
    c, nc = sheet(4.0, [0, 0, -1.0], rng)  # the other side of a 4 mm wall faces the other way
    wall = metrics.double_surface_ratio(np.r_[a, c], np.r_[na, nc])
    single = metrics.double_surface_ratio(a, na)
    assert doubled > 0.9 and wall < 0.05 and single < 0.05


def test_pose_jumps_flags_only_the_jump():
    poses = []
    for i in range(10):
        p = np.eye(4)
        p[0, 3] = 0.005 * i + (0.12 if i == 5 else 0.0)  # 5 mm steps, one 120 mm jump out and back
        poses.append((i, p))
    jumps = metrics.pose_jumps(poses, centre_mm=[0, 0, 250])
    assert [(a, b) for a, b, _, _ in jumps] == [(4, 5), (5, 6)]


def test_marker_scatter_sees_smearing():
    rng = np.random.default_rng(2)
    markers = rng.uniform(-50, 50, (6, 3))
    tight = np.repeat(markers, 20, axis=0) + rng.normal(0, 0.3, (120, 3))
    good = metrics.marker_scatter(tight)
    assert good["markers"] == 6 and good["scatter_mm"] < 0.8
    shifted = np.r_[tight, np.repeat(markers + [40, 0, 0], 20, axis=0)]  # a doubled half
    bad = metrics.marker_scatter(shifted)
    assert bad["markers"] > 6


def test_ate_rpe_zero_for_perfect_tracking():
    truth = {}
    for i in range(5):
        p = np.eye(4)
        p[:3, 3] = [0.01 * i, 0, 0]
        truth[i] = p
    r = metrics.ate_rpe(dict(truth), truth)
    assert r["ate_mm"] < 1e-9 and r["rpe_mm"] < 1e-9 and r["frames"] == 5
