import numpy as np

from tribuutscan import calib


def test_lens_model_round_trip():
    d = [0.08, -0.15, 0.001, -0.002, 0.05]
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 9), np.linspace(-0.35, 0.35, 7))
    xd, yd = calib.distort(x, y, d)
    xu, yu = calib.undistort(xd, yd, d)
    assert np.abs(xu - x).max() < 1e-6 and np.abs(yu - y).max() < 1e-6


def test_estimates_reproduce_the_old_mapping():
    cam = calib.Camera(calib.ESTIMATES)
    x, y = cam.colour_px(np.array([100, 900]), np.array([50, 700]), z_mm=180.0)  # distance ignored
    r = calib.ESTIMATES["colour_from_depth"]
    assert np.allclose(x, (np.array([100, 900]) - r["bx"]) / r["ax"]) and np.allclose(y, np.array([50, 700]) - r["by"])


def test_depth_law_fit_recovers_a_and_b():
    try:
        from tribuutscan import calibrate
    except ImportError:  # needs OpenCV (.venv)
        return
    rng = np.random.default_rng(0)
    z = rng.uniform(180, 310, 5000)
    raw = 320000.0 / z - 1000.0 + rng.normal(0, 0.3, len(z))
    form, laws = calibrate.fit_law(raw, z)
    assert form == "a/z" and abs(laws["a/z"]["A"] - 320000) < 2000 and laws["a/z"]["median_mm"] < 0.3
