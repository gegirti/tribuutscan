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


def test_ir_corners_through_projector_dots():
    try:
        import cv2
        from tribuutscan import calibrate
    except ImportError:  # needs OpenCV (.venv)
        return
    rng = np.random.default_rng(3)
    brd = calibrate.board()
    k = np.array([[1108.0, 0, 640], [0, 1108.0, 400], [0, 0, 1]])
    rot = cv2.Rodrigues(np.radians([12.0, -15.0, 3.0]))[0]
    to_img = k @ np.c_[rot[:, :2], np.array([0, 0, 260.0]) - rot @ np.array([132.0, 96.0, 0])]  # board mm -> px
    tpl, to_px = calibrate.template(brd, 10.0)
    refl = cv2.warpPerspective(tpl / 255 * 0.84 + 0.08, to_img @ np.linalg.inv(to_px), (1280, 800), borderValue=0.25)
    dots = cv2.GaussianBlur((rng.random((800, 1280)) < 1 / 20).astype(np.float32), (0, 0), 0.7) / 0.325
    ir = np.clip(230 * refl * dots + 3 + rng.normal(0, 3, refl.shape), 0, 255).astype(np.uint8)  # lit by the dots only
    obj = brd.getChessboardCorners()
    truth = cv2.perspectiveTransform(np.float64(obj[:, :2]).reshape(-1, 1, 2), to_img).reshape(-1, 2)
    r = calib.ESTIMATES["colour_from_depth"]  # a colour detection that maps 15 px off in IR
    col = (obj, np.float32(np.c_[(truth[:, 0] + 15 - r["bx"]) / r["ax"], (truth[:, 1] - r["by"]) / r["ay"]]),
           np.arange(len(obj)))
    found = calibrate.ir_corners(ir, col, brd)
    assert found is not None and len(found[2]) >= 60
    err = np.linalg.norm(found[1] - truth[found[2]], axis=1)  # the dots leave about 0.3 px per corner
    assert np.median(err) < 0.4 and err.max() < 1.0


def test_depth_modes_switch_law_and_window():
    cam = calib.Camera({**calib.ESTIMATES, "estimate": False})
    a, b, window = cam.base_law
    assert cam.mode == calib.CLOSE and (cam.a, cam.b, cam.window_mm) == (a, b, window) and not cam.law_estimate
    cam.set_mode(calib.LONG)
    assert cam.b == b + calib.LONG_B_SHIFT and cam.law_estimate
    assert np.allclose(cam.window_mm, (cam.depth_mm(1023), cam.depth_mm(1)))
    cam.set_mode(calib.CLOSE)
    assert (cam.a, cam.b, cam.window_mm) == (a, b, window)


def test_a_calibrated_mode_wins_over_the_estimate():
    spec = {**calib.ESTIMATES, "modes": {calib.LONG: {"law": {"a_mm": 300000.0, "b": 400.0}, "window_mm": [210, 700]}}}
    cam = calib.Camera(spec)
    cam.set_mode(calib.LONG)
    assert (cam.a, cam.b, cam.window_mm, cam.law_estimate) == (300000.0, 400.0, (210, 700), False)


def test_long_estimate_matches_the_50cm_check():
    cam = calib.Camera({**calib.ESTIMATES, "law": {"a_mm": 289570.18, "b": 873.42}})
    cam.set_mode(calib.LONG)
    assert abs(cam.depth_mm(175.7) - 501) < 2  # hand-measured 50 cm read 175.7 (2026-10-31)
