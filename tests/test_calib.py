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


# our unit's factory calibration (read 2026-11-01), as vendor.Channel.factory_calibration returns it
FACTORY = {
    "depth_calib": {"depth_mode": 1, "baseline": 150.2841, "z0": 662.1558, "focal": 960.5139, "unit": 0.5,
                    "disp_offset": 218.0, "invalid_disp": 0},
    "cameras": [{
        "depth": {"fx": 960.5139, "fy": 960.5139, "cx": 646.2972, "cy": 398.9413, "w": 1280, "h": 800},
        "colour": {"fx": 842.0515, "fy": 842.3624, "cx": 637.1276, "cy": 395.3939, "w": 1280, "h": 800},
        "depth_dist": dict.fromkeys(("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"), 0.0),
        "colour_dist": {"k1": 13.2937, "k2": -30.6221, "k3": 19.6306, "k4": 12.8383, "k5": -29.3117,
                        "k6": 18.6331, "p1": -0.00054969, "p2": 9.0764e-05},
        "R": [0.99999, -0.0024178, 0.0037682, 0.0024172, 0.999997, 0.0001689, -0.0037686, -0.0001598, 0.999993],
        "T_mm": [19.5479, -0.0345, -0.7658]}],
    "profiles": [{"colour": (1280, 720), "depth": (1280, 800), "align": 3, "record": 0, "scale": 1.0,
                  "crop": (0, -40, 0, -40)},
                 {"colour": (1280, 800), "depth": (1280, 800), "align": 3, "record": 255, "scale": 1.0,
                  "crop": (0, 0, 0, 0)}],
}


def test_rational_lens_model_round_trip():
    d = [13.2937, -30.6221, -0.00054969, -9.0764e-05, 19.6306, 12.8383, -29.3117, 18.6331]
    x, y = np.meshgrid(np.linspace(-0.75, 0.75, 9), np.linspace(-0.45, 0.45, 7))
    xu, yu = calib.undistort(*calib.distort(x, y, d), d)
    assert np.abs(xu - x).max() < 1e-6 and np.abs(yu - y).max() < 1e-6
    assert np.allclose(calib.distort(x, y, d[:5]), calib.distort(x, y, d[:5] + [0, 0, 0]))


def test_factory_spec_matches_the_factory_formulas():
    """A depth pixel of OUR frame, taken through the factory formulas in the factory's (mirrored) frame,
    lands on the colour pixel the converted spec gives."""
    cam = calib.Camera(calib.from_factory(FACTORY, calib.CLOSE))
    rec, w = FACTORY["cameras"][0], 1280
    u, v, raw = np.array([100.0, 640.0, 1100.0]), np.array([80.0, 400.0, 700.0]), np.array([50.0, 280.0, 900.0])
    dc = FACTORY["depth_calib"]
    z = dc["unit"] * dc["baseline"] * dc["focal"] / (raw / 4 + dc["disp_offset"])
    assert np.allclose(cam.depth_mm(raw), z)
    d, c, cd = rec["depth"], rec["colour"], rec["colour_dist"]
    p = np.stack([(w - 1 - u - d["cx"]) / d["fx"] * z, (v - d["cy"]) / d["fy"] * z, z], -1)
    q = p @ np.reshape(rec["R"], (3, 3)).T + rec["T_mm"]
    xd, yd = calib.distort(q[:, 0] / q[:, 2], q[:, 1] / q[:, 2],
                           [cd["k1"], cd["k2"], cd["p1"], cd["p2"], cd["k3"], cd["k4"], cd["k5"], cd["k6"]])
    x_rec, y_rec = xd * c["fx"] + c["cx"], yd * c["fy"] + c["cy"]
    x, y = cam.colour_px(u, v, z)
    assert np.allclose(x, w - 1 - x_rec, atol=1e-6) and np.allclose(y, (y_rec - 40 + 0.5) * 800 / 720 - 0.5, atol=1e-6)
    assert np.allclose(cam.depth_px(x, y, q[:, 2]), (u, v), atol=1e-3)  # and back (depth_px: colour-axis distance)


def test_disparity_offsets_make_long_range_exact():
    import json
    import os
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        board, factory = Path(tmp) / "calibration.json", Path(tmp) / "factory.json"
        board.write_text(json.dumps({**calib.ESTIMATES, "estimate": False, "law": {"a_mm": 289570.0, "b": 873.4}}))
        cam = calib.Camera(calib.chosen_spec(board, factory))
        cam.set_mode(calib.LONG)
        assert cam.source == "board" and np.isclose(cam.b, 873.4 - 472) and cam.law_estimate  # default offsets
        long_read = {**FACTORY, "depth_calib": {**FACTORY["depth_calib"], "disp_offset": 100.0}}
        factory.write_text(json.dumps(calib.from_factory(long_read, calib.LONG, {calib.CLOSE: 218.0})))
        cam = calib.Camera(calib.chosen_spec(board, factory))
        cam.set_mode(calib.LONG)
        assert np.isclose(cam.b, 873.4 - 472) and not cam.law_estimate  # this scanner's own offsets
        os.environ["TRIBUUTSCAN_CAMERA"] = "factory"
        try:
            cam = calib.Camera(calib.chosen_spec(board, factory))
        finally:
            del os.environ["TRIBUUTSCAN_CAMERA"]
        cam.set_mode(calib.LONG)
        assert cam.source == "factory" and np.isclose(cam.b, 400) and not cam.law_estimate
        cam.set_mode(calib.CLOSE)
        assert np.isclose(cam.b, 872) and not cam.law_estimate


def test_note_factory_keeps_the_offsets_of_one_scanner():
    import json
    import tempfile
    from pathlib import Path
    before = calib.CAMERA.spec
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "factory.json"
            calib.note_factory(FACTORY, calib.CLOSE, path)
            long_read = {**FACTORY, "depth_calib": {**FACTORY["depth_calib"], "disp_offset": 100.0}}
            calib.note_factory(long_read, calib.LONG, path)
            assert json.loads(path.read_text())["disp_offsets"] == {calib.CLOSE: 218.0, calib.LONG: 100.0}
            other = {**FACTORY, "depth_calib": {**FACTORY["depth_calib"], "baseline": 149.907}}
            calib.note_factory(other, calib.CLOSE, path)
            assert json.loads(path.read_text())["disp_offsets"] == {calib.CLOSE: 218.0}
    finally:
        calib.use(before)
