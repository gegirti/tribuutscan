import numpy as np

from tribuutscan import viewer


def test_previews_resample_to_any_width():
    img = np.arange(800 * 1280).reshape(800, 1280)
    for width in (viewer.SIDE_W, viewer.DEPTH_W, 1280, 333):
        out = viewer.fit(img, width)
        assert out.shape == (round(800 * width / 1280), width)
    assert viewer.fit(img, 640)[1, 1] == img[2, 2]  # half size: every second pixel


def test_open_app_asks_to_retry_until_the_scanner_opens():
    tries, asked = [], []

    class FlakyApp:
        def __init__(self, root):
            tries.append(root)
            if len(tries) < 3:
                raise RuntimeError("scanner not found or unexpected layout: {}")

    real = viewer.App
    viewer.App = FlakyApp
    try:
        app = viewer.open_app("root", ask=lambda title, text, parent: asked.append(text) or True)
        assert isinstance(app, FlakyApp) and len(tries) == 3 and len(asked) == 2
        tries.clear()
        assert viewer.open_app("root", ask=lambda title, text, parent: False) is None and len(tries) == 1  # gave up
    finally:
        viewer.App = real


def test_depth_preview_dims_all_but_the_object():
    import time
    from tribuutscan.depth import encode_y10

    class Cap:
        width, height, bytesperline = 1280, 800, 1600

    buf = encode_y10(np.full((800, 1280), 400, np.uint16))
    as_array = lambda ppm: np.frombuffer(ppm.split(b"\n", 1)[1], np.uint8).reshape(500, 800, 3).astype(int)
    plain = as_array(viewer.render_depth(Cap, buf)[0])
    mask = np.zeros((100, 160), np.uint8)
    mask[:, :80] = 2  # object on the left, base top right, the rest bottom right
    mask[:50, 80:] = 1
    try:
        viewer.scan_view.update(mask=mask, t=time.monotonic(), keep_base=True)
        shaded = as_array(viewer.render_depth(Cap, buf)[0])
    finally:
        viewer.scan_view.update(mask=None, t=0.0)
    assert (shaded[:, :380] == plain[:, :380]).all()  # object unchanged (the centre mark aside)
    assert (shaded[300:, 450:] == plain[300:, 450:] // 4).all()  # other dimmed
    assert (shaded[:200, 450:] == plain[:200, 450:] // 2 + viewer.BASE_TINT).all()  # kept base tinted
    assert (shaded[249:251, 386:396] == 255).all() and (shaded[250, 398:402] != 255).any()  # centre mark, open


def test_depth_preview_has_a_red_frame_while_tracking_is_lost():
    import time
    from tribuutscan.depth import encode_y10

    class Cap:
        width, height, bytesperline = 1280, 800, 1600

    buf = encode_y10(np.full((800, 1280), 400, np.uint16))
    as_array = lambda ppm: np.frombuffer(ppm.split(b"\n", 1)[1], np.uint8).reshape(500, 800, 3)
    viewer.scan_view.update(mask=None, t=0.0)
    plain = as_array(viewer.render_depth(Cap, buf)[0])
    try:
        viewer.scan_view.update(t=time.monotonic(), lost=True)
        lost = as_array(viewer.render_depth(Cap, buf)[0])
        viewer.scan_view.update(t=0.0)  # no frame for a second: not scanning any more
        idle = as_array(viewer.render_depth(Cap, buf)[0])
    finally:
        viewer.scan_view.update(mask=None, lost=False, t=0.0)
    for edge in (lost[0], lost[-1], lost[:, 0], lost[:, -1]):
        assert (edge == viewer.LOST_RGB).all()
    assert (lost[100:400, 100:300] == plain[100:400, 100:300]).all()  # inside unchanged
    assert (idle == plain).all()
