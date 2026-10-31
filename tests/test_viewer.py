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
        viewer.scan_view.update(mask=None)
    assert (shaded[:, :400] == plain[:, :400]).all()  # object unchanged
    assert (shaded[300:, 450:] == plain[300:, 450:] // 4).all()  # other dimmed
    assert (shaded[:200, 450:] == plain[:200, 450:] // 2 + viewer.BASE_TINT).all()  # kept base tinted
