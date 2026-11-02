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


def test_depth_preview_paints_the_live_model_of_the_tracked_frame():
    """While scanning with the GPU tracker the preview shows the frame the worker tracked (not the newest),
    with the model painted over it: grey where seen often enough, yellow too few times, orange missing."""
    import time
    from tribuutscan.depth import encode_y10

    class Cap:
        width, height, bytesperline = 1280, 800, 1600

    as_array = lambda ppm: np.frombuffer(ppm.split(b"\n", 1)[1], np.uint8).reshape(500, 800, 3).astype(int)
    tracked, newest = encode_y10(np.full((800, 1280), 400, np.uint16)), encode_y10(np.full((800, 1280), 100, np.uint16))
    viewer.scan_view.update(mask=None, t=0.0)
    plain = as_array(viewer.render_depth(Cap, tracked)[0])
    codes = np.zeros((400, 640), np.uint8)
    codes[:, :160] = 255  # done, facing the scanner
    codes[:, 160:320] = 127  # seen too few times, facing
    codes[:, 320:480] = 1  # the object, not in the model yet
    mask = np.full((100, 160), 2, np.uint8)  # all object: nothing dimmed
    try:
        viewer.scan_view.update(mask=mask, t=time.monotonic(), lost=False, model=(codes, tracked))
        ppm, stats = viewer.render_depth(Cap, newest)
        shown = as_array(ppm)
    finally:
        viewer.scan_view.update(mask=None, t=0.0, model=None)
    assert (shown[:200, :150] == viewer.MODEL_DONE).all()
    assert (shown[:200, 210:390] == viewer.MODEL_THIN).all()
    want = (plain[:200, 410:590] * 0.3 + viewer.MODEL_MISSING * 0.7).astype(np.uint8)
    assert (shown[:200, 410:590] == want).all()
    assert (shown[:200, 610:] == plain[:200, 610:]).all()  # the tracked frame's depth, not the newest's
    assert viewer.MODEL_LEGEND in stats["text"]


def test_status_box_says_what_runs_and_what_it_uses():
    """The Status box's text from the viewer's state, without a scanner or a window: idle, then a scan
    (two sleeping processes stand in for the worker and a meshing job) beside a recording."""
    import re
    import subprocess
    import sys
    import types
    from collections import deque
    from pathlib import Path

    class Proc:
        def __init__(self):
            self.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            self.pid = self.child.pid

        def is_alive(self):
            return self.child.poll() is None

    app = viewer.App.__new__(viewer.App)  # the state only
    app.live = {"proc": None, "url": None, "done": deque(maxlen=60), "last": None, "name": "", "gpu": "", "merging": ""}
    app.rates, app.depth = viewer.sysinfo.Rates(), types.SimpleNamespace(cap=None)
    app.streams = {name: types.SimpleNamespace(fps=fps) for name, fps in
                   (("Colour", 30.0), ("IR (flipped)", 29.6), ("Depth", 30.2))}
    app.imu, app.imu_note = None, "IMU needs udev/70-tribuutscan.rules (see README)"
    app.job, app.scan_state, app.lost = {"process": None, "dir": None}, "idle", False
    app.switching, app.auto, app.rec = {"busy": False}, {"running": False}, {"recorder": None, "missed": 0}
    app.range_names = {mode: label for label, mode in viewer.RANGES.items()}
    lines = app.status_lines()
    assert [line[:viewer.LABEL_W].strip() for line in lines] == \
        ["Now", "Scanner", "Camera", "Streams", "Scan worker", "CPU", "Memory", "GPU", "Disk"]
    assert all(line[viewer.LABEL_W - 1] == " " for line in lines)  # the labels' column
    text = "\n".join(lines)
    assert "idle" in lines[0] and "10-bit (V4L2)" in text and "colour 30, ir 30, depth 30 fps" in text
    assert "udev" in text and "not started" in text and "not in use" in text and "GB free for captures/" in text

    worker, job = Proc(), Proc()
    try:
        app.scan_state = "scanning"
        app.live.update(proc=worker, name="20261102-170723-scan", gpu="Intel(R) Arc(tm) A750 Graphics (DG2)",
                        last=(12, 80, 30411, False), url="http://localhost:8765/")
        app.live["done"].extend([10.0, 10.5, 11.0])  # three frames back in a second
        app.rec["recorder"] = types.SimpleNamespace(dir=Path("20261102-171000-rec"), count=40, megabytes=120.4)
        app.job = {"process": job, "dir": Path("20261102-160000-rec")}
        app.status_lines()
        lines = app.status_lines()  # (the CPU rates need a reading before)
        text = "\n".join(lines)
        assert lines[0].split(None, 1)[1] == ("scanning 20261102-170723-scan: 80 frames (2.0 a second), 12 merged, "
                                              "30,411 points, TRACKING LOST; recording 20261102-171000-rec: 40 sets, "
                                              "120 MB, 0 missed; meshing 20261102-160000-rec")
        assert f"pid {worker.pid}, tracks on Intel(R) Arc(tm) A750 Graphics (DG2)" in text and "localhost:8765" in text
        cpu = next(line for line in lines if line.startswith("CPU"))
        assert all(re.search(rf"{name} \d+%", cpu) for name in ("viewer", "worker", "meshing")), cpu
        memory = next(line for line in lines if line.startswith("Memory"))
        assert "viewer" in memory and "worker" in memory and "meshing" in memory
    finally:
        for p in (worker, job):
            p.child.kill()
            p.child.wait()
