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


def test_depth_preview_holds_depth_for_a_moment():
    hv = viewer.hold_view
    saved = dict(hv)
    try:
        hv.update(on=True, s=0.5, depth=None)
        a = np.zeros((4, 6), np.float32)
        a[1, 2] = 300.0
        shown, held = viewer.hold_depth(a, 10.0)
        assert shown[1, 2] == 300 and not held.any()
        b = np.zeros_like(a)
        b[3, 3] = 200.0
        shown, held = viewer.hold_depth(b, 10.3)  # within 0.5 s: the old pixel held, the new one live
        assert shown[1, 2] == 300 and held[1, 2] and shown[3, 3] == 200 and not held[3, 3]
        shown, held = viewer.hold_depth(b, 10.6)  # 0.6 s after it was last seen: gone
        assert shown[1, 2] == 0 and not held[1, 2] and shown[3, 3] == 200
        hv["mode"] = "another mode"  # a range switch: nothing is held over
        shown, held = viewer.hold_depth(np.zeros_like(a), 10.7)
        assert not shown.any() and not held.any()
    finally:
        hv.clear()
        hv.update(saved)


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
    app.switching, app.ir_auto, app.rec = {"busy": False}, True, {"recorder": None, "missed": 0}
    app.range_names = {mode: label for label, mode in viewer.RANGES.items()}
    lines = app.status_lines()
    assert [line[:viewer.LABEL_W].strip() for line in lines] == \
        ["Now", "Scanner", "Camera", "Streams", "Scan worker", "CPU", "Memory", "GPU", "Disk"]
    assert all(line[viewer.LABEL_W - 1] == " " for line in lines)  # the labels' column
    text = "\n".join(lines)
    assert "idle" in lines[0] and "10-bit (V4L2)" in text and "colour 30, ir 30, depth 30 fps" in text
    assert "IR auto (the scanner's own)" in text
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


def _aim_scene():
    """A 14-bit depth frame: a flat table (raw 300) with two boxes on it (raw 420): one over the image centre,
    one off to the left."""
    from tribuutscan.depth import encode_y14

    class Cap:
        width, height, bytesperline = 1280, 800, 1280 * 14 // 8

    raw = np.full((800, 1280), 300.0)
    raw[300:500, 560:760] = 420.0  # under the centre
    raw[300:500, 150:350] = 420.0  # off to the left
    return Cap, encode_y14((raw * 16 + 8).astype(np.uint16))


def test_the_aim_outlines_the_centre_piece_and_follows_a_pick():
    import collections
    Cap, buf = _aim_scene()
    saved = dict(viewer.aim_view)
    try:
        viewer.aim_view.update(pick=None, follow=None, idle=True)
        piece = viewer.aim_once(Cap, buf, collections.deque(maxlen=5))
        assert piece is not None and piece["at"] is None  # automatic: the image centre, as a scan without an aim
        u = piece["pix"][:, 0]
        assert 560 <= u.min() and u.max() < 760 and piece["plane"]  # the centre box, the table cut
        assert piece["coverage"] > 0.9 and piece["colour"].shape == piece["pix"].shape
        viewer.aim_view.update(pick=(250.0, 400.0), follow=None)  # a click on the left box
        piece = viewer.aim_once(Cap, buf, collections.deque(maxlen=5))
        assert 150 <= piece["pix"][:, 0].min() and piece["pix"][:, 0].max() < 350
        assert 150 <= piece["at"][0] < 350 and viewer.aim_view["follow"] is not None
        again = viewer.aim_once(Cap, buf, collections.deque(maxlen=5))  # followed by its centre
        assert np.array_equal(again["pix"], piece["pix"])
        viewer.aim_view["t"] = viewer.time.monotonic()
        assert viewer.fresh_aim() is again
        viewer.aim_view["idle"] = False  # scanning: no aim drawn or handed over
        assert viewer.fresh_aim() is None
    finally:
        viewer.aim_view.clear()
        viewer.aim_view.update(saved)


def test_the_aim_hint_names_what_needs_doing():
    from tribuutscan import calib
    near, far = viewer.pointcloud.window_cm()
    piece = {"z": (near + far) / 2 * 10, "coverage": 0.6, "plane": True, "at": None}
    text, colour = viewer.aim_hint(piece, {"clipped": 0.5}, None, False)
    assert colour == viewer.GOOD and "depth 60%" in text and not text.startswith("Picked")
    text, colour = viewer.aim_hint(dict(piece, coverage=0.1, plane=False, at=(1, 2)), {"clipped": 9.0},
                                   {"pitch": -30.0}, False)
    assert colour == viewer.BAD and text.startswith("Picked") and "try another angle" in text
    assert "IR clipping 9%" in text and "no base, 30° down (ok on a platform)" in text
    far_piece = dict(piece, z=(far - 1) * 10)
    text, colour = viewer.aim_hint(far_piece, {}, None, False)
    assert colour == viewer.EDGE and "far edge" in text
    assert ("Long range" in text) == (not calib.CAMERA.mode.startswith("Long"))
    assert viewer.aim_hint(None, {}, None, False)[1] == viewer.BAD


def test_the_countdown_starts_the_scan_at_zero_with_the_picked_aim():
    import time
    app = viewer.App.__new__(viewer.App)  # the state only
    started, logged = [], []
    app.begin_scan = lambda continue_from=None: started.append(continue_from)
    app.log = logged.append
    app.scan_state, app.count = "counting", {"until": time.monotonic() + 2, "from": None}
    text, colour = app.check_countdown()
    assert text.startswith("Starting in 2") and not started
    app.count["until"] = time.monotonic() - 0.01
    assert app.check_countdown() is None and started == [None] and app.scan_state == "idle"
    app.scan_state, app.count = "counting", {"until": time.monotonic() + 2, "from": None}
    app.escape()
    assert app.scan_state == "idle" and logged == ["start cancelled"]
    saved = dict(viewer.aim_view)
    try:
        viewer.aim_view.update(idle=True, t=time.monotonic(), piece={"at": (412.34, 380.0)})
        assert app.aim_for_start() == [412.3, 380.0]
        viewer.aim_view["piece"] = {"at": None}  # automatic: the scan takes the centre itself
        assert app.aim_for_start() is None
    finally:
        viewer.aim_view.clear()
        viewer.aim_view.update(saved)


def test_auto_ir_follows_the_scanner_through_a_rounding_slider():
    """The IR sliders show the scanner's own exposure while Auto IR is on; a slider rounds what it is given
    (its step) and reports the rounded value to its callback: that must not read as a move by hand."""
    class Scale:  # a Tk Scale's rounding: step 100, range 100-6800
        def __init__(self):
            self.v = 3000

        def set(self, v):
            self.v = min(max(round(float(v) / 100) * 100, 100), 6800)

        def get(self):
            return self.v

    app = viewer.App.__new__(viewer.App)
    scale = Scale()
    app.controls = {"IR exposure": ("depth", 1, {}, scale)}
    app.targets = {"depth": None}
    app.synced = {}
    values = iter([5486, 5486, 581])
    real = viewer.get_control
    viewer.get_control = lambda target, cid: next(values)
    try:
        app.follow_ir()
        assert scale.get() == 5500 and app.synced["IR exposure"] == 5500  # what its callback will say
        app.follow_ir()  # the same value again: nothing to do
        assert app.synced["IR exposure"] == 5500
        app.follow_ir()
        assert scale.get() == 600 and app.synced["IR exposure"] == viewer.held(scale) == 600
    finally:
        viewer.get_control = real


def test_the_feeder_keeps_the_worker_fed_off_the_tk_loop():
    """The feeder sends each new depth frame (with its colour pair) as soon as fewer than 2 are in flight and
    hands every reply on with the frame's raw depth, without the Tk loop: a fake worker taking 30 ms a frame
    gets one after another."""
    import queue
    import threading
    import time
    import types
    commands, replies, alive = queue.Queue(), queue.Queue(), threading.Event()
    alive.set()
    proc = types.SimpleNamespace(is_alive=alive.is_set)
    cap = types.SimpleNamespace(bytesperline=2240)
    depth = types.SimpleNamespace(raw=None, stamp=0.0, fresh=threading.Event(), cap=cap)
    color = types.SimpleNamespace(raw=b"c", nearest=lambda t: (b"colour", t - 0.006))
    app = types.SimpleNamespace(live={"proc": proc, "commands": commands, "replies": replies}, depth=depth, color=color,
                                scan_state="scanning")
    feeder = viewer.Feeder(app)
    most, done = [0], []

    def worker():
        while alive.is_set():
            try:
                cmd = commands.get(timeout=0.1)
            except queue.Empty:
                continue
            most[0] = max(most[0], feeder.inflight)
            time.sleep(0.03)
            replies.put(("frame_done", 1, 1, 10, True, {}, None, None, cmd[5][2]))

    threading.Thread(target=worker, daemon=True).start()
    feeder.start()
    try:
        for k in range(1, 31):  # a 100 fps camera: the worker can't take every frame
            depth.raw, depth.stamp = b"d%d" % k, time.monotonic()
            depth.fresh.set()
            time.sleep(0.01)
        end = time.time() + 2
        while time.time() < end and feeder.inflight:
            time.sleep(0.01)
        while True:
            try:
                done.append(feeder.out.get_nowait())
            except queue.Empty:
                break
    finally:
        alive.clear()
    assert most[0] <= 2 and len(done) >= 8 and feeder.inflight == 0
    assert all(kind == "frame_done" and raw is not None and raw.startswith(b"d") for kind, _, raw in done)
    assert feeder.stats["sent"] == len(done) and "a second" in feeder.report()
    assert float(np.median(feeder.stats["lag"])) < 0.02  # sent within ~a camera frame of arriving


def test_a_click_while_scanning_tells_the_worker_where_the_object_is():
    """While scanning, a click on the depth or colour preview goes to the worker as a pick in that image's full-size
    pixels (cloud.Scanner.pick); while optimising it does nothing."""
    import queue

    class Proc:
        def is_alive(self):
            return True

    app = viewer.App.__new__(viewer.App)
    logged = []
    app.log = logged.append
    app.live = {"proc": Proc(), "commands": queue.Queue()}
    app.scan_state = "scanning"
    app.aim_click("Depth", viewer.DEPTH_W / 2, 100)
    app.aim_click("Colour", viewer.SIDE_W / 4, 50)
    kind, how, uv = app.live["commands"].get_nowait()
    assert (kind, how) == ("pick", "depth") and np.allclose(uv, (640, 100 * 1280 / viewer.DEPTH_W))
    kind, how, uv = app.live["commands"].get_nowait()
    assert (kind, how) == ("pick", "colour") and np.allclose(uv, (320, 50 * 1280 / viewer.SIDE_W))
    assert logged and logged[0].startswith("picked")
    app.scan_state = "optimising"
    app.aim_click("Depth", 10, 10)
    assert app.live["commands"].empty()
