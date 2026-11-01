import types

from tribuutscan import bench

FDINFO = """pos:\t0
drm-driver:\ti915
drm-client-id:\t7
drm-pdev:\t0000:03:00.0
drm-total-system0:\t2048 KiB
drm-total-local0:\t3 MiB
drm-resident-local0:\t1 MiB
drm-engine-render:\t1500000000 ns
drm-engine-capacity-video:\t2
"""


def test_gpu_usage_from_fdinfo():
    other = FDINFO.replace("client-id:\t7", "client-id:\t8").replace("3 MiB", "1048576")  # plain bytes
    usage = bench.parse_fdinfo([FDINFO, FDINFO, other, "pos:\t0\nflags:\t02\n"])  # fds of client 7 count once
    assert usage["driver"] == "i915"
    assert usage["memory_mib"] == {"system0": 4.0, "local0": 4.0}
    assert usage["busy_s"] == {"render": 3.0}
    assert bench.parse_fdinfo(["pos:\t0\n"]) is None


def test_stage_timer_times_only_when_on_and_restores():
    owner = types.SimpleNamespace(step=lambda x: x + 1)
    owner.__dict__["step"] = owner.step
    original = owner.step
    with bench.StageTimer([(owner, "step", "step")]) as timer:
        assert owner.step(1) == 2 and not timer.calls
        timer.on = True
        owner.step(1)
        owner.step(2)
    assert owner.step is original
    assert timer.calls["step"] == 2 and timer.seconds["step"] >= 0


def test_bench_runs_on_a_short_recording():
    try:
        from tribuutscan import synth
    except ImportError:  # needs Open3D (.venv)
        return
    import tempfile
    from pathlib import Path
    from tribuutscan import calib, cloud
    before, feed = calib.CAMERA.spec, cloud.Scanner.feed
    calib.use(calib.ESTIMATES)  # synthetic frames are rendered with it
    try:
        with tempfile.TemporaryDirectory() as tmp:
            synth.generate(Path(tmp) / "t-rec", n_frames=bench.WARMUP + 4)
            report = bench.run(Path(tmp) / "t-rec", log=lambda m: None)
    finally:
        calib.use(before)
    assert cloud.Scanner.feed is feed  # unwrapped again
    assert report["frames"] == bench.WARMUP + 4 and report["measured"] == 4
    rows = {r["stage"]: r for r in report["stages"]}
    assert rows["frame (all of feed)"]["calls"] == 4 and rows["solve"]["calls"] >= 1
    assert report["camera"]["source"] == "estimate" and report["machine"]["numpy"]
