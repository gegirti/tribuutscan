import numpy as np

from tribuutscan import cloud as cl


def test_the_merge_process_builds_what_the_merge_thread_builds():
    """A synthetic scan with its colour left as YUYV (as the live worker gets it), merged on the thread and
    in the merge process: the same cloud, merged frames, keyframes' ORB, markers and poses."""
    try:
        from tribuutscan import synth
    except ImportError:  # needs Open3D (.venv)
        return
    import tempfile
    from pathlib import Path
    from tribuutscan import calib
    from tribuutscan import frames as fr
    from tribuutscan.merger import CLOUD_ARRAYS
    before = calib.CAMERA.spec
    calib.use(calib.ESTIMATES)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp) / "t-rec"
            synth.generate(rec, n_frames=16)
            frames = list(fr.read_recording(rec, yuyv=True))
            runs = []
            for process in (False, True):
                sc = cl.Scanner(log=lambda *a: None, track_with="gpu", merge_min_s=0.0, merge_process=process)
                assert (sc.merge_proc is not None) == process
                for i, frame in enumerate(frames):
                    sc.feed(frame, i)
                if process:
                    live = sc.live_points()  # the page's view while scanning: never waits
                    xyz, rgb, seen = sc.cloud_points(1)  # waits for every merge sent
                sc.flush()
                sc.close()
                if process:
                    assert sc.merge_proc is None  # the state came back and the process ended
                    want = sc.cloud.points(1)
                    assert all(np.array_equal(a, b) for a, b in zip((xyz, rgb, seen), want))
                    assert len(live[0]) <= len(xyz)
                runs.append(sc)
            thread, proc = runs
            assert thread.stats["merged"] >= 4 and len(thread.cloud) > 1000
            for name in CLOUD_ARRAYS:
                assert np.array_equal(getattr(thread.cloud, name), getattr(proc.cloud, name)), name
            assert thread.cloud.raw_points == proc.cloud.raw_points and thread.points_kept == proc.points_kept
            assert sorted(thread.merged_data) == sorted(proc.merged_data)
            for i in thread.merged_data:
                assert all(np.array_equal(a, b) for a, b in zip(thread.merged_data[i], proc.merged_data[i]))
            assert thread.stats["markers"] == proc.stats["markers"] and thread.stats["markers"]
            assert len(thread.keyframes) == len(proc.keyframes)
            for a, b in zip(thread.keyframes, proc.keyframes):
                assert a[0] == b[0] and np.array_equal(a[1], b[1])
                if a[3] is not None or b[3] is not None:
                    assert np.array_equal(a[3][0], b[3][0]) and np.array_equal(a[3][1], b[3][1])
            assert len(thread.track_log) == len(proc.track_log)
            assert all((p is None and q is None) or (p is not None and q is not None and np.array_equal(p, q))
                       for (_, _, p), (_, _, q) in zip(thread.track_log, proc.track_log))
    finally:
        calib.use(before)


def test_without_a_merge_process_the_scan_merges_on_the_thread():
    from tribuutscan import merger
    real, said = merger.MergeProcess, []

    def broken(*a, **k):
        raise OSError("no spawn here")

    merger.MergeProcess = broken
    try:
        sc = cl.Scanner(log=said.append, merge_process=True)
    finally:
        merger.MergeProcess = real
    assert sc.merge_proc is None and sc.merger is not None and "merge process did not start" in said[-1]


def test_the_merge_process_follows_the_depth_mode_of_each_job():
    """A replay switches to its recording's depth mode only as the frames are read, after the Scanner
    (and so the merge process) has started: each job carries the mode it was sent under."""
    from tribuutscan import calib, merger
    before = calib.CAMERA.mode
    rng = np.random.default_rng(18)
    depth = np.zeros((800, 1280), np.float32)
    depth[200:600, 300:900] = rng.uniform(300, 600, (400, 600)).astype(np.float32)
    frame = {"depth": depth, "yuyv": rng.integers(0, 256, 1280 * 800 * 2, np.uint8).tobytes()}
    v, u = np.mgrid[200:600:4, 300:900:4]
    pix = np.stack([u.ravel(), v.ravel()], 1)
    calib.use_mode(calib.CLOSE)
    proc = merger.MergeProcess(0.5, 2, False, lambda reply: None)
    try:
        calib.use_mode(calib.LONG)
        proc.submit(frame, 0, np.eye(4), pix, 4)
        _, merged = proc.state()
        want = cl.merge_points(frame, pix, 4, 2, False)
    finally:
        proc.close()
        calib.use_mode(before)
    assert np.array_equal(merged[0][0], want[0].astype(np.float32)) and len(want[0]) > 10000


def _start_from_a_daemon(out):
    """Runs in a daemonic process, as the live worker does: start a merge process there."""
    from tribuutscan import merger
    proc = merger.MergeProcess(0.5, 2, False, lambda reply: None)
    out.put(proc.proc.is_alive())
    arrays, _ = proc.state()
    out.put(len(arrays["keys"]))


def test_a_daemonic_process_can_start_the_merge_process():
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    worker = ctx.Process(target=_start_from_a_daemon, args=(out,), daemon=True)
    worker.start()
    try:
        assert out.get(timeout=60) is True and out.get(timeout=60) == 0
    finally:
        worker.join(30)


def test_a_merge_process_lost_mid_scan_leaves_the_scan_merging_on_the_thread():
    said = []
    sc = cl.Scanner(log=said.append, merge_process=True)
    assert sc.merge_proc is not None
    sc.merge_proc.proc.kill()
    sc.merge_proc.proc.join(10)
    xyz, _, _ = sc.cloud_points(1)  # a wait on it finds it gone: the local (thread) cloud answers
    assert len(xyz) == 0 and sc.merge_proc is None and sc.merger is not None and "merge process lost" in said[-1]
    sc.flush()  # nothing left to pull: the local cloud stays


def test_live_frames_with_their_depth_packed_track_as_decoded_ones():
    """A 14-bit synthetic scan fed as the live worker feeds it (depth.LazyFrame: every 4th pixel decoded on
    the CPU, the rest unpacked by the GPU and by whoever merges) and fully decoded: the same poses, cloud,
    keyframes; in the thread and in the merge process."""
    try:
        from tribuutscan import synth
    except ImportError:
        return
    import tempfile
    from pathlib import Path
    from tribuutscan import calib, depth as dp
    from tribuutscan import frames as fr
    from tribuutscan.merger import CLOUD_ARRAYS
    before = calib.CAMERA.spec
    calib.use(calib.ESTIMATES)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp) / "t-rec"
            synth.generate(rec, n_frames=16, bits=14)
            decoded = list(fr.read_recording(rec, yuyv=True))
            h, w = decoded[0]["depth"].shape

            def lazy():
                for b in fr.read_raw(rec):
                    d4, cleared = dp.decode_strided(b["depth"], w, h, b["bytesperline"])
                    yield dp.LazyFrame(yuyv=b["color"], depth_raw=b["depth"], bytesperline=b["bytesperline"], depth4=d4,
                                       cleared=cleared, shape=(h, w), t=b["t"][2], skew_ms=(b["t"][0] - b["t"][2]) * 1000)

            runs, real, decodes = [], dp.decode_depth, []
            for frames, process in ((decoded, False), (lazy(), False), (lazy(), True)):
                sc = cl.Scanner(log=lambda *a: None, track_with="gpu", merge_min_s=0.0, merge_process=process)
                dp.decode_depth = lambda *a, **k: decodes.append(process) or real(*a, **k)
                try:
                    for i, frame in enumerate(frames):
                        sc.feed(frame, i)
                    sc.flush()  # (the thread's merges decode: counted till they are done)
                finally:
                    dp.decode_depth = real
                sc.close()
                runs.append(sc)
            # with the merge process no full CPU decode here at all (this box's markers take the depth from the GPU)
            assert decodes.count(True) == 0
            ref = runs[0]
            assert ref.stats["merged"] >= 4 and len(ref.cloud) > 1000
            for sc in runs[1:]:
                assert all((p is None and q is None) or (p is not None and q is not None and np.array_equal(p, q))
                           for (_, _, p), (_, _, q) in zip(ref.track_log, sc.track_log))
                for name in CLOUD_ARRAYS:
                    assert np.array_equal(getattr(ref.cloud, name), getattr(sc.cloud, name)), name
                assert [k[0] for k in ref.keyframes] == [k[0] for k in sc.keyframes]
    finally:
        calib.use(before)
