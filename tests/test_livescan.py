import numpy as np

from tribuutscan import livescan


def test_view_mask_marks_object_and_base_at_one_eighth():
    pix = np.array([[0, 0], [100, 50], [1279, 799], [640, 400]])
    obj = np.array([False, True, False, True])
    base = np.array([True, False, True, False])
    m = livescan.view_mask((pix, obj, base, (800, 1280)))
    assert m.shape == (100, 160) and m.dtype == np.uint8
    assert (m[0, 0], m[6, 12], m[99, 159], m[50, 80]) == (1, 2, 1, 2) and int((m > 0).sum()) == 4
    assert livescan.view_mask(None) is None


def test_the_raw_copy_writes_every_frame_beside_tracking():
    import tempfile
    from tribuutscan import frames as fr
    rng = np.random.default_rng(16)
    sets = [{"color": rng.integers(0, 256, fr.FRAME_BYTES["color"], np.uint8).tobytes(), "ir": None,
             "depth": rng.integers(0, 256, 1600 * fr.H, np.uint8).tobytes()} for _ in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        copy = livescan.RawCopy(fr.Recorder(1600, root=tmp))
        for i, bufs in enumerate(sets):
            copy.put(bufs, {"color": i + 0.01, "ir": None, "depth": float(i)})
        copy.close()
        back = list(fr.read_raw(copy.recorder.dir))
    assert len(back) == 4 and all(b["color"] == s["color"] and b["depth"] == s["depth"] for b, s in zip(back, sets))
    assert [b["t"][2] for b in back] == [0.0, 1.0, 2.0, 3.0]


def test_a_full_disk_stops_the_raw_copy_not_the_scan():
    import tempfile
    from tribuutscan import frames as fr

    class Full(fr.Recorder):
        def write(self, bufs, stamps):
            raise fr.DiskFull("only 1.0 GB free")

    with tempfile.TemporaryDirectory() as tmp:
        copy = livescan.RawCopy(Full(1600, root=tmp))
        copy.put({}, {})
        copy.put({}, {})  # queued behind the failed one: skipped
        copy.close()
    assert isinstance(copy.error, fr.DiskFull)


def test_the_reader_decodes_every_4th_pixel_beside_tracking_and_keeps_the_order():
    import queue
    from tribuutscan import frames as fr
    from tribuutscan.depth import decode_depth, encode_y14
    rng = np.random.default_rng(17)
    depths = [encode_y14(rng.integers(0, 16000, (fr.H, fr.W), np.uint16)) for _ in range(2)]
    commands = queue.Queue()
    sent = [("start", "small", None, False, None, "close", True)]
    sent += [("frame", b"yuyv", None, d, fr.W * 14 // 8, (i, 0.0, i)) for i, d in enumerate(depths)]
    sent += [("stop",), ("quit",)]
    for cmd in sent:
        commands.put(cmd)
    reader = livescan.Reader(commands)
    reader.start()
    reader.join(30)
    got = [reader.out.get_nowait() for _ in range(len(sent))]
    assert [g[0] for g in got] == [c[0] for c in sent] and reader.out.empty()
    for g, c in zip(got, sent):
        if c[0] == "frame":
            full = decode_depth(c[3], fr.W, fr.H, c[4])
            assert g[:-2] == c and (g[-2] == full[::4, ::4]).all() and g[-1] is None  # (no garbled rows)
        else:
            assert g == c


def test_stop_makes_the_mesh_after_optimise():
    """The live worker, as the viewer runs it, on a short synthetic scan: Stop with the mesh flag replies
    saved, optimised (a mesh follows) and meshed, and mesh.ply / mesh.stl are there."""
    try:
        from tribuutscan import synth
    except ImportError:  # needs Open3D (.venv)
        return
    import multiprocessing
    import os
    import queue
    import tempfile
    from pathlib import Path
    from tribuutscan import calib, livescan
    from tribuutscan import frames as fr
    before, env = calib.CAMERA.spec, os.environ.get("TRIBUUTSCAN_CAMERA")
    calib.use(calib.ESTIMATES)
    os.environ["TRIBUUTSCAN_CAMERA"] = "estimate"  # (the spawned worker: the camera the set is rendered with)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = synth.generate(Path(tmp) / "t-rec", n_frames=40)
            ctx = multiprocessing.get_context("spawn")
            commands, replies = ctx.Queue(), ctx.Queue()
            proc = ctx.Process(target=livescan.run, args=(commands, replies, 0), daemon=True)
            proc.start()
            got = []

            def reply(want):
                while True:
                    try:
                        kind, *rest = replies.get(timeout=120)
                    except queue.Empty:
                        raise AssertionError(f"no {want} from the worker; got {got}")
                    got.append(kind)
                    if kind == want:
                        return rest

            reply("ready")
            commands.put(("start", "small", Path(tmp) / "scan", True, None, calib.CAMERA.mode, True))
            for bufs in fr.read_raw(rec):
                commands.put(("frame", bufs["color"], None, bufs["depth"], bufs["bytesperline"], bufs["t"]))
                reply("frame_done")
            commands.put(("stop", True))
            reply("saved")
            assert reply("optimised")[1] is True
            message = reply("meshed")[0]
            commands.put(("quit",))
            proc.join(30)
            assert message.startswith("mesh:"), message
            assert (Path(tmp) / "scan" / "mesh.ply").exists() and (Path(tmp) / "scan" / "mesh.stl").exists()
    finally:
        calib.use(before)
        if env is None:
            os.environ.pop("TRIBUUTSCAN_CAMERA", None)
        else:
            os.environ["TRIBUUTSCAN_CAMERA"] = env
