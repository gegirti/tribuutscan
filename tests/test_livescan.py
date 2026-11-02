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
