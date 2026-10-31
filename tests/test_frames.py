import tempfile

import numpy as np

from tribuutscan import frames as fr
from tribuutscan.depth import encode_y10


def test_record_roundtrip():
    rng = np.random.default_rng(2)
    depth = rng.integers(0, 1024, size=(fr.H, fr.W), dtype=np.uint16)
    bufs = {
        "color": bytes(fr.FRAME_BYTES["color"]),
        "ir": rng.integers(0, 256, fr.FRAME_BYTES["ir"], dtype=np.uint8).tobytes(),
        "depth": encode_y10(depth),
    }
    with tempfile.TemporaryDirectory() as tmp:
        rec = fr.Recorder(fr.W * 10 // 8, root=tmp)
        for t in range(3):
            rec.write(bufs, {k: float(t) for k in fr.STREAMS})
        rec.close()
        sets = list(fr.read_recording(rec.dir))
    assert len(sets) == 3
    assert np.array_equal(sets[2]["depth"], depth)
    ir = np.frombuffer(bufs["ir"], np.uint8).reshape(fr.H, fr.W)
    assert np.array_equal(sets[0]["ir"], ir[:, ::-1])


def test_record_roundtrip_y14():
    from tribuutscan.depth import encode_y14
    rng = np.random.default_rng(3)
    y14 = rng.integers(16, 1 << 14, size=(fr.H, fr.W), dtype=np.uint16)
    bufs = {"color": bytes(fr.FRAME_BYTES["color"]), "ir": bytes(fr.FRAME_BYTES["ir"]), "depth": encode_y14(y14)}
    with tempfile.TemporaryDirectory() as tmp:
        rec = fr.Recorder(fr.W * 14 // 8, root=tmp)
        for t in range(2):
            rec.write(bufs, {k: float(t) for k in fr.STREAMS})
        rec.close()
        sets = list(fr.read_recording(rec.dir))
    assert len(sets) == 2
    assert np.array_equal(sets[1]["depth"], (y14.astype(np.float32) - 8) / 16)
