import tempfile
from pathlib import Path

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
        rec = fr.Recorder(fr.W * 10 // 8, root=tmp, streams=fr.STREAMS, compress_depth=False)  # the old layout
        for t in range(3):
            rec.write(bufs, {k: float(t) for k in fr.STREAMS})
        rec.close()
        assert sorted(p.name for p in rec.dir.glob("*.raw")) == ["color.raw", "depth.raw", "ir.raw"]
        sets = list(fr.read_recording(rec.dir))
    assert len(sets) == 3
    assert np.array_equal(sets[2]["depth"], depth)
    ir = np.frombuffer(bufs["ir"], np.uint8).reshape(fr.H, fr.W)
    assert np.array_equal(sets[0]["ir"], ir[:, ::-1])


def test_scan_recording_leaves_ir_out_and_packs_depth():
    rng = np.random.default_rng(4)
    depth = np.where(rng.random((fr.H, fr.W)) < 0.6, 0, rng.integers(1, 1024, size=(fr.H, fr.W))).astype(np.uint16)
    color = rng.integers(0, 256, fr.FRAME_BYTES["color"], dtype=np.uint8).tobytes()
    with tempfile.TemporaryDirectory() as tmp:
        rec = fr.Recorder(fr.W * 10 // 8, root=tmp)
        for t in range(3):
            rec.write({"color": color, "ir": None, "depth": encode_y10(depth)}, {"color": t + 0.01, "depth": float(t)})
        rec.close()
        assert sorted(p.name for p in rec.dir.iterdir()) == ["color.raw", "depth.zlib", "info.json", "stamps.csv"]
        assert rec.megabytes < 3 * (fr.FRAME_BYTES["color"] + fr.FRAME_BYTES["depth"]) / 1e6
        sets = list(fr.read_recording(rec.dir))
        assert len(sets) == 3 and sets[0]["ir"] is None
        assert np.array_equal(sets[2]["depth"], depth) and abs(sets[1]["skew_ms"] - 10) < 0.01
        with open(rec.dir / "depth.zlib", "ab") as f:  # a set cut off while writing
            f.write(b"\xff\xff\x00\x00partial")
        assert len(list(fr.read_recording(rec.dir))) == 3
        empty = fr.Recorder(fr.W * 10 // 8, root=Path(tmp) / "x")  # started and stopped at once
        empty.close()
        assert list(fr.read_recording(empty.dir)) == []


def test_recording_stops_when_the_disk_is_nearly_full():
    bufs = {"color": bytes(fr.FRAME_BYTES["color"]), "ir": None, "depth": bytes(fr.FRAME_BYTES["depth"])}
    floor = fr.Recorder.FREE_FLOOR_GB
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fr.Recorder.FREE_FLOOR_GB = 1e9
            try:
                fr.Recorder(fr.W * 10 // 8, root=tmp)
                raise AssertionError("started with too little free space")
            except fr.DiskFull:
                pass
            assert not list(Path(tmp).iterdir())  # nothing created
            fr.Recorder.FREE_FLOOR_GB = floor
            rec = fr.Recorder(fr.W * 10 // 8, root=tmp)
            for t in range(fr.Recorder.CHECK_EVERY):
                rec.write(bufs, {"color": 0.0, "depth": 0.0})
            fr.Recorder.FREE_FLOOR_GB = 1e9
            try:
                rec.write(bufs, {"color": 0.0, "depth": 0.0})  # the next check stops it
                raise AssertionError("kept writing")
            except fr.DiskFull:
                pass
            rec.close()
        finally:
            fr.Recorder.FREE_FLOOR_GB = floor
        assert len(list(fr.read_recording(rec.dir))) == fr.Recorder.CHECK_EVERY


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


def test_recording_carries_the_depth_mode():
    from tribuutscan import calib
    bufs = {"color": bytes(fr.FRAME_BYTES["color"]), "ir": bytes(fr.FRAME_BYTES["ir"]),
            "depth": encode_y10(np.zeros((fr.H, fr.W), np.uint16))}
    try:
        calib.use_mode(calib.LONG)
        with tempfile.TemporaryDirectory() as tmp:
            rec = fr.Recorder(fr.W * 10 // 8, root=tmp)
            rec.write(bufs, {k: 0.0 for k in fr.STREAMS})
            rec.close()
            calib.use_mode(calib.CLOSE)
            next(fr.read_recording(rec.dir))
            assert calib.CAMERA.mode == calib.LONG  # reading it switched the law
            (rec.dir / "info.json").write_text((rec.dir / "info.json").read_text().replace('"depth_mode"', '"x"'))
            next(fr.read_recording(rec.dir))
            assert calib.CAMERA.mode == calib.CLOSE  # not noted: an old recording, close range
    finally:
        calib.use_mode(calib.CLOSE)


def test_recording_keeps_imu_samples():
    rows = np.array([[10.5 + k / 200, 1_000_000 + 5000 * k, 0, -1, 0, 0.5, -0.2, 0.1, 30.25, k] for k in range(7)])
    with tempfile.TemporaryDirectory() as tmp:
        rec = fr.Recorder(fr.W * 10 // 8, root=tmp)
        rec.write_imu(rows[:3])
        rec.write_imu(rows[3:])
        rec.close()
        rec.write_imu(rows)  # after close: ignored
        imu = fr.read_imu(rec.dir)
        assert fr.read_imu(tmp) is None
    assert rec.imu_rows == 7 and len(imu["host_t"]) == 7
    assert np.allclose(imu["device_us"], rows[:, 1]) and np.allclose(imu["ay_g"], -1) and np.allclose(imu["temp_c"], 30.25)


def test_depth_mode_survives_a_failing_factory_read():
    from tribuutscan import calib, vendor
    entry = bytes(16) + b"Long_range_mode Default".ljust(32, b"\0") + bytes(4)
    noted, logs = [], []

    class FakeChannel:
        fail = False

        def __init__(self, dev=None, usb=None):
            pass

        def current_mode(self):
            return entry

        def factory_calibration(self):
            if FakeChannel.fail:
                raise vendor.VendorError("unsupported")
            return {"depth_calib": {}}

        def close(self):
            pass

    real_channel, real_note = vendor.Channel, calib.note_factory
    vendor.Channel, calib.note_factory = FakeChannel, lambda fac, mode: noted.append(mode)
    try:
        assert fr.depth_mode({"depth": "x"}, logs.append) == calib.LONG and noted == [calib.LONG]
        FakeChannel.fail = True
        assert fr.depth_mode({"depth": "x"}, logs.append) == calib.LONG and noted == [calib.LONG]
        assert "factory calibration not readable" in logs[-1]
    finally:
        vendor.Channel, calib.note_factory = real_channel, real_note
