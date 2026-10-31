import ctypes
import struct

from tribuutscan import usbdepth


def payload(info, data=b"", pts=None):
    """A UVC payload: 12-byte header (PTS when given) + data."""
    if pts is not None:
        info |= 0x04
    return bytes([12, info]) + struct.pack("<I", pts or 0) + bytes(6) + data


def test_urb_layout_matches_the_kernel():
    assert ctypes.sizeof(usbdepth.Urb) == 56
    assert usbdepth.Urb.buffer.offset == 16 and usbdepth.Urb.usercontext.offset == 48


def test_assembler_joins_payloads_until_eof():
    a = usbdepth.FrameAssembler(6, clock=lambda pts, t: t)
    assert a.feed(payload(0, b"abc"), 1.0) == []
    assert a.feed(payload(0x02, b"def"), 1.1) == [(b"abcdef", 1.0, 0)]  # stamped at its first payload
    assert a.feed(payload(0x01 | 0x02, b"ghijkl"), 2.0) == [(b"ghijkl", 2.0, 1)]


def test_assembler_ends_a_frame_on_fid_toggle_and_drops_bad_ones():
    a = usbdepth.FrameAssembler(4, clock=lambda pts, t: t)
    a.feed(payload(0, b"abcd"), 1.0)  # EOF lost
    assert a.feed(payload(0x01, b"ef"), 2.0) == [(b"abcd", 1.0, 0)]
    assert a.feed(payload(0x01 | 0x02, b"g"), 2.1) == []  # 3 bytes: truncated
    a.feed(payload(0x40, b"hi"), 3.0)  # error bit
    assert a.feed(payload(0x02, b"jk"), 3.1) == []
    assert a.bad == 2
    assert a.feed(payload(0x01 | 0x02), 4.0) == [] and a.bad == 2  # header only: not a frame


def test_assembler_uses_the_pts_clock():
    a = usbdepth.FrameAssembler(2)
    out = []
    for k in range(5):  # PTS every 33 ms; arrival 20-25 ms later
        pts = 1_000_000 + 33_333 * k
        out += a.feed(payload(0x02 | (k & 1), b"xy", pts=pts), pts / 1e6 + 0.020 + 0.005 * (k % 2))
    assert [round(t - (1 + 0.033333 * i), 6) for i, (_, t, _) in enumerate(out)] == [0.02] * 5


def test_pts_clock_survives_a_32_bit_wrap():
    clock = usbdepth.PtsClock()
    t1 = clock((1 << 32) - 10_000, 100.0)
    t2 = clock(23_333, 100.040)  # 33.3 ms later, after the wrap (arrived a little later)
    assert abs((t2 - t1) - 0.033333) < 1e-6


def test_pts_clock_restarts_after_a_jump():
    clock = usbdepth.PtsClock()
    clock(5_000_000, 100.0)
    assert abs(clock(1_000, 200.0) - 200.0) < 1e-9  # device clock restarted: old offset forgotten
