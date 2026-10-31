import struct

import numpy as np

from tribuutscan import imu


def packet(groups):
    """A report as the scanner sends it: header (report id, rate, group length, count, reserved) + groups."""
    head = struct.pack("<BBBBI", 1, 7, imu.GROUP.size, len(groups), 0)
    return (head + b"".join(imu.GROUP.pack(*g) for g in groups)).ljust(imu.PACKET, b"\0")


def test_parse_scales_accel_gyro_and_time():
    p = packet([(0, 0, -8192, 0, 32767 // 2, 0, 0, 0, 1_000_000), (1, 8192, 0, 0, 0, -3277, 0, 1325, 1_004_992)])
    s = imu.parse(p, accel_g=4, gyro_dps=1000)
    assert s.shape == (2, 9)
    assert np.allclose(s[:, 0], [1_000_000, 1_004_992])
    assert np.allclose(s[0, 1:4], [0, -1, 0])  # -8192 of +-4 g = -1 g
    assert abs(s[0, 4] - 500) < 0.1 and abs(s[1, 5] + 100) < 0.1  # deg/s
    assert abs(s[1, 7] - (1325 / 132.48 + 25)) < 1e-6 and list(s[:, 8]) == [0, 1]


def test_parse_drops_a_bad_header():
    p = bytearray(packet([(0,) * 8 + (0,)]))
    p[2] = 20  # group length that does not match the layout
    assert imu.parse(bytes(p)).shape == (0, 9)
    assert imu.parse(b"\x01\x07").shape == (0, 9)


def test_state_reads_tilt_from_gravity():
    class Reader:
        error = None
    stream = imu.ImuStream(Reader(), 4, 1000)
    for k in range(20):  # upright, then the same pitched 30 degrees: gravity in the y-z plane
        stream.history.append(np.array([k, k, 0, -np.cos(np.radians(30)), np.sin(np.radians(30)), 0.1, 0, 0, 30, k]))
    st = stream.state()
    assert abs(st["pitch"] - 30) < 1e-6 and abs(st["roll"]) < 1e-6 and st["still"] and abs(st["g"] - 1) < 1e-9
