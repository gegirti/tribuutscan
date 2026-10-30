import struct

from tribuutscan import v4l2


def test_buffer_time_reads_kernel_timestamp_and_sequence():
    b = bytearray(88)
    struct.pack_into("<qq", b, 24, 1234, 567890)  # struct timeval: seconds, microseconds
    struct.pack_into("<I", b, 56, 42)
    t, seq = v4l2.buffer_time(b)
    assert abs(t - 1234.56789) < 1e-9 and seq == 42
