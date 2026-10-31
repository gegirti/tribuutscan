import numpy as np

from tribuutscan.depth import decode_y10, encode_y10


def test_roundtrip():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 1024, size=(8, 16), dtype=np.uint16)
    buf = encode_y10(values)
    assert len(buf) == 8 * 16 * 10 // 8
    assert np.array_equal(decode_y10(buf, 16, 8, 16 * 10 // 8), values)


def test_matches_msb_first_bitstream():
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 256, size=(4, 20), dtype=np.uint8)
    bits = np.unpackbits(raw, axis=1).reshape(4, 16, 10).astype(np.uint16)
    expected = (bits * (1 << np.arange(9, -1, -1))).sum(-1)
    assert np.array_equal(decode_y10(raw.tobytes(), 16, 4, 20), expected)


def test_y14_roundtrip():
    from tribuutscan.depth import encode_y14, unpack_y14
    rng = np.random.default_rng(2)
    values = rng.integers(0, 1 << 14, size=(8, 16), dtype=np.uint16)
    buf = encode_y14(values)
    assert len(buf) == 8 * 16 * 14 // 8
    assert np.array_equal(unpack_y14(buf, 16, 8, 16 * 14 // 8), values)


def test_y14_matches_msb_first_bitstream():
    from tribuutscan.depth import unpack_y14
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, size=(4, 28), dtype=np.uint8)
    bits = np.unpackbits(raw, axis=1).reshape(4, 16, 14).astype(np.uint32)
    expected = (bits * (1 << np.arange(13, -1, -1))).sum(-1)
    assert np.array_equal(unpack_y14(raw.tobytes(), 16, 4, 28), expected)


def test_decode_depth_puts_y14_on_the_y10_scale():
    from tribuutscan.depth import decode_depth, encode_y14
    y10 = np.array([[0, 1, 280, 1023]], np.uint16)
    y14 = np.array([[15, 16, 280 * 16 + 5, 1023 * 16 + 15]], np.uint16)  # each inside its Y10 step
    got = decode_depth(encode_y14(y14), 4, 1, 7)
    assert got.dtype == np.float32
    assert got[0, 0] == 0 and np.all(np.abs(got[0, 1:] - y10[0, 1:]) <= 0.5)
    assert np.array_equal(decode_depth(encode_y10(y10), 4, 1, 5), y10)  # Y10 untouched
