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


def garbled_frame():
    """A sloped surface (rows step by 1 raw unit) with a 6-row run of the scanner's garbage in it:
    down each column the values count in steps of 64 (the column keeps its low 6 bits)."""
    rng = np.random.default_rng(4)
    h, w = 40, 160
    yy, xx = np.mgrid[0:h, 0:w]
    clean = (300 + yy + xx // 4).astype(np.uint16)
    bad = clean.copy()
    low, high = rng.integers(0, 64, w), rng.integers(0, 16, w)
    for y in range(20, 26):
        high = (high - rng.integers(0, 2, w)) % 16
        bad[y] = 64 * high + low
    return clean, bad


def test_garbled_rows_are_cleared():
    from tribuutscan import depth
    clean, bad = garbled_frame()
    h, w = clean.shape
    assert not depth.garbled_rows(clean).any()
    assert depth.garbled_rows(bad)[21:26].all() and not depth.garbled_rows(bad)[:20].any()
    before = depth.garbled["frames"]
    assert np.array_equal(depth.decode_depth(depth.encode_y10(clean), w, h, w * 10 // 8), clean)
    got = depth.decode_depth(depth.encode_y10(bad), w, h, w * 10 // 8)
    m = depth.GARBLE_MARGIN
    assert not got[20 - m + 1:26 + m].any()  # the run (its first row found through the margin) and around it
    assert np.array_equal(got[:20 - m], clean[:20 - m]) and np.array_equal(got[26 + m + 1:], clean[26 + m + 1:])
    assert np.array_equal(depth.decode_depth(depth.encode_y10(bad), w, h, w * 10 // 8, clean=False), bad)
    y14 = lambda v: depth.encode_y14(v * 16 + 8)  # the same frames in 14 bits
    assert depth.decode_depth(y14(clean), w, h, w * 14 // 8).all()
    assert not depth.decode_depth(y14(bad), w, h, w * 14 // 8)[20:26].any()
    assert depth.garbled["frames"] == before + 2
