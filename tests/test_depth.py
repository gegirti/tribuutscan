import numpy as np

from ferret.depth import decode_y10, encode_y10


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
