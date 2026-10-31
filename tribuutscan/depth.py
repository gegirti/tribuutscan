"""Decode the depth function's packed 10- and 14-bit streams.

V4L2 offers only 'Y10 ' (bytesperline = width * 10 / 8): an MSB-first 10-bit bitstream, 4 pixels per
5 bytes. This is NOT the MIPI Y10P layout (which keeps the low bits in the 5th byte). The device also
has a 14-bit format that uvcvideo does not know; usbdepth.py streams it: the same MSB-first
bitstream, 4 pixels per 7 bytes, and Y10 is Y14 with the low 4 bits cut off. Nearer = larger, 0 =
invalid. decode_depth picks the format by bytesperline and returns values on the Y10 scale, so the
calibrated depth law works for both.
"""
import numpy as np


def decode_y10(buf, width, height, bytesperline):
    """Return a (height, width) uint16 array of raw depth values (0 = invalid)."""
    rows = np.frombuffer(buf, dtype=np.uint8, count=height * bytesperline).reshape(height, bytesperline)
    g = rows[:, : width // 4 * 5].reshape(height, width // 4, 5).astype(np.uint16)
    out = np.empty((height, width // 4, 4), dtype=np.uint16)
    out[..., 0] = (g[..., 0] << 2) | (g[..., 1] >> 6)
    out[..., 1] = ((g[..., 1] & 0x3F) << 4) | (g[..., 2] >> 4)
    out[..., 2] = ((g[..., 2] & 0x0F) << 6) | (g[..., 3] >> 2)
    out[..., 3] = ((g[..., 3] & 0x03) << 8) | g[..., 4]
    return out.reshape(height, width)


def encode_y10(values):
    """Inverse of decode_y10 (for tests)."""
    h, w = values.shape
    v = values.astype(np.uint16).reshape(h, w // 4, 4)
    out = np.empty((h, w // 4, 5), dtype=np.uint8)
    out[..., 0] = v[..., 0] >> 2
    out[..., 1] = ((v[..., 0] & 0x3) << 6) | (v[..., 1] >> 4)
    out[..., 2] = ((v[..., 1] & 0xF) << 4) | (v[..., 2] >> 6)
    out[..., 3] = ((v[..., 2] & 0x3F) << 2) | (v[..., 3] >> 8)
    out[..., 4] = v[..., 3] & 0xFF
    return out.reshape(h, -1).tobytes()


def unpack_y14(buf, width, height, bytesperline):
    """Return a (height, width) uint16 array of raw 14-bit values (0 = invalid)."""
    rows = np.frombuffer(buf, dtype=np.uint8, count=height * bytesperline).reshape(height, bytesperline)
    g = rows[:, : width // 4 * 7].reshape(height, width // 4, 7).astype(np.uint16)
    out = np.empty((height, width // 4, 4), dtype=np.uint16)
    out[..., 0] = (g[..., 0] << 6) | (g[..., 1] >> 2)
    out[..., 1] = ((g[..., 1] & 0x03) << 12) | (g[..., 2] << 4) | (g[..., 3] >> 4)
    out[..., 2] = ((g[..., 3] & 0x0F) << 10) | (g[..., 4] << 2) | (g[..., 5] >> 6)
    out[..., 3] = ((g[..., 5] & 0x3F) << 8) | g[..., 6]
    return out.reshape(height, width)


def encode_y14(values):
    """Inverse of unpack_y14 (for tests and synthetic data)."""
    h, w = values.shape
    v = values.astype(np.uint16).reshape(h, w // 4, 4)
    out = np.empty((h, w // 4, 7), dtype=np.uint8)
    out[..., 0] = v[..., 0] >> 6
    out[..., 1] = ((v[..., 0] & 0x3F) << 2) | (v[..., 1] >> 12)
    out[..., 2] = (v[..., 1] >> 4) & 0xFF
    out[..., 3] = ((v[..., 1] & 0xF) << 4) | (v[..., 2] >> 10)
    out[..., 4] = (v[..., 2] >> 2) & 0xFF
    out[..., 5] = ((v[..., 2] & 0x3) << 6) | (v[..., 3] >> 8)
    out[..., 6] = v[..., 3] & 0xFF
    return out.reshape(h, -1).tobytes()


def y10_scale(raw14):
    """14-bit values on the Y10 scale (float32). Y10 = floor(Y14 / 16), so the middle of each Y10 step
    is (Y14 - 8) / 16 (the probe fitted Y14 = 15.996 Y10 + 7.8); values that were 0 in Y10 stay 0."""
    return np.where(raw14 >= 16, (raw14.astype(np.float32) - 8) / 16, np.float32(0))


def decode_y14(buf, width, height, bytesperline):
    """Return a (height, width) float32 array of depth on the Y10 scale (0 = invalid)."""
    return y10_scale(unpack_y14(buf, width, height, bytesperline))


def decode_depth(buf, width, height, bytesperline):
    """Decode either format, told apart by bytesperline (Y10: width * 10 / 8, Y14: width * 14 / 8)."""
    if bytesperline >= width * 14 // 8:
        return decode_y14(buf, width, height, bytesperline)
    return decode_y10(buf, width, height, bytesperline)
