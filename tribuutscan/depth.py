"""Decode the depth function's packed 10- and 14-bit streams.

V4L2 offers only 'Y10 ' (bytesperline = width * 10 / 8): an MSB-first 10-bit bitstream, 4 pixels per
5 bytes. This is NOT the MIPI Y10P layout (which keeps the low bits in the 5th byte). The device also
has a 14-bit format that uvcvideo does not know; usbdepth.py streams it: the same MSB-first
bitstream, 4 pixels per 7 bytes, and Y10 is Y14 with the low 4 bits cut off. Nearer = larger, 0 =
invalid. decode_depth picks the format by bytesperline and returns values on the Y10 scale, so the
calibrated depth law works for both, and clears the rows the scanner garbled (garbled_rows).
"""
import numpy as np

# Garbled rows (seen 2026-11-01 in Y10 over V4L2 and Y14 over USB): now and then the scanner fills a run
# of 1-27 rows with values that are not depth, in frames of the right size; far more often when the
# host is slow to take the data (usbdepth.NURBS). Down each column such a run counts in steps of
# exactly 64 raw units (Y10 914, 850, 786, ...; Y14 in steps of 64 or 1024): 30-97% of the
# row-to-row differences in a garbled row are nonzero multiples of 64, against 0-3% in real rows
# (smooth surfaces change by less, edges by any amount). Left in, they become sheets in cloud and mesh.
# No false hits on the synthetic sets (0 of 614 frames); every hit looked at was garbage. Next to such
# runs the old V4L2 recording also had a few rows of another kind (a smooth ramp across the row,
# wrapping at 1024) that this test misses, hence the wide margin. Dropping the whole frame instead
# was measured worse: on the marker recording (3.3 fps) the 44 lost frames let tracking double the part.
GARBLE_STEP = 64  # raw units, both formats
GARBLE_SHARE, GARBLE_MIN = 0.15, 20  # a row is garbled when this share (and number) of its steps are such
GARBLE_MARGIN = 8  # rows cleared above and below each garbled row
garbled = {"frames": 0, "rows": 0}  # what decode_depth cleared (for the viewer's log)


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


def garbled_rows(raw):
    """Bool per row: rows the scanner filled with garbage (see GARBLE_STEP), except the first of each
    run. `raw` holds the raw integer values (Y10, or Y14 before y10_scale); judged on every 2nd
    column, over the pixels that have depth in both rows."""
    s = raw[:, ::2].astype(np.int32)
    ok = (s[1:] > 0) & (s[:-1] > 0)
    step = s[1:] - s[:-1]
    hits = (ok & (step != 0) & (step % GARBLE_STEP == 0)).sum(1)
    hit = np.zeros(len(raw), bool)
    hit[1:] = (hits >= GARBLE_MIN) & (hits >= GARBLE_SHARE * ok.sum(1))
    return hit


def decode_depth(buf, width, height, bytesperline, clean=True):
    """Decode either format, told apart by bytesperline (Y10: width * 10 / 8, Y14: width * 14 / 8).
    Garbled rows and GARBLE_MARGIN rows around them come back with no depth (0) unless clean is False."""
    y14 = bytesperline >= width * 14 // 8
    raw = unpack_y14(buf, width, height, bytesperline) if y14 else decode_y10(buf, width, height, bytesperline)
    rows = garbled_rows(raw) if clean else None
    if rows is not None and rows.any():
        rows = np.convolve(rows, np.ones(2 * GARBLE_MARGIN + 1), "same") > 0
        raw[rows] = 0
        garbled["frames"] += 1
        garbled["rows"] += int(rows.sum())
    return y10_scale(raw) if y14 else raw
