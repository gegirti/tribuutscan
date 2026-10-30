"""Minimal V4L2 capture (ctypes-free: raw ioctl + mmap) for the Ferret Pro's UVC nodes.

ffmpeg and GStreamer cannot negotiate the depth node's packed 10-bit format, so we talk to
V4L2 directly. Struct offsets assume a 64-bit kernel (the v4l2_format union starts at byte 8).
"""
import fcntl
import mmap
import os
import select
import struct
import time
from pathlib import Path

VIDIOC_S_FMT = 0xC0D05605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C
VIDIOC_QUERYCTRL = 0xC0445624
CTRL_FLAG_NEXT = 0x80000000
CTRL_TYPE_CLASS = 6

# Standard control IDs seen on the Ferret Pro (all three nodes have gain + exposure).
CID_BRIGHTNESS = 0x00980900
CID_GAIN = 0x00980913
CID_EXPOSURE_AUTO = 0x009A0901  # menu: 0 auto, 2 shutter priority (manual exposure time)
CID_EXPOSURE = 0x009A0902  # exposure time, units of 100 us per the V4L2 spec

BUF_TYPE_CAPTURE = 1
MEMORY_MMAP = 1

PRODUCT_PREFIX = "CR-Scan Ferret"


def discover():
    """Return {'depth': path, 'ir': path, 'color': path} for the connected scanner.

    The scanner exposes three UVC functions (USB interfaces 1.0, 1.2, 1.4). Empirically they
    are depth, IR and colour, in that order. Odd /dev/videoN nodes are metadata and skipped.
    """
    found = {}
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        if not (node / "name").read_text().startswith(PRODUCT_PREFIX):
            continue
        if (node / "index").read_text().strip() != "0":
            continue
        iface = (node / "device").resolve().name  # e.g. 4-2:1.0
        found[iface.rsplit(".", 1)[-1]] = f"/dev/{node.name}"
    order = ["0", "2", "4"]
    if not all(k in found for k in order):
        raise RuntimeError(f"scanner not found or unexpected layout: {found}")
    return dict(zip(("depth", "ir", "color"), (found[k] for k in order)))


def controls(dev):
    """Return {name: dict(id, min, max, step, default, value)} for a node's integer/menu controls."""
    out = {}
    fd = os.open(dev, os.O_RDWR)
    try:
        cid = CTRL_FLAG_NEXT
        while True:
            q = bytearray(struct.pack("<II32siiiiI8x", cid, 0, b"", 0, 0, 0, 0, 0))
            try:
                fcntl.ioctl(fd, VIDIOC_QUERYCTRL, q)
            except OSError:
                break
            i, typ, name, lo, hi, step, default, _ = struct.unpack("<II32siiiiI8x", q)
            if typ != CTRL_TYPE_CLASS:
                out[name.rstrip(b"\0").decode()] = dict(
                    id=i, min=lo, max=hi, step=step, default=default, value=_get(fd, i))
            cid = i | CTRL_FLAG_NEXT
    finally:
        os.close(fd)
    return out


def _get(fd, cid):
    c = bytearray(struct.pack("<Ii", cid, 0))
    fcntl.ioctl(fd, VIDIOC_G_CTRL, c)
    return struct.unpack("<Ii", c)[1]


def get_control(dev, cid):
    fd = os.open(dev, os.O_RDWR)
    try:
        return _get(fd, cid)
    finally:
        os.close(fd)


def set_control(dev, cid, value):
    """Set one control; works while another handle is streaming from the same node."""
    fd = os.open(dev, os.O_RDWR)
    try:
        fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("<Ii", cid, int(value)))
    finally:
        os.close(fd)


def buffer_time(b):
    """(kernel timestamp in seconds, sequence number) of a dequeued v4l2_buffer (64-bit layout:
    struct timeval at byte 24, sequence at byte 56)."""
    sec, usec = struct.unpack_from("<qq", b, 24)
    return sec + usec / 1e6, struct.unpack_from("<I", b, 56)[0]


class Capture:
    """One streaming capture node. Iterate `frames()` for raw frame bytes."""

    def __init__(self, dev, fourcc, width, height, nbufs=4):
        self.fd = os.open(dev, os.O_RDWR)
        fmt = bytearray(208)
        struct.pack_into("<I", fmt, 0, BUF_TYPE_CAPTURE)
        struct.pack_into("<III", fmt, 8, width, height, struct.unpack("<I", fourcc.ljust(4).encode())[0])
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        self.width, self.height, fcc, _, self.bytesperline, self.sizeimage = struct.unpack_from("<IIIIII", fmt, 8)
        self.fourcc = struct.pack("<I", fcc).decode()

        req = bytearray(20)
        struct.pack_into("<IIII", req, 0, nbufs, BUF_TYPE_CAPTURE, MEMORY_MMAP, 0)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
        self._bufs = []
        for i in range(struct.unpack_from("<I", req, 0)[0]):
            b = self._buffer(i)
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, b)
            length = struct.unpack_from("<I", b, 72)[0]
            offset = struct.unpack_from("<I", b, 64)[0]
            self._bufs.append(mmap.mmap(self.fd, length, mmap.MAP_SHARED, mmap.PROT_READ, offset=offset))
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, struct.pack("<I", BUF_TYPE_CAPTURE))
        self._streaming = True

    @staticmethod
    def _buffer(index=0):
        b = bytearray(88)
        struct.pack_into("<II", b, 0, index, BUF_TYPE_CAPTURE)
        struct.pack_into("<I", b, 60, MEMORY_MMAP)
        return b

    def read(self, timeout=5.0):
        """Block for the next frame and return it as bytes, or None on timeout.

        Also sets `last_time` (kernel capture time, seconds on the CLOCK_MONOTONIC scale of
        time.monotonic(); time.monotonic() at dequeue if the driver gave none) and `last_sequence`.
        """
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        b = self._buffer()
        fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
        index = struct.unpack_from("<I", b, 0)[0]
        used = struct.unpack_from("<I", b, 8)[0]
        data = bytes(self._bufs[index][:used])
        self.last_time, self.last_sequence = buffer_time(b)  # before QBUF: re-queueing clears these fields
        fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
        if not self.last_time:
            self.last_time = time.monotonic()
        return data

    def ready(self):
        """True if another frame is already waiting (reading it now would not block)."""
        return bool(select.select([self.fd], [], [], 0)[0])

    def frames(self):
        while self._streaming:
            try:
                data = self.read()
            except OSError:
                if self._streaming:
                    raise
                return  # closed from another thread mid-read
            if data is not None:
                yield data

    def close(self):
        if self._streaming:
            self._streaming = False
            fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, struct.pack("<I", BUF_TYPE_CAPTURE))
            for m in self._bufs:
                m.close()
            os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
