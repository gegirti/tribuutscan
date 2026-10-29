"""Minimal V4L2 capture (ctypes-free: raw ioctl + mmap) for the Ferret Pro's UVC nodes.

ffmpeg and GStreamer cannot negotiate the depth node's packed 10-bit format, so we talk to
V4L2 directly. Struct offsets assume a 64-bit kernel (the v4l2_format union starts at byte 8).
"""
import fcntl
import mmap
import os
import select
import struct
from pathlib import Path

VIDIOC_S_FMT = 0xC0D05605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

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
        """Block for the next frame and return it as bytes, or None on timeout."""
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        b = self._buffer()
        fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
        index = struct.unpack_from("<I", b, 0)[0]
        used = struct.unpack_from("<I", b, 8)[0]
        data = bytes(self._bufs[index][:used])
        fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
        return data

    def frames(self):
        while self._streaming:
            data = self.read()
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
