"""Stream the depth function over USB ourselves, for the 14-bit format uvcvideo drops.

The depth function offers Y14 (format 1), Y10 (2) and Y8 (3); uvcvideo does not know the Y14 GUID
("Unknown video format 20343159-..."), so V4L2 never shows it. This module unbinds uvcvideo from the
depth function only (USB interfaces 0 and 1; IR and colour stay on V4L2), negotiates the format with
UVC probe/commit and reads the bulk endpoint through usbfs: raw ioctls, stdlib only (ctypes holds the
URB memory). It needs write access to /dev/bus/usb/BBB/DDD (udev/70-tribuutscan.rules).

DepthCapture has the same API as v4l2.Capture (fd for select, read, ready, frames, last_time, ...)
plus the IR exposure and gain controls, which live on the depth function. Run
`python3 -m tribuutscan.usbdepth [seconds] [--all] [--y10]` for a throughput check.
"""
import collections
import ctypes
import errno
import fcntl
import os
import select
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import v4l2

VID, PID = "2bc5", "069c"
VC, VS, EP = 0, 1, 0x81  # depth function: control and streaming interface, bulk IN endpoint
W, H, FRAME_INDEX = 1280, 800, 3  # frame index 3 is 1280x800 in both formats
FORMATS = {"Y14": (1, 14), "Y10": (2, 10)}  # name -> (UVC format index, bits per pixel)
INTERVAL = 333333  # 100 ns units: 30 fps

# usbfs ioctls (linux/usbdevice_fs.h, 64-bit layouts)
USBDEVFS_CONTROL = 0xC0185500
USBDEVFS_SUBMITURB = 0x8038550A
USBDEVFS_DISCARDURB = 0x0000550B
USBDEVFS_REAPURBNDELAY = 0x4008550D
USBDEVFS_RELEASEINTERFACE = 0x80045510
USBDEVFS_IOCTL = 0xC0105512
USBDEVFS_CLEAR_HALT = 0x80045515
USBDEVFS_CONNECT = 0x00005517
USBDEVFS_DISCONNECT_CLAIM = 0x8108551B
CLAIM_EXCEPT_USBFS = struct.pack("<I", 0x02)  # DISCONNECT_CLAIM_EXCEPT_DRIVER "usbfs": take an interface
# from uvcvideo/usbhid, but fail with EBUSY instead of taking it from another program reading it over usbfs


def claim(fd, interface):
    """Unbind the kernel driver from `interface` and claim it; EBUSY if another program has it."""
    fcntl.ioctl(fd, USBDEVFS_DISCONNECT_CLAIM, struct.pack("<I", interface) + CLAIM_EXCEPT_USBFS + b"usbfs".ljust(256, b"\0"))
URB_TYPE_BULK = 3

# UVC requests and the two controls the viewer uses (unit IDs from the depth function's descriptors)
SET_CUR, GET_CUR, GET_MIN, GET_MAX, GET_RES, GET_DEF = 0x01, 0x81, 0x82, 0x83, 0x84, 0x87
PROBE, COMMIT = 0x01, 0x02
CONTROLS = {  # V4L2 id -> (name, unit, selector, size); values are the same as V4L2's
    v4l2.CID_EXPOSURE: ("Exposure Time, Absolute", 1, 0x04, 4),  # camera terminal: 100 us units
    v4l2.CID_GAIN: ("Gain", 3, 0x04, 2),  # processing unit
}


class Urb(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ubyte), ("endpoint", ctypes.c_ubyte), ("status", ctypes.c_int),
                ("flags", ctypes.c_uint), ("buffer", ctypes.c_void_p), ("buffer_length", ctypes.c_int),
                ("actual_length", ctypes.c_int), ("start_frame", ctypes.c_int),
                ("number_of_packets", ctypes.c_int), ("error_count", ctypes.c_int), ("signr", ctypes.c_uint),
                ("usercontext", ctypes.c_void_p)]


def find():
    """Path of the scanner's usbfs node (/dev/bus/usb/BBB/DDD), or None."""
    for d in Path("/sys/bus/usb/devices").glob("*"):
        try:
            if (d / "idVendor").read_text().strip() == VID and (d / "idProduct").read_text().strip() == PID:
                bus, dev = int((d / "busnum").read_text()), int((d / "devnum").read_text())
                return f"/dev/bus/usb/{bus:03d}/{dev:03d}"
        except OSError:
            continue
    return None


def available():
    """True if the scanner is connected and we may open it (the udev rule is installed)."""
    path = find()
    return path is not None and os.access(path, os.W_OK)


def address(buf):
    """Memory address of a bytearray (it must not be resized while the kernel uses it)."""
    return ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))


_libc = ctypes.CDLL(None, use_errno=True)
_libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]


def ioctl_ptr(fd, request, pointer):
    """ioctl whose argument is a pointer the kernel keeps (an URB): fcntl.ioctl takes only a C int
    or a buffer it may copy. ctypes releases the GIL during the call."""
    if _libc.ioctl(fd, request, pointer) < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))


class PtsClock:
    """Map the payload PTS (device clock, 32 bits) to time.monotonic(): device time plus the smallest
    (arrival - device time) of the last `window` frames, i.e. the least-delayed frame sets the offset.
    A jump of more than `reset_s` (device clock restarted) starts the window afresh."""

    def __init__(self, hz=1_000_000, window=90, reset_s=0.5):
        self.hz, self.reset_s = hz, reset_s
        self.offsets = collections.deque(maxlen=window)
        self.last, self.wraps = None, 0

    def __call__(self, pts, arrival):
        if self.last is not None and pts < self.last and self.last - pts > 1 << 31:
            self.wraps += 1
        self.last = pts
        device = (pts + (self.wraps << 32)) / self.hz
        offset = arrival - device
        if self.offsets and abs(offset - min(self.offsets)) > self.reset_s:
            self.offsets.clear()
        self.offsets.append(offset)
        return device + min(self.offsets)


class FrameAssembler:
    """Join UVC payloads (header + data) into frames: a frame ends on the EOF bit or when the FID bit
    toggles. Frames with the error bit, or not exactly `size` bytes, are counted in `bad` and dropped.
    feed() returns the finished frames as (bytes, time, sequence)."""

    def __init__(self, size, clock=None):
        self.size, self.clock = size, clock or PtsClock()
        self.buf, self.fid, self.started, self.error = bytearray(), None, False, False
        self.t, self.pts, self.sequence, self.bad = 0.0, None, 0, 0
        self.header_len = None

    def feed(self, payload, t):
        if len(payload) < 2 or not 2 <= payload[0] <= len(payload):
            return []
        hl, info = payload[0], payload[1]
        self.header_len = hl
        out = []
        if self.started and self.fid is not None and (info & 1) != self.fid:
            out += self._finish()  # EOF missed
        self.fid = info & 1
        if not self.started:
            self.started, self.t = True, t  # arrival of a frame's first payload
            self.pts = struct.unpack_from("<I", payload, 2)[0] if info & 0x04 and hl >= 6 else None
        self.error |= bool(info & 0x40)
        self.buf += payload[hl:]
        if info & 0x02:
            out += self._finish()
        return out

    def _finish(self):
        frame, ok = bytes(self.buf), len(self.buf) == self.size and not self.error
        t = self.clock(self.pts, self.t) if self.pts is not None else self.t
        self.buf, self.started, self.error = bytearray(), False, False
        if not frame:
            return []  # header-only payload between frames
        self.sequence += 1
        if not ok:
            self.bad += 1
            return []
        return [(frame, t, self.sequence - 1)]


class DepthCapture:
    """The depth function streamed over usbfs. Same reading API as v4l2.Capture."""

    def __init__(self, fmt="Y14", nbufs=4, nurbs=8, payload=None):
        path = find()
        if path is None:
            raise OSError(errno.ENODEV, "scanner not found")
        self.usb = os.open(path, os.O_RDWR)
        self._claimed = []
        try:
            for i in (VC, VS):  # unbind uvcvideo (if bound) and claim, in one step
                claim(self.usb, i)
                self._claimed.append(i)
            index, bits = FORMATS[fmt]
            self.width, self.height, self.fourcc = W, H, fmt.ljust(4)
            self.bytesperline = W * bits // 8
            self.sizeimage = self.bytesperline * H
            self.payload = self._negotiate(index, payload)
        except BaseException:
            self._release()
            raise
        self.assembler = FrameAssembler(self.sizeimage)
        self.last_time, self.last_sequence, self.dropped, self.error = 0.0, 0, 0, None
        self._ready = collections.deque(maxlen=nbufs)
        self._lock = threading.Lock()
        self.fd, self._wake = os.pipe()
        self._urbs, self._pending = {}, set()
        for _ in range(nurbs):
            buf = bytearray(self.payload)
            urb = Urb(type=URB_TYPE_BULK, endpoint=EP, buffer=address(buf), buffer_length=len(buf))
            self._urbs[ctypes.addressof(urb)] = (urb, buf)
        self._streaming = True
        for key in self._urbs:
            self._submit(key)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def control(self, request_type, request, value, index, data_or_len, timeout_ms=1000):
        """One control transfer on endpoint 0; data_or_len is the data to send or the length to read."""
        buf = bytearray(data_or_len)
        ct = bytearray(struct.pack("<BBHHHI4xQ", request_type, request, value, index, len(buf), timeout_ms,
                                   address(buf) if buf else 0))
        n = fcntl.ioctl(self.usb, USBDEVFS_CONTROL, ct)  # mutable buffer: returns the byte count
        return bytes(buf[:n])

    def _submit(self, key):
        urb = self._urbs[key][0]
        urb.status = urb.actual_length = 0
        ioctl_ptr(self.usb, USBDEVFS_SUBMITURB, key)
        self._pending.add(key)

    def _reap(self, ptr):
        """One completed URB (its key) or None; raises OSError when the device is gone."""
        try:
            fcntl.ioctl(self.usb, USBDEVFS_REAPURBNDELAY, ptr)
        except BlockingIOError:
            return None
        key = struct.unpack("<Q", ptr)[0]
        self._pending.discard(key)
        return key

    def _negotiate(self, index, payload):
        """UVC probe/commit; returns the payload transfer size the device grants (one URB each)."""
        probe = bytearray(34)  # UVC 1.1 streaming control
        struct.pack_into("<HBBI", probe, 0, 1, index, FRAME_INDEX, INTERVAL)
        if payload:
            struct.pack_into("<I", probe, 22, payload)
        self.control(0x21, SET_CUR, PROBE << 8, VS, probe)
        got = self.control(0xA1, GET_CUR, PROBE << 8, VS, 34)
        self.control(0x21, SET_CUR, COMMIT << 8, VS, got)
        self.probe = got
        return struct.unpack_from("<I", got, 22)[0]

    def _run(self):
        ptr = bytearray(8)
        try:
            while self._streaming:
                if not select.select([], [self.usb], [], 0.2)[1]:
                    continue
                while (key := self._reap(ptr)) is not None:
                    urb, buf = self._urbs[key]
                    if urb.status == 0 and urb.actual_length:
                        for frame in self.assembler.feed(memoryview(buf)[:urb.actual_length], time.monotonic()):
                            self._push(frame)
                    if self._streaming:
                        self._submit(key)
        except OSError as e:  # device gone (e.g. its own USB reset)
            if self._streaming:
                self.error = e
                os.write(self._wake, b"!")

    def _push(self, frame):
        with self._lock:
            if len(self._ready) == self._ready.maxlen:
                self._ready.popleft()  # reader is behind: drop the oldest, like a full V4L2 queue
                self.dropped += 1
            else:
                os.write(self._wake, b"\0")
            self._ready.append(frame)

    def read(self, timeout=5.0):
        """Block for the next frame and return it as bytes, or None on timeout. Sets `last_time`
        (time.monotonic() scale, from the payload PTS) and `last_sequence`."""
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        if os.read(self.fd, 1) == b"!":
            raise self.error
        with self._lock:
            data, self.last_time, self.last_sequence = self._ready.popleft()
        return data

    def ready(self):
        return bool(select.select([self.fd], [], [], 0)[0])

    def frames(self):
        while self._streaming:
            try:
                data = self.read()
            except OSError:
                if self._streaming:
                    raise
                return
            if data is not None:
                yield data

    # IR controls (on the depth function); same values as v4l2.controls / get_control / set_control
    def controls(self):
        out = {}
        for cid, (name, unit, sel, size) in CONTROLS.items():
            get = lambda req: int.from_bytes(self.control(0xA1, req, sel << 8, unit << 8 | VC, size), "little")
            out[name] = dict(id=cid, min=get(GET_MIN), max=get(GET_MAX), step=get(GET_RES),
                             default=get(GET_DEF), value=get(GET_CUR))
        return out

    def get_control(self, cid):
        _, unit, sel, size = CONTROLS[cid]
        return int.from_bytes(self.control(0xA1, GET_CUR, sel << 8, unit << 8 | VC, size), "little")

    def set_control(self, cid, value):
        _, unit, sel, size = CONTROLS[cid]
        self.control(0x21, SET_CUR, sel << 8, unit << 8 | VC, int(value).to_bytes(size, "little"))

    def _release(self):
        for i in self._claimed:
            try:
                fcntl.ioctl(self.usb, USBDEVFS_RELEASEINTERFACE, struct.pack("<I", i))
            except OSError:
                pass
        for i in self._claimed:  # hand the depth function back to uvcvideo
            try:
                fcntl.ioctl(self.usb, USBDEVFS_IOCTL, struct.pack("<iiQ", i, USBDEVFS_CONNECT, 0))
            except OSError:
                pass
        self._claimed = []
        os.close(self.usb)

    def close(self):
        if not self._streaming:
            return
        self._streaming = False
        self._thread.join(timeout=2)  # it stops resubmitting
        for key in list(self._pending):
            try:
                ioctl_ptr(self.usb, USBDEVFS_DISCARDURB, key)
            except OSError:
                pass
        ptr, end = bytearray(8), time.monotonic() + 1
        while self._pending and time.monotonic() < end:  # the kernel must hand every URB back
            try:
                if self._reap(ptr) is None:
                    select.select([], [self.usb], [], 0.05)
            except OSError:
                break
        try:
            fcntl.ioctl(self.usb, USBDEVFS_CLEAR_HALT, struct.pack("<I", EP))  # stops streaming (as uvcvideo does)
        except OSError:
            pass
        self._release()
        os.close(self.fd)
        os.close(self._wake)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    """Throughput check: stream for N seconds, optionally with colour and IR on V4L2 at the same time."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seconds = float(args[0]) if args else 10.0
    payload = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--payload=")), None)
    nodes = v4l2.discover() if "--all" in sys.argv else None
    others = {}
    if nodes:
        others = {"color": v4l2.Capture(nodes["color"], "YUYV", W, H), "ir": v4l2.Capture(nodes["ir"], "GREY", W, H)}
    counts = {k: 0 for k in others}

    def pump(name, cap):
        for buf in cap.frames():
            if len(buf) >= cap.sizeimage:
                counts[name] += 1

    threads = [threading.Thread(target=pump, args=kv, daemon=True) for kv in others.items()]
    for t in threads:
        t.start()
    cpu0 = time.process_time()
    with DepthCapture("Y10" if "--y10" in sys.argv else "Y14", payload=payload) as cap:
        print(f"{cap.fourcc.strip()}: payload {cap.payload} bytes, frame {cap.sizeimage} bytes", flush=True)
        stamps, lags, t0 = [], [], time.monotonic()
        while time.monotonic() - t0 < seconds:
            buf = cap.read(timeout=2)
            if buf is not None:
                stamps.append(cap.last_time)
                lags.append(time.monotonic() - cap.last_time)
        bad, dropped, hl = cap.assembler.bad, cap.dropped, cap.assembler.header_len
    cpu = time.process_time() - cpu0
    for cap in others.values():
        cap.close()
    elapsed = time.monotonic() - t0
    d = np.diff(stamps) * 1000 if len(stamps) > 1 else np.zeros(1)
    print(f"depth {len(stamps) / elapsed:.1f} fps, {bad} bad frames, {dropped} dropped by reader, header {hl} bytes")
    print(f"stamp step {np.median(d):.2f} ms (spread p5-p95 {np.percentile(d, 5):.2f}-{np.percentile(d, 95):.2f}), "
          f"reached us {1000 * np.median(lags):.1f} ms after stamp")
    print(f"CPU {100 * cpu / elapsed:.0f}% of one core (whole process)")
    for k, n in counts.items():
        print(f"{k} {n / elapsed:.1f} fps (V4L2)")


if __name__ == "__main__":
    main()
