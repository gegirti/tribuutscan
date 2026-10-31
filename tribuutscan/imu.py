"""The scanner's IMU (ICM-40608 accelerometer + gyroscope), read over USB ourselves.

The IMU is USB interface 6 (HID "CR-Scan FerretIMU": one vendor-defined 248-byte input report on
interrupt endpoint 0x85). It only sends while switched on over the command channel (vendor.py,
properties 2019 gyro and 2020 accel). This module unbinds usbhid from interface 6 only and reads the
endpoint through usbfs, like usbdepth.py (same udev rule, no root). Packet layout from OrbbecSDK_v2
ImuStreamer.hpp (MIT): an 8-byte header (report id, sample rate, group length, group count, reserved),
then groups of int16 group id, accel xyz, gyro xyz, temperature and a 64-bit device timestamp.

    python3 -m tribuutscan.imu [seconds]     # switch the IMU on, print what it sends, switch it off
"""
import collections
import ctypes
import fcntl
import os
import select
import struct
import sys
import threading
import time

import numpy as np

from . import usbdepth, vendor
from .usbdepth import Urb, address, ioctl_ptr

IFACE, EP, PACKET = 6, 0x85, 248
URB_TYPE_INTERRUPT = 1
GROUP = struct.Struct("<8hQ")  # group id, accel xyz, gyro xyz, temperature, timestamp
# OrbbecSDK_v2 ObTypes.h enums -> physical values; value = raw * full scale / 32768
RATE_HZ = {7: 100, 8: 200, 9: 500, 10: 1000}
ACCEL_G = {1: 2, 2: 4, 3: 8, 4: 16, 5: 3, 6: 6, 7: 12, 8: 24}
GYRO_DPS = {1: 16, 2: 31, 3: 62, 4: 125, 5: 250, 6: 500, 7: 1000, 8: 2000, 9: 400, 10: 800}
GYRO_ON, ACCEL_ON, GYRO_RATE, ACCEL_RATE, GYRO_RANGE, ACCEL_RANGE = 2019, 2020, 2021, 2022, 2023, 2024


def parse(packet, accel_g=4, gyro_dps=1000):
    """(n, 9) float array from one report: device time (us), accel xyz (g), gyro xyz (deg/s), temperature
    (C; the SDK's ICM-42668 formula, an estimate here), group id. Empty if the header does not fit."""
    if len(packet) < 8:
        return np.empty((0, 9))
    glen, count = packet[2], packet[3]
    if glen != GROUP.size or 8 + count * glen > len(packet):
        return np.empty((0, 9))
    raw = np.array([GROUP.unpack_from(packet, 8 + k * glen) for k in range(count)], np.float64)
    out = np.empty((count, 9))
    out[:, 0] = raw[:, 8]
    out[:, 1:4] = raw[:, 1:4] * accel_g / 32768
    out[:, 4:7] = raw[:, 4:7] * gyro_dps / 32768
    out[:, 7] = raw[:, 7] / 132.48 + 25
    out[:, 8] = raw[:, 0]
    return out


class ImuReader:
    """Interface 6 over usbfs: a thread reaps interrupt URBs and keeps the raw reports (with arrival
    time.monotonic()) until `reports()` takes them."""

    def __init__(self, nurbs=4, keep=2000):
        path = usbdepth.find()
        if path is None:
            raise OSError("scanner not found")
        self.usb = os.open(path, os.O_RDWR)
        try:
            fcntl.ioctl(self.usb, usbdepth.USBDEVFS_DISCONNECT_CLAIM, struct.pack("<II256s", IFACE, 0, b""))
        except OSError:
            os.close(self.usb)
            raise
        self._ready = collections.deque(maxlen=keep)
        self._urbs, self._pending, self.error = {}, set(), None
        for _ in range(nurbs):
            buf = bytearray(PACKET)
            urb = Urb(type=URB_TYPE_INTERRUPT, endpoint=EP, buffer=address(buf), buffer_length=len(buf))
            self._urbs[ctypes.addressof(urb)] = (urb, buf)
        self._running = True
        for key in self._urbs:
            self._submit(key)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _submit(self, key):
        urb = self._urbs[key][0]
        urb.status = urb.actual_length = 0
        ioctl_ptr(self.usb, usbdepth.USBDEVFS_SUBMITURB, key)
        self._pending.add(key)

    def _reap(self, ptr):
        try:
            fcntl.ioctl(self.usb, usbdepth.USBDEVFS_REAPURBNDELAY, ptr)
        except BlockingIOError:
            return None
        key = struct.unpack("<Q", ptr)[0]
        self._pending.discard(key)
        return key

    def _run(self):
        ptr = bytearray(8)
        try:
            while self._running:
                if not select.select([], [self.usb], [], 0.2)[1]:
                    continue
                while (key := self._reap(ptr)) is not None:
                    urb, buf = self._urbs[key]
                    if urb.status == 0 and urb.actual_length:
                        self._ready.append((time.monotonic(), bytes(buf[:urb.actual_length])))
                    if self._running:
                        self._submit(key)
        except OSError as e:
            self.error = e

    def reports(self):
        """[(arrival time, report bytes)] received since the last call."""
        out = []
        while self._ready:
            out.append(self._ready.popleft())
        return out

    def close(self):
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=2)
        for key in list(self._pending):
            try:
                ioctl_ptr(self.usb, usbdepth.USBDEVFS_DISCARDURB, key)
            except OSError:
                pass
        ptr, end = bytearray(8), time.monotonic() + 1
        while self._pending and time.monotonic() < end:
            try:
                if self._reap(ptr) is None:
                    select.select([], [self.usb], [], 0.05)
            except OSError:
                break
        for request in (usbdepth.USBDEVFS_RELEASEINTERFACE,):
            try:
                fcntl.ioctl(self.usb, request, struct.pack("<I", IFACE))
            except OSError:
                pass
        try:  # hand interface 6 back to usbhid
            fcntl.ioctl(self.usb, usbdepth.USBDEVFS_IOCTL, struct.pack("<iiQ", IFACE, usbdepth.USBDEVFS_CONNECT, 0))
        except OSError:
            pass
        os.close(self.usb)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    ch = vendor.Channel()
    try:
        settings = {pid: ch.get_property(pid)[0] for pid in (GYRO_RATE, ACCEL_RATE, GYRO_RANGE, ACCEL_RANGE)}
        accel_g, gyro_dps = ACCEL_G[settings[ACCEL_RANGE]], GYRO_DPS[settings[GYRO_RANGE]]
        print(f"gyro {RATE_HZ.get(settings[GYRO_RATE], '?')} Hz, +-{gyro_dps} deg/s; "
              f"accel {RATE_HZ.get(settings[ACCEL_RATE], '?')} Hz, +-{accel_g} g")
        ch.set_property(GYRO_ON, 1)
        ch.set_property(ACCEL_ON, 1)
        try:
            with ImuReader() as imu:
                time.sleep(seconds)
                reports = imu.reports()
        finally:
            ch.set_property(GYRO_ON, 0)
            ch.set_property(ACCEL_ON, 0)
    finally:
        ch.close()
    if not reports:
        sys.exit("no IMU reports")
    heads = {r[:4].hex(" ") for _, r in reports}
    s = np.concatenate([parse(r, accel_g, gyro_dps) for _, r in reports])
    dt = np.diff(s[:, 0])
    print(f"{len(reports)} reports ({len(reports) / seconds:.0f}/s), {len(s)} samples ({len(s) / seconds:.0f}/s); "
          f"header bytes {sorted(heads)[:3]}")
    print(f"device time step: median {np.median(dt):.0f}, min {dt.min():.0f}, max {dt.max():.0f} (units as sent)")
    acc, gyr = s[:, 1:4], s[:, 4:7]
    print(f"accel mean {acc.mean(0).round(3)} g, |a| {np.linalg.norm(acc, axis=1).mean():.3f} g, "
          f"std {acc.std(0).round(4)}")
    print(f"gyro mean {gyr.mean(0).round(3)} deg/s, std {gyr.std(0).round(3)}; temperature {s[:, 7].mean():.1f} C")


if __name__ == "__main__":
    main()
