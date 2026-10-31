"""The scanner's IMU (ICM-40608 accelerometer + gyroscope), read over USB ourselves.

The IMU is USB interface 6 (HID "CR-Scan FerretIMU": one vendor-defined 248-byte input report on
interrupt endpoint 0x85). It only sends while switched on over the command channel (vendor.py,
properties 2019 gyro and 2020 accel). This module unbinds usbhid from interface 6 only and reads the
endpoint through usbfs, like usbdepth.py (same udev rule, no root). Packet layout from OrbbecSDK_v2
ImuStreamer.hpp (MIT): an 8-byte header (report id, sample rate, group length, group count, reserved),
then groups of int16 group id, accel xyz, gyro xyz, temperature and a 64-bit device timestamp.

    python3 -m tribuutscan.imu [seconds]     # switch the IMU on, print what it sends, switch it off

`switch_on(channel)` returns a running ImuStream (latest values, a few seconds of history, tilt and turn
rate, an optional listener such as a Recorder); `switch_off(channel, stream)` undoes it. The axes are
the IMU's own until its calibration (raw data 4036) is read; at rest upright, gravity is on -y.
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
            usbdepth.claim(self.usb, IFACE)
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


STILL_DPS, STILL_G = 2.0, 0.05  # "still": |gyro| under 2 deg/s (the bias is ~0.9) and |a| within 0.05 g of 1


class ImuStream(threading.Thread):
    """Reads an ImuReader in the background. `history` holds rows of host time (time.monotonic() at
    arrival) followed by parse()'s columns; `listener(rows)`, when set, gets every new block."""

    def __init__(self, reader, accel_g, gyro_dps, settings=None, keep_s=5.0):
        super().__init__(daemon=True)
        self.reader, self.accel_g, self.gyro_dps = reader, accel_g, gyro_dps
        self.settings = settings or {}
        self.history = collections.deque(maxlen=int(keep_s * 1000))
        self.listener, self.rate, self.running = None, 0.0, True

    def run(self):
        count, t0 = 0, time.monotonic()
        while self.running and self.reader.error is None:
            time.sleep(0.02)
            blocks = [np.c_[np.full(len(p), t), p] for t, r in self.reader.reports() if len(p := parse(r, self.accel_g, self.gyro_dps))]
            if blocks:
                rows = np.concatenate(blocks)
                self.history.extend(rows)
                count += len(rows)
                listener = self.listener
                if listener is not None:
                    listener(rows)
            if time.monotonic() - t0 >= 1:
                self.rate, count, t0 = count / (time.monotonic() - t0), 0, time.monotonic()

    def state(self, window_s=0.1):
        """For display: tilt from upright (pitch, roll, degrees, IMU axes), turn rate (deg/s), still,
        rate (Hz), temperature (C), |a| (g); None before data or after the device went away."""
        recent = list(self.history)[-max(1, int(window_s * 200)):]
        if not recent or self.reader.error is not None:
            return None
        r = np.array(recent)
        a, g = r[:, 2:5].mean(0), r[:, 5:8].mean(0)
        turn, norm = float(np.linalg.norm(g)), float(np.linalg.norm(a))
        return {"pitch": float(np.degrees(np.arctan2(a[2], -a[1]))), "roll": float(np.degrees(np.arctan2(a[0], -a[1]))),
                "turn": turn, "still": turn < STILL_DPS and abs(norm - 1) < STILL_G, "rate": self.rate,
                "temp": float(r[-1, 8]), "g": norm}

    def stop(self):
        self.running = False
        self.join(timeout=1)
        self.reader.close()


def switch_on(ch):
    """Switch the IMU on over vendor channel `ch` and start reading it; returns the running ImuStream."""
    s = {pid: ch.get_property(pid)[0] for pid in (GYRO_RATE, ACCEL_RATE, GYRO_RANGE, ACCEL_RANGE)}
    settings = {"gyro_hz": RATE_HZ.get(s[GYRO_RATE]), "gyro_dps": GYRO_DPS[s[GYRO_RANGE]],
                "accel_hz": RATE_HZ.get(s[ACCEL_RATE]), "accel_g": ACCEL_G[s[ACCEL_RANGE]]}
    ch.set_property(GYRO_ON, 1)
    ch.set_property(ACCEL_ON, 1)
    try:
        reader = ImuReader()
    except OSError:
        switch_off(ch)
        raise
    stream = ImuStream(reader, settings["accel_g"], settings["gyro_dps"], settings)
    stream.start()
    return stream


def switch_off(ch, stream=None):
    """Stop reading (interface 6 goes back to usbhid) and switch the IMU off."""
    if stream is not None:
        stream.stop()
    ch.set_property(GYRO_ON, 0)
    ch.set_property(ACCEL_ON, 0)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    ch = vendor.Channel()
    try:
        stream = switch_on(ch)
        try:
            print(", ".join(f"{k} {v}" for k, v in stream.settings.items()))
            time.sleep(seconds)
            rows = np.array(list(stream.history))
        finally:
            switch_off(ch, stream)
    finally:
        ch.close()
    if not len(rows):
        sys.exit("no IMU samples")
    s = rows[:, 1:]
    dt = np.diff(s[:, 0])
    print(f"{len(s)} samples ({len(s) / seconds:.0f}/s)")
    print(f"device time step: median {np.median(dt):.0f}, min {dt.min():.0f}, max {dt.max():.0f} (us)")
    acc, gyr = s[:, 1:4], s[:, 4:7]
    print(f"accel mean {acc.mean(0).round(3)} g, |a| {np.linalg.norm(acc, axis=1).mean():.3f} g, "
          f"std {acc.std(0).round(4)}")
    print(f"gyro mean {gyr.mean(0).round(3)} deg/s, std {gyr.std(0).round(3)}; temperature {s[:, 7].mean():.1f} C")


if __name__ == "__main__":
    main()
