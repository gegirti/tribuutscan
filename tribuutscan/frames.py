"""Grab and save synchronised-ish frame sets (colour, IR, depth), all 1280x800.

Colour is requested at 1280x800 so it shares the depth map's pixel grid. Whether the two are truly
registered (different cameras, different baselines) is not verified; identity is assumed.
Depth is 14-bit over USB (usbdepth.py) when allowed, else 10-bit over V4L2; decoded depth is on the
Y10 scale either way, and recordings note which one they hold (depth_bytesperline). They also note the
scanner's depth mode (close / long range), whose law calib.CAMERA then uses when they are read.
"""
import json
import os
import select
import threading
import time
from pathlib import Path

import numpy as np

from . import calib, usbdepth, v4l2, vendor
from .depth import decode_depth
from .image import depth_to_rgb, write_png, yuyv_to_rgb

W, H = 1280, 800
CAPTURES = Path(__file__).resolve().parent.parent / "captures"


def decode(color_buf, ir_buf, depth_buf, depth_bytesperline):
    return {
        "color": yuyv_to_rgb(color_buf, W, H),
        "ir": np.frombuffer(ir_buf, np.uint8, W * H).reshape(H, W)[:, ::-1].copy(),  # flipped to match colour/depth
        "depth": decode_depth(depth_buf, W, H, depth_bytesperline),
    }


def depth_mode(nodes, log=print):
    """The scanner's active depth mode (read only, over vendor.py), or close range if it can't be read."""
    try:
        ch = vendor.Channel(nodes["depth"])
        try:
            return vendor.mode_name(ch.current_mode())
        finally:
            ch.close()
    except (OSError, vendor.VendorError) as e:
        log(f"depth mode not readable ({e}); assuming {calib.CLOSE}")
        return calib.CLOSE


def open_depth(nodes, log=print):
    """The depth capture: 14-bit over USB when the udev rule allows it, else V4L2 10-bit (also when
    TRIBUUTSCAN_DEPTH=y10). Reads the depth mode first and switches calib.CAMERA to it (the capture's
    `depth_mode`). Call v4l2.discover() first: the USB capture takes /dev/video0 away."""
    mode = depth_mode(nodes, log)
    calib.use_mode(mode)
    cap = None
    if os.environ.get("TRIBUUTSCAN_DEPTH", "").lower() != "y10":
        if usbdepth.available():
            try:
                cap = usbdepth.DepthCapture("Y14")
            except OSError as e:
                log(f"14-bit depth over USB failed ({e.strerror or e}); using 10-bit V4L2 depth")
        else:
            log("14-bit depth needs udev/70-tribuutscan.rules (see README); using 10-bit V4L2 depth")
    cap = cap or v4l2.Capture(nodes["depth"], "Y10", W, H)
    cap.depth_mode = mode
    return cap


def grab(warmup=10):
    """Open all three nodes, discard warm-up frames, and return one decoded frame set."""
    nodes = v4l2.discover()
    with v4l2.Capture(nodes["color"], "YUYV", W, H) as c, v4l2.Capture(nodes["ir"], "GREY", W, H) as i, \
            open_depth(nodes) as d:
        last = {}
        for _ in range(warmup):
            for key, cap in (("color", c), ("ir", i), ("depth", d)):
                buf = cap.read()
                if buf is not None and len(buf) >= cap.sizeimage:
                    last[key] = buf
        if len(last) != 3:
            raise RuntimeError(f"no usable frames from: {set(('color', 'ir', 'depth')) - set(last)}")
        return decode(last["color"], last["ir"], last["depth"], d.bytesperline)


def paired(caps, max_skew_s=0.02, timeout=2.0):
    """The newest frames of {name: Capture}, all captured within max_skew_s of each other, as
    {name: (bytes, capture time)}; None after `timeout` seconds. Reads whichever stream is ready, so
    no stream waits behind another (reading them in turn leaves them a frame or more apart)."""
    names = {cap.fd: name for name, cap in caps.items()}
    latest, end = {}, time.monotonic() + timeout
    while time.monotonic() < end:
        for fd in select.select(list(names), [], [], 0.1)[0]:
            cap = caps[names[fd]]
            buf = cap.read(timeout=0)
            if buf is not None and len(buf) >= cap.sizeimage:
                latest[names[fd]] = (buf, cap.last_time)
        if len(latest) == len(caps) and not any(cap.ready() for cap in caps.values()):
            times = [t for _, t in latest.values()]
            if max(times) - min(times) <= max_skew_s:
                return latest
    return None


def save(frames, root=CAPTURES, extra=None):
    """Write a frame set to captures/<timestamp>/ and return the directory.

    `extra` (e.g. the camera control values in use) is merged into info.json.
    """
    out = Path(root) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    write_png(out / "color.png", frames["color"])
    write_png(out / "ir.png", frames["ir"])
    write_png(out / "depth_preview.png", depth_to_rgb(frames["depth"]))
    np.savez_compressed(out / "frames.npz", **frames)
    valid = frames["depth"][frames["depth"] > 0]
    (out / "info.json").write_text(json.dumps({
        "size": [W, H],
        "depth_valid_fraction": round(float((frames["depth"] > 0).mean()), 4),
        "depth_median": float(np.median(valid)) if valid.size else None,
        "note": "ir is flipped horizontally to match colour/depth (sensor output is mirrored); depth is raw, on the Y10 scale",
        "depth_mode": calib.CAMERA.mode,
        **(extra or {}),
    }, indent=2))
    return out


STREAMS = ("color", "ir", "depth")
FRAME_BYTES = {"color": W * H * 2, "ir": W * H, "depth": W * H * 10 // 8}  # YUYV, GREY, packed Y10 (Y14: 14 // 8)


class Recorder:
    """Append raw frame sets to captures/<timestamp>-rec/: one file per stream plus stamps.csv.

    Frames are stored undecoded (about 4.3 MB per set, 4.9 MB with 14-bit depth) so recording keeps up
    with the camera; `read_recording` decodes them later.
    """

    def __init__(self, depth_bytesperline, root=CAPTURES, extra=None):
        self.dir = Path(root) / (time.strftime("%Y%m%d-%H%M%S") + "-rec")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frame_bytes = {**FRAME_BYTES, "depth": depth_bytesperline * H}
        (self.dir / "info.json").write_text(json.dumps({
            "size": [W, H], "depth_bytesperline": depth_bytesperline, "frame_bytes": self.frame_bytes,
            "depth_mode": calib.CAMERA.mode,
            "note": "raw sensor buffers; ir is NOT flipped here (read_recording flips it)",
            **(extra or {}),
        }, indent=2))
        self.files = {k: open(self.dir / f"{k}.raw", "wb") for k in STREAMS}
        self.stamps = open(self.dir / "stamps.csv", "w")
        self.stamps.write("index,color_t,ir_t,depth_t\n")
        self.imu, self.imu_rows, self.lock = None, 0, threading.Lock()
        self.count = 0

    def write(self, bufs, stamps):
        """bufs/stamps: dicts keyed by stream name; buffers are cut to their fixed frame size."""
        for k in STREAMS:
            self.files[k].write(bufs[k][: self.frame_bytes[k]])
        self.stamps.write(f"{self.count}," + ",".join(f"{stamps[k]:.4f}" for k in STREAMS) + "\n")
        self.count += 1

    @property
    def megabytes(self):
        return self.count * sum(self.frame_bytes.values()) / 1e6

    def write_imu(self, rows):
        """Append IMU samples (imu.ImuStream rows) to imu.csv; called from the IMU thread."""
        with self.lock:
            if self.stamps.closed:
                return
            if self.imu is None:
                self.imu = open(self.dir / "imu.csv", "w")
                self.imu.write(IMU_HEADER + "\n")
            np.savetxt(self.imu, np.asarray(rows)[:, :9], fmt=IMU_FORMAT, delimiter=",")
            self.imu_rows += len(rows)

    def close(self):
        with self.lock:
            if self.imu is not None:
                self.imu.close()
        for f in (*self.files.values(), self.stamps):
            f.close()


IMU_HEADER = "host_t,device_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_c"
IMU_FORMAT = ["%.5f", "%d"] + ["%.5f"] * 6 + ["%.2f"]


def read_imu(directory):
    """{column: array} from a recording's imu.csv (see IMU_HEADER), or None if it has none.
    host_t is time.monotonic() at arrival, like the frame stamps; device_us is the IMU's own clock."""
    path = Path(directory) / "imu.csv"
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(IMU_HEADER.split(","))}


def read_recording(directory):
    """Yield decoded frame sets (as from `decode`) from a Recorder directory.

    Each set also carries "t" (depth timestamp, s) and "skew_ms" (colour minus depth time) when the
    recording has stamps.csv. Switches calib.CAMERA to the recording's depth mode (close if not noted).
    """
    directory = Path(directory)
    info = json.loads((directory / "info.json").read_text())
    calib.use_mode(info.get("depth_mode", calib.CLOSE))
    sizes = {**FRAME_BYTES, **info.get("frame_bytes", {})}
    data = {k: np.memmap(directory / f"{k}.raw", np.uint8, "r") for k in STREAMS}
    stamps_file = directory / "stamps.csv"
    stamps = np.loadtxt(stamps_file, delimiter=",", skiprows=1, ndmin=2) if stamps_file.exists() else None
    for i in range(len(data["depth"]) // sizes["depth"]):
        bufs = [data[k][i * sizes[k]:(i + 1) * sizes[k]].tobytes() for k in STREAMS]
        frame = decode(*bufs, info["depth_bytesperline"])
        if stamps is not None and i < len(stamps):
            frame["t"] = float(stamps[i, 3])
            frame["skew_ms"] = float((stamps[i, 1] - stamps[i, 3]) * 1000)
        yield frame
