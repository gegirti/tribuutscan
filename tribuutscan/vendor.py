"""Orbbec vendor commands over the depth function's UVC extension unit. Reads, plus one write.

The depth function's extension unit 4 (GUID a55751a1-f3c5-4a5e-8d5a-6854b8fa2716) is Orbbec's
command channel, the same one OpenCV's cap_obsensor and OrbbecSDK_v2 (Protocol.hpp, MIT) use:
- a request is a header (u16 magic 0x4d47, size in half-words, opcode, request id) and a payload,
  sent with SET_CUR on selector 2 / 1 / 3 (64 / 512 / 1024 bytes, chosen by request length)
- the reply (u16 magic 0x4252, size, opcode, request id, error code, then data) is read with
  GET_CUR from the selector that fits the expected reply length
Goes through uvcvideo (UVCIOC_CTRL_QUERY on the depth node), or, while usbdepth.DepthCapture owns the
depth function (14-bit depth streaming, /dev/video0 gone), as usbfs control transfers through it. Only the read opcodes below can be sent: `request` refuses the others.
The writes: `switch_depth_mode` puts an entry of the scanner's own mode list into the current depth
mode (1043), as OrbbecSDK_v2's depth work mode manager does, with no stream running; `set_property`
sets only the properties in SETTABLE (the IMU's on/off switches, as the SDK's IMU sensors do).

    python3 -m tribuutscan.vendor            # known properties and structures
    python3 -m tribuutscan.vendor modes      # the depth modes the scanner offers (list 4034)
    python3 -m tribuutscan.vendor factory    # the factory calibration (lists 4026, 4027, 4024)
    python3 -m tribuutscan.vendor switch "Long_range_mode Default"   # WRITES the current depth mode
    python3 -m tribuutscan.vendor scan       # also try every property id 0-299 and structure 1000-1043
Careful: a read of structure 1044 hung the scanner's firmware until it was replugged. Any timeout
ends the run (TimeoutError is not caught).
"""
import errno
import fcntl
import os
import struct
import sys
import threading
import time
from pathlib import Path

from . import usbdepth, v4l2
from .usbdepth import address

UVCIOC_CTRL_QUERY = 0xC0107521
UNIT = 4
SET_CUR, GET_CUR = 0x01, 0x81
SELECTOR_SIZE = {2: 64, 1: 512, 3: 1024}
REQUEST_MAGIC, RESPONSE_MAGIC = 0x4D47, 0x4252

GET_PROPERTY, GET_STRUCTURE = 1, 3
INIT_READ_RAW, READ_RAW, FINISH_READ_RAW = 17, 18, 19
GET_CMD_VERSION, GET_STRUCTURE_V11 = 26, 27
INIT_READ_LIST, READ_LIST, FINISH_READ_LIST = 29, 30, 31
GET_PROTOCOL_VERSION = 0xFE
SET_PROPERTY = 2  # used only by set_property, for SETTABLE
SET_STRUCTURE_V11 = 28  # used only by switch_depth_mode
SETTABLE = {2019: "GYRO_SWITCH", 2020: "ACCEL_SWITCH"}  # OrbbecSDK_v2 InternalProperty.hpp
CURRENT_DEPTH_MODE = 1043
READ_OPCODES = {GET_PROPERTY, GET_STRUCTURE, INIT_READ_RAW, READ_RAW, FINISH_READ_RAW, GET_CMD_VERSION,
                GET_STRUCTURE_V11, INIT_READ_LIST, READ_LIST, FINISH_READ_LIST, GET_PROTOCOL_VERSION}
ERRORS = {1: "invalid request", 2: "unsupported", 3: "busy", 4: "forbidden", 5: "error command",
          6: "not found", 0xFFFF: "unknown error"}

# OrbbecSDK_v2 include/libobsensor/h/Property.h
PROPERTIES = {2: "LDP", 3: "LASER", 4: "LASER_PULSE_WIDTH", 14: "DEPTH_MIRROR", 18: "IR_MIRROR",
              22: "MIN_DEPTH", 23: "MAX_DEPTH", 32: "LDP_STATUS", 75: "DEPTH_PRECISION_LEVEL", 79: "LASER_MODE",
              95: "DEVICE_WORK_MODE", 98: "SWITCH_IR_MODE", 99: "LASER_POWER_LEVEL", 100: "LDP_MEASURE_DISTANCE",
              119: "LASER_POWER_ACTUAL_LEVEL", 182: "LASER_CONTROL", 184: "IR_BRIGHTNESS"}
DEPTH_MODE_LIST = 4034  # OB_RAW_DATA_DEPTH_ALG_MODE_LIST (OrbbecSDK_v2 InternalProperty.hpp)
# the factory calibration, read as the SDK's G2AlgParamManager does (list command versions 1, 0, 0)
DEPTH_CALIB_PARAM, ALIGN_CALIB_PARAM, D2C_PROFILE_LIST = 4026, 4027, 4024
LIST_CHUNK = 252  # bytes per list read; the SDK uses up to DEFAULT_CMD_MAX_DATA_SIZE
STRUCTURES = {1002: "BASELINE_CALIBRATION_PARAM", 1003: "DEVICE_TEMPERATURE", 1035: "DEVICE_SERIAL_NUMBER",
              1043: "CURRENT_DEPTH_ALG_MODE", 1045: "DEPTH_PRECISION_SUPPORT_LIST",
              1069: "PRESET_RESOLUTION_CONFIG"}
PRIVATE = {1035, 1063}  # serial numbers: never printed
HANGS = {1044}  # GET_STRUCTURE 1044 hung the firmware (2026-10-31): no reply, then the scanner left USB


class VendorError(Exception):
    pass


class Channel:
    """The command channel: through uvcvideo on the depth node (`dev`), or through `usb`, a
    usbdepth.DepthCapture that owns the depth function. One request/reply at a time (a lock)."""

    def __init__(self, dev=None, usb=None):
        self.usb = usb
        self.fd = None if usb else os.open(dev or v4l2.discover()["depth"], os.O_RDWR)
        self.request_id = int(time.time()) & 0x7FFF
        self.lock = threading.Lock()

    def _xu(self, query, selector, buf):
        if self.usb is not None:  # class request to unit 4 on interface 0 (the depth function's VC)
            index = UNIT << 8 | usbdepth.VC
            if query == SET_CUR:
                self.usb.control(0x21, SET_CUR, selector << 8, index, bytes(buf))
            else:
                data = self.usb.control(0xA1, query, selector << 8, index, len(buf))
                buf[:len(data)] = data
            return buf
        arg = bytearray(struct.pack("<BBBxH2xQ", UNIT, selector, query, len(buf), address(buf)))
        for _ in range(5):
            try:
                fcntl.ioctl(self.fd, UVCIOC_CTRL_QUERY, arg)
                return buf
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EBUSY):
                    raise
                time.sleep(0.01)
        raise VendorError("extension unit busy")

    def request(self, opcode, payload=b"", reply_len=64):
        """Send one READ request; returns the reply data (bytes after the 10-byte header).
        A timeout is raised as TimeoutError: callers must stop then, not try the next id."""
        if opcode not in READ_OPCODES:
            raise VendorError(f"opcode {opcode} is not a read: refused")
        if opcode in (GET_STRUCTURE, GET_STRUCTURE_V11) and struct.unpack_from("<I", payload)[0] in HANGS:
            raise VendorError("known to hang the firmware: refused")
        return self._exchange(opcode, payload, reply_len)

    def _exchange(self, opcode, payload, reply_len):
        with self.lock:
            return self._exchange_locked(opcode, payload, reply_len)

    def _exchange_locked(self, opcode, payload, reply_len):
        self.request_id = (self.request_id + 1) & 0xFFFF
        req = struct.pack("<HHHH", REQUEST_MAGIC, len(payload) // 2, opcode, self.request_id) + payload
        send_sel = 2 if len(req) <= 64 else 1 if len(req) <= 512 else 3
        self._xu(SET_CUR, send_sel, bytearray(req.ljust(SELECTOR_SIZE[send_sel], b"\0")))
        recv_sel = 2 if reply_len <= 64 else 1 if reply_len <= 512 else 3
        for _ in range(3):
            resp = bytes(self._xu(GET_CUR, recv_sel, bytearray(SELECTOR_SIZE[recv_sel])))
            magic, half_words, op, rid, err = struct.unpack_from("<HHHHH", resp)
            if magic != RESPONSE_MAGIC or rid != self.request_id:
                raise VendorError(f"bad reply: magic {magic:#06x}, id {rid} (sent {self.request_id})")
            if err != 3:  # 3 = busy: read again
                break
            time.sleep(0.02)
        if err:
            raise VendorError(ERRORS.get(err, f"error {err:#06x}"))
        size = half_words * 2 - 2  # the size counts the error code too
        if 10 + size > len(resp):
            raise VendorError(f"reply of {size} bytes does not fit selector {recv_sel}")
        return resp[10:10 + size]

    def get_property(self, pid):
        """(cur, max, min, def, step)"""
        return struct.unpack("<5i", self.request(GET_PROPERTY, struct.pack("<I", pid))[:20])

    def set_property(self, pid, value):
        """Set an integer property; only the ids in SETTABLE are allowed."""
        if pid not in SETTABLE:
            raise VendorError(f"property {pid} is not settable here: refused")
        self._exchange(SET_PROPERTY, struct.pack("<Ii", pid, int(value)), 64)

    def get_structure(self, sid, reply_len=512):
        return self.request(GET_STRUCTURE, struct.pack("<I", sid), reply_len)

    def read_list(self, pid, cmd_version=0):
        """A structure list, the SDK's way (getStructureDataListProtoV1_1): start (cmdVer, size), read in
        chunks, finish. Returns the bytes; finish is sent whenever start succeeded (not after a timeout)."""
        ver, size = struct.unpack_from("<HI", self.request(INIT_READ_LIST, struct.pack("<I", pid)))
        out = bytearray()
        try:
            if ver != cmd_version:
                raise VendorError(f"list {pid}: command version {ver}, expected {cmd_version}")
            while len(out) < size:
                n = min(LIST_CHUNK, size - len(out))
                out += self.request(READ_LIST, struct.pack("<III", pid, len(out), n), 1024)[:n]
        finally:
            self.request(FINISH_READ_LIST, struct.pack("<I", pid), 1024)
        return bytes(out)

    def mode_entries(self):
        """The raw 52-byte entries of the mode list (OBDepthWorkMode_Internal: checksum, name, option)."""
        data = self.read_list(DEPTH_MODE_LIST)
        return [data[i:i + 52] for i in range(0, len(data) - 51, 52)]

    def depth_modes(self):
        """[(name, option code, checksum hex)] from the mode list."""
        return [(mode_name(e), struct.unpack_from("<I", e, 48)[0], e[:16].hex()) for e in self.mode_entries()]

    def current_mode(self):
        ver, entry = struct.unpack_from("<H", r := self.request(GET_STRUCTURE_V11, struct.pack("<I", CURRENT_DEPTH_MODE)))[0], r[2:]
        if ver != 0:
            raise VendorError(f"current depth mode: command version {ver}, expected 0")
        return entry

    def factory_calibration(self):
        """The scanner's own calibration for its current depth mode (reads only):
        {"depth_calib": parse_depth_calib, "cameras": parse_camera_params, "profiles": parse_d2c_profiles}."""
        return {"depth_calib": parse_depth_calib(self.read_list(DEPTH_CALIB_PARAM, 1)),
                "cameras": parse_camera_params(self.read_list(ALIGN_CALIB_PARAM)),
                "profiles": parse_d2c_profiles(self.read_list(D2C_PROFILE_LIST))}

    def switch_depth_mode(self, name):
        """THE WRITE: make `name`, an entry of the scanner's own list, the current depth mode
        (SET_STRUCTURE_V1_1 of 1043, command version 0, the entry's bytes unchanged). Returns False if
        it already was. No stream may run (the SDK refuses too); callers check that."""
        entry = next((e for e in self.mode_entries() if mode_name(e) == name), None)
        if entry is None:
            raise VendorError(f"{name!r} is not in the scanner's mode list")
        if self.current_mode() == entry:
            return False
        self._exchange(SET_STRUCTURE_V11, struct.pack("<IH", CURRENT_DEPTH_MODE, 0) + entry, 64)
        return True

    def close(self):
        if self.fd is not None:
            os.close(self.fd)


def mode_name(entry):
    return entry[16:48].split(b"\0")[0].decode(errors="replace")


# packed layouts from OrbbecSDK_v2 src/shared/InternalTypes.hpp and include/libobsensor/h/ObTypes.h (MIT)
def parse_depth_calib(data):
    """OBDepthCalibrationParam (28 bytes): distance = unit * baseline * focal / (disparity + offset)."""
    keys = ("depth_mode", "baseline", "z0", "focal", "unit", "disp_offset", "invalid_disp")
    return dict(zip(keys, struct.unpack_from("<I5fi", data)))


def parse_camera_params(data):
    """OBCameraParam_Internal_V0 records (152 bytes each): depth and colour intrinsics (fx fy cx cy w h),
    distortion (k1-k6, p1, p2) and the depth -> colour transform (colour = R depth + T, mm)."""
    out = []
    for o in range(0, len(data) - 151, 152):
        intr = [dict(zip(("fx", "fy", "cx", "cy", "w", "h"), struct.unpack_from("<4f2h", data, o + i)))
                for i in (0, 20)]
        dist = [dict(zip(("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"), struct.unpack_from("<8f", data, o + i)))
                for i in (40, 72)]
        rt = struct.unpack_from("<12f", data, o + 104)
        out.append({"depth": intr[0], "colour": intr[1], "depth_dist": dist[0], "colour_dist": dist[1],
                    "R": list(rt[:9]), "T_mm": list(rt[9:])})
    return out


def parse_d2c_profiles(data):
    """OBD2CProfile entries (22 bytes each): which camera record serves a colour/depth size pair, and the
    crop (left, top, right, bottom; negative = rows/columns cut) after scaling by `scale`. record 255: none."""
    out = []
    for o in range(0, len(data) - 21, 22):
        cw, ch, dw, dh, align, record, scale, *crop = struct.unpack_from("<4HBBf4h", data, o)
        out.append({"colour": (cw, ch), "depth": (dw, dh), "align": align & 7, "record": record,
                    "scale": scale, "crop": tuple(crop)})
    return out


def camera_users():
    """PIDs (other than ours) holding any of the scanner's video nodes open."""
    nodes = {os.path.realpath(p) for p in v4l2.discover().values()}
    users = set()
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            if any(os.path.realpath(f) in nodes for f in fd_dir.iterdir()):
                users.add(int(fd_dir.parent.name))
        except OSError:
            continue
    return users - {os.getpid()}


def show(data):
    text = bytes(c if 32 <= c < 127 else 46 for c in data).decode()
    return f"{len(data)} bytes  {data[:48].hex(' ')}{' ...' if len(data) > 48 else ''}  |{text[:48]}|"


def main():
    if sys.argv[1:2] == ["switch"]:
        if camera_users():
            sys.exit(f"close the programs using the scanner first (pids {sorted(camera_users())})")
        ch = Channel()
        try:
            before = mode_name(ch.current_mode())
            changed = ch.switch_depth_mode(sys.argv[2])
            print(f"{before} -> {mode_name(ch.current_mode())}" if changed else f"already {before}")
        finally:
            ch.close()
        return
    ch = Channel()
    try:
        if "modes" in sys.argv:
            cur = ch.get_structure(1043)[16:48].split(b"\0")[0].decode()
            for name, option, checksum in ch.depth_modes():
                print(f"  {'*' if name == cur else ' '} {name:32} option {option:#x}  checksum {checksum}")
            return
        if "factory" in sys.argv:
            print(f"factory calibration ({mode_name(ch.current_mode())}):")
            fac = ch.factory_calibration()
            print(f"  depth law {fac['depth_calib']}")
            for n, cam in enumerate(fac["cameras"]):
                print(f"  record {n}: {cam}")
            for p in fac["profiles"]:
                print(f"  profile {p}")
            return
        print("properties (cur, max, min, def, step):")
        ids = range(300) if "scan" in sys.argv else PROPERTIES
        for pid in ids:
            try:
                print(f"  {pid:4} {PROPERTIES.get(pid, ''):26} {ch.get_property(pid)}")
            except VendorError as e:
                if "scan" not in sys.argv:
                    print(f"  {pid:4} {PROPERTIES.get(pid, ''):26} -- {e}")
        print("structures:")
        ids = range(1000, 1043) if "scan" in sys.argv else STRUCTURES  # 1044 hung the firmware; nothing past it yet
        for sid in ids:
            try:
                data = ch.get_structure(sid)
                print(f"  {sid:4} {STRUCTURES.get(sid, ''):26} " + (f"{len(data)} bytes (private)" if sid in PRIVATE else show(data)))
            except VendorError as e:
                if "scan" not in sys.argv:
                    print(f"  {sid:4} {STRUCTURES.get(sid, ''):26} -- {e}")
    finally:
        ch.close()


if __name__ == "__main__":
    main()
