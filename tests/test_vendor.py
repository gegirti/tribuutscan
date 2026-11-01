import struct

from tribuutscan import vendor


class FakeUsb:
    """Records control transfers; answers GET_CUR with a canned reply."""

    def __init__(self, reply):
        self.calls, self.reply = [], reply

    def control(self, request_type, request, value, index, data_or_len, timeout_ms=1000):
        self.calls.append((request_type, request, value, index, data_or_len))
        return self.reply if isinstance(data_or_len, int) else b""


def test_channel_over_usb_uses_class_requests_to_unit_4():
    ch = vendor.Channel(usb=FakeUsb(b""))
    reply = struct.pack("<HHHHH", vendor.RESPONSE_MAGIC, 11, vendor.GET_PROPERTY, ch.request_id + 1, 0)
    ch.usb.reply = (reply + struct.pack("<5i", 1, 1, 0, 0, 1)).ljust(64, b"\0")
    assert ch.get_property(2019) == (1, 1, 0, 0, 1)
    (t1, r1, v1, i1, d1), (t2, r2, v2, i2, n2) = ch.usb.calls
    assert (t1, r1, v1, i1, len(d1)) == (0x21, 0x01, 2 << 8, 4 << 8 | 0, 64)  # SET_CUR, selector 2, unit 4, interface 0
    assert struct.unpack_from("<HHHHI", d1) == (vendor.REQUEST_MAGIC, 2, vendor.GET_PROPERTY, ch.request_id, 2019)
    assert (t2, r2, v2, i2, n2) == (0xA1, 0x81, 2 << 8, 4 << 8 | 0, 64)  # GET_CUR the reply
    ch.close()


def test_only_the_imu_switches_are_settable():
    ch = vendor.Channel(usb=FakeUsb(b""))
    try:
        ch.set_property(1043, 1)
        raise AssertionError("set_property(1043) was not refused")
    except vendor.VendorError:
        pass
    assert ch.usb.calls == []


def test_factory_calibration_parsers():
    law = vendor.parse_depth_calib(struct.pack("<I5fi", 1, 150.25, 662.0, 960.5, 0.5, 218.0, 0))
    assert law["baseline"] == 150.25 and law["disp_offset"] == 218.0 and law["unit"] == 0.5
    rec = struct.pack("<4f2h", 960.5, 960.5, 646.25, 398.75, 1280, 800) + struct.pack("<4f2h", 842, 842, 637, 395, 1280, 800)
    rec += struct.pack("<8f", *[0] * 8) + struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 8) + struct.pack("<12f", *range(12))
    (cam,) = vendor.parse_camera_params(rec)
    assert cam["depth"]["cx"] == 646.25 and cam["colour"]["h"] == 800 and cam["colour_dist"]["p2"] == 8
    assert cam["R"] == list(range(9)) and cam["T_mm"] == [9, 10, 11]
    (prof,) = vendor.parse_d2c_profiles(struct.pack("<4HBBf4h", 1280, 720, 1280, 800, 3, 0, 1.0, 0, -40, 0, -40))
    assert prof == {"colour": (1280, 720), "depth": (1280, 800), "align": 3, "record": 0, "scale": 1.0,
                    "crop": (0, -40, 0, -40)}


def test_read_list_finishes_after_a_wrong_command_version():
    ch = vendor.Channel(usb=FakeUsb(b""))
    sent = []

    def request(opcode, payload=b"", reply_len=64):
        sent.append(opcode)
        return struct.pack("<HI", 1, 28) if opcode == vendor.INIT_READ_LIST else b""

    ch.request = request
    try:
        ch.read_list(vendor.ALIGN_CALIB_PARAM, 0)
        raise AssertionError("a wrong command version was accepted")
    except vendor.VendorError:
        pass
    assert sent == [vendor.INIT_READ_LIST, vendor.FINISH_READ_LIST]
