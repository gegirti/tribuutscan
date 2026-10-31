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
