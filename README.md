# ferret-linux

Using a Creality CR-Scan Ferret Pro directly on Linux, with no Creality software.

Status: **capture works** (colour, IR, depth). There is no scanning/fusion yet, and the depth scale
and camera calibration are unknown.

```
python3 -m ferret.viewer        # live colour + IR + depth
```

Needs Python 3.10+, numpy and tkinter. No driver: the stock `uvcvideo` module handles the device.
Light the scene and uncover the lenses, or every stream is black.

## What the device looks like

USB `2bc5:069c` (Orbbec vendor ID, strings "Creality" / "CR-Scan Ferret", also on the Pro). USB 3.

| Node | USB iface | Content | Format |
|---|---|---|---|
| `/dev/video0` | 1.0 | depth, 1280x800 | `Y10 `, bytesperline 1600 (packed 10-bit) |
| `/dev/video2` | 1.2 | IR, 1280x800, dot pattern visible | GREY / MJPG |
| `/dev/video4` | 1.4 | colour, up to 1920x1080 | YUYV / MJPG |

Odd nodes (1, 3, 5) are metadata. `/dev/hidraw*` is a vendor control interface (root only).
The IR projector turns on by itself when the IR stream is opened.

## Findings

- **Depth packing**: an MSB-first 10-bit bitstream, 4 pixels per 5 bytes. It is not MIPI Y10P.
  See `ferret/depth.py`. Raw values are ~1..200, nearer = larger. 0 = invalid.
- **IR is mirrored** horizontally relative to colour and depth.
- ffmpeg/GStreamer/OBS-style apps can't negotiate the depth format, and camera apps may pick that
  node and show black. `ferret/v4l2.py` talks to V4L2 directly (raw ioctl + mmap, 64-bit offsets).
- Some depth frames arrive truncated; drop any frame shorter than `sizeimage`.
- The Orbbec SDK (`pyorbbecsdk2` 2.9.3) does not enumerate the device ("No matched usb device
  found"), although its udev rules list `2bc5:069c`.
- Only ~10% of depth pixels are valid in a test scene (plastic handle yes; glass, box, floor no).
  Cause not yet known.

## Open questions

1. What unit is the depth in (disparity? mm/8?). Plan: capture a flat wall at known distances.
2. Where do the camera intrinsics / depth-to-colour alignment come from? Plan: capture the USB
   traffic from Creality's app (usbmon, needs sudo).
3. Why is depth sparse, and can a full-range mode be enabled?
4. Fusion: depth + colour -> point cloud -> Open3D / RTAB-Map -> mesh.

## Layout

- `ferret/v4l2.py` capture and node discovery
- `ferret/depth.py` depth decode (and encode, for tests)
- `ferret/viewer.py` live viewer
- `tests/` run with pytest, or call the test functions directly
