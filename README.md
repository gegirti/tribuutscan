# triBUUTSCAN

*3BUUTscan* — a 3D scanning toolkit for Linux, starting with the Creality CR-Scan Ferret Pro used
directly over USB, with no Creality software.

Made in 3BUUT. Free for everyone, see [License](#license).

## Why this exists

Until now our scanning workflow was: scan with an Android phone, then move the data to Linux and
finish in CrealityScan under Bottles and Wine. It works, but it is not a smooth environment: lots of
hand-offs, and the scanner itself never talks to the Linux machine.

Now the scanner stays connected to the PC, and we can simply move it around while it streams.
The preliminary results are pretty good. The logic is simple, so why not go further: the aim is to
build an entire scanning setup on Linux, step by step, in the open.

We are open to testing more devices and to improvements from anyone. See
[Compatibility](#compatibility) and the [roadmap](#todo--roadmap).

Status: **early days, and so far we are simply happy to have the data.** Capture works (colour, IR,
depth) and a single frame exports to a coloured point cloud. There is no scanning/fusion yet, and
the depth scale and camera calibration are unknown. This is only the beginning.

```
python3 -m tribuutscan.viewer        # live colour + IR + depth
```

Needs Python 3.10+, numpy and tkinter. No driver: the stock `uvcvideo` module handles the device.
Light the scene and uncover the lenses, or every stream is black. Press `s` in the viewer to save a
frame set (and a `cloud.ply`) to `captures/<timestamp>/`.

## 0-day case

The first real frame we saved, kept as-is: a glass kettle on the floor, scanner hand-held in front
of it (distance not measured). It is the starting point everything else gets compared against. Files are in
[`examples/0-day-kettle/`](examples/0-day-kettle/).

<p align="center">
  <img src="examples/0-day-kettle/color.png" width="420" alt="colour"><br>
  <sub>Colour</sub>
</p>
<p align="center">
  <img src="examples/0-day-kettle/ir.png" width="420" alt="IR with projector dots"><br>
  <sub>IR (projector dots)</sub>
</p>
<p align="center">
  <img src="examples/0-day-kettle/depth_preview.png" width="420" alt="raw depth, near = red"><br>
  <sub>Depth (raw, near = red)</sub>
</p>

What it shows, honestly:

- Depth comes out clean and smooth, but only for the plastic handle and the lid ring, about 9% of
  the frame. Our working explanation is the default ~17-32 cm depth window (the handle is the
  nearest surface); the glass body gives no depth, as expected for a clear surface. The window is a
  hypothesis from wall measurements, not yet confirmed.
- The resulting `cloud.ply` has 93,094 points. Shapes are right; dimensions are not trustworthy yet
  (the depth law and camera field of view are estimates, see below).

Regenerate the cloud from the saved frames: `python3 -m tribuutscan.pointcloud examples/0-day-kettle`

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
  See `tribuutscan/depth.py`. Raw values are ~1..200, nearer = larger. 0 = invalid.
- **IR is mirrored** horizontally relative to colour and depth.
- ffmpeg/GStreamer/OBS-style apps can't negotiate the depth format, and camera apps may pick that
  node and show black. `tribuutscan/v4l2.py` talks to V4L2 directly (raw ioctl + mmap, 64-bit offsets).
- Some depth frames arrive truncated; drop any frame shorter than `sizeimage`.
- The Orbbec SDK (`pyorbbecsdk2` 2.9.3) does not enumerate the device ("No matched usb device
  found"), although its udev rules list `2bc5:069c`.
- **Colour and depth are not registered 1:1.** Different cameras: depth looks magnified about
  1.13x horizontally vs the 1280x800 colour image, offset ~-31 px, with ~5 px vertical shift, and
  the leftmost ~17% of the depth frame is empty. `tribuutscan/pointcloud.py` carries a rough affine
  estimate (`REG`) read off box edges in a single frame; an overlay check looked right on a flat
  shelf, but it is not a calibration (parallax varies with depth).
- Only ~10% of depth pixels are valid in a test scene (plastic handle yes; glass, box, floor no).
  Cause not yet known.

## Depth versus distance (hand-measured, flat matte wall, `python3 -m tribuutscan.measure <cm>`)

| cm | centre median | centre valid |
|---|---|---|
| 20 | 590 | 88% |
| 25 | 251 | 92% |
| 30 | 66 | 90% |
| 35 | (none) | 0.1% |

Value falls steeply with distance and reaches zero near 31-33 cm: a disparity-like stream relative
to a reference plane, in a default near-range window of roughly 17-32 cm (15 cm extrapolates past
the 10-bit ceiling; not measured). Distances were held by hand (about +-1-2 cm), so the exact law
is not settled: `a/z + b` and `a/z^2 + b` both fit three points. This explains the sparse kettle
scene: only the nearest surface was inside the window.

## Open questions

1. Exact depth law and units. Needs a fixed rig, not a hand-held reading.
2. Where do the camera intrinsics / depth-to-colour alignment come from? Plan: capture the USB
   traffic from Creality's app (usbmon, needs sudo).
3. Why is depth sparse, and can a full-range mode be enabled?
4. Fusion: depth + colour -> point cloud -> Open3D / RTAB-Map -> mesh.

## Layout

- `tribuutscan/v4l2.py` capture and node discovery
- `tribuutscan/depth.py` depth decode (and encode, for tests)
- `tribuutscan/viewer.py` live viewer; `s` / Save writes `captures/<stamp>/` (colour, IR, depth, npz)
- `tribuutscan/frames.py` grab and save a 1280x800 colour+IR+depth set
- `tribuutscan/pointcloud.py` rough .ply export (guessed intrinsics, fitted depth law, estimated
  registration): `python3 -m tribuutscan.pointcloud [captures/<stamp>]`
- `tribuutscan/measure.py` depth-vs-distance readings
- `tribuutscan-viewer.sh` launcher (copy to `~/.local/bin/tribuutscan-viewer`)
- `examples/0-day-kettle/` the first saved frame set, with its point cloud
- `tests/` run with pytest, or call the test functions directly

## TODO / roadmap

Next up:

- [ ] **USB capture**: record the traffic between Creality's app and the scanner (usbmon, under
      Wine/Bottles or Android) to learn the init commands; they may widen the 17-32 cm depth window.
- [ ] **Calibration**: depth law and units, camera intrinsics, proper colour-to-depth registration
      (replace the eyeballed `REG`), using a fixed rig instead of hand-held readings.
- [ ] Multi-view fusion: burst capture, ICP alignment (Open3D), merged cloud.
- [ ] Why only ~10% of depth is valid in some scenes beyond the range cutoff (glass, dark surfaces).

Open for everyone:

- [ ] **More device tests**: other Creality scanners, other Orbbec-vendor devices (see Compatibility).
- [ ] **An entire scanning setup**: capture, calibration, fusion, meshing and export in one workflow.

Later, once the above works:

- [ ] **Geometry**: cleanup, meshing, scale-correct export.
- [ ] **Markers**: for calibration and for scanning (marker-based tracking and alignment).
- [ ] **Texturing** from the colour camera.
- [ ] **A better interface**, to be built in VS Code (the current viewer is a plain tkinter window).
- [ ] Proper tests (pytest), packaging, other devices (see Compatibility).

## Compatibility

Tested on exactly one setup: **Creality CR-Scan Ferret Pro**, Ubuntu (GNOME, Wayland), kernel 7.0.

Possibly related, **untested**:

- The original **CR-Scan Ferret**. The Pro identifies itself over USB with the same product string
  ("CR-Scan Ferret"), so the two may behave alike. Not verified.
- Other Creality scanners, and other Orbbec-vendor (`2bc5`) devices. Nothing here suggests they work.

Want to help? Plug your scanner in and send the output of `lsusb` and
`for d in /sys/class/video4linux/video*; do echo "$d $(cat $d/name)"; done`, then try
`python3 -m tribuutscan.viewer` and tell us which streams you get.

## Alternatives

As of October 2026 we found no other Linux support for this scanner: Creality ships no Linux
scanning software, and the Orbbec SDK we tried (`pyorbbecsdk2` 2.9.3) does not recognise it. If you
know of one, please open an issue. Until then, the practical options are Creality's Android app or
CrealityScan under Wine/Bottles for actual scanning; this project covers raw capture, depth and
rough point clouds only.

## License

MIT, see [LICENSE](LICENSE). Use it for anything, no strings attached beyond keeping the copyright
notice. Made in 3BUUT.

This is an independent project, not affiliated with or endorsed by Creality or Orbbec. Product
names belong to their owners.

This project was made with the help of Claude (Claude Sonnet 5.5, through Claude Code), using a
Claude account provided by 3BUUT.
