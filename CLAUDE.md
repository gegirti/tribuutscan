# CLAUDE.md

Working notes for Claude Code sessions in this repo. The README has the public write-up; this file
has the working context and ground rules; `docs/history.md` has the dated log of how everything got
here (measurements, what was tried and dropped, every scan looked at). This file is public, so keep it
free of anything private.

## What this is

triBUUTSCAN (also written 3BUUTscan): a Linux toolkit for the Creality CR-Scan Ferret Pro, used
directly over USB through the stock `uvcvideo` driver. Made in 3BUUT, MIT licensed. Status (2026-11-05):
live scanning works end to end on one scanner. Tracking runs on the GPU (WebGPU, any vendor, CPU
fallback), objects can be scanned hand-held or turned on a platform, and Stop gives a cloud, a mesh,
a closed STL for printing and a coloured solid for Blender. Not done: accuracy against a printed test
block, a long-range board calibration, 30 fps.

## Ground rules

- **Commits**: author and committer are `gegirti <47884926+gegirti@users.noreply.github.com>`
  (GitHub no-reply). The repo-local git config already has this. The global git config holds a
  personal email: never use it, never put it in commits, files or the README. Keep the
  `Co-Authored-By: Claude ...` trailer on commits.
- **Ask before anything outward-facing**: pushing, publishing, changing repo visibility, creating
  releases or issues. The repo is public at https://github.com/gegirti/tribuutscan.
- **Be honest about what is estimated.** Calibrated on 2026-10-30 (captures/calibration.json, on
  this one scanner): depth intrinsics and lens distortion, the depth law, the colour camera, and
  colour <-> depth registration. The factory calibration (captures/factory.json, the scanner's own,
  used when there is no calibration.json) is checked on this unit only. Still estimates: the near
  end of the board calibration's depth window (17 cm, never reached during calibration), depth
  noise, the long-range law when this scanner's disparity offsets were never read (LONG_B_SHIFT, from
  two units), and anything computed with neither calibration (the ESTIMATES block in calib.py: 60
  degree FOV, eyeballed REG). Say "estimate" for those.
- Keep captures out of git: `captures/` is ignored (it holds photos of the workspace). Only
  `examples/` is meant to be public; check an image before adding it.
- Match the existing style: small numpy-only modules, short docstrings, no heavy dependencies.
  The viewer and capture: Python 3.10+, numpy and tkinter only, on the system python3. Scanning,
  calibration and meshing: `.venv`, built with uv (`uv sync --all-extras` from uv.lock; Open3D, OpenCV,
  wgpu). The browser view (`tribuutscan/web/`) uses a vendored copy of three.js r160 (MIT,
  `THREE-LICENSE`) so it works offline; the server is Python stdlib. No pytest installed on the dev
  machine; the tests are plain functions and can be called directly.
- Measure before and after. A change to tracking or meshing is checked with evaluate.py against a run
  of the code before it (identical where nothing should change), and speed only with runs made back to
  back (the desktop is busy and timings swing +-15%). Heavy jobs run in a capped scope (Pitfalls).

## Device facts (verified on one Ferret Pro, Ubuntu GNOME/Wayland)

- USB `2bc5:069c` (Orbbec vendor ID; strings "Creality" / "CR-Scan Ferret"). USB 3.
- Three UVC functions: `/dev/video0` depth (Y10, 1280x800), `/dev/video2` IR (GREY, 1280x800,
  shows the projector dot pattern, mirrored vs colour/depth), `/dev/video4` colour (YUYV/MJPG).
  Odd nodes are metadata. `tribuutscan/v4l2.py:discover()` finds them by USB interface.
- Depth packing is an MSB-first 10-bit bitstream, 4 pixels per 5 bytes (not MIPI Y10P).
  Values seen: about 280 at 25 cm, about 3-55 at 30-31 cm (window edge); nearer = larger, 0 = invalid.
- 14-bit depth (found 2026-10-31): the depth function's descriptors list Y14 (format 1), Y10 (2) and
  Y8 (3); uvcvideo drops Y14 ("Unknown video format 20343159-..."). Y14 is the same MSB-first
  bitstream, 4 px per 7 bytes, 1,792,000 bytes per frame; Y10 = Y14 >> 4 (fit Y14 = 15.996 Y10 + 7.8),
  same window. `usbdepth.py` streams it through usbfs (stdlib: fcntl + ctypes; the URB pointer
  ioctls go through libc because fcntl.ioctl takes only a C int). Measured: 30 fps with colour + IR on
  V4L2 at the same time, 0 bad frames, ~10% of one core, 8 URBs of 12288 bytes (the device refuses
  whole-frame payloads; 64 URBs since 2026-11-01, see GARBLED ROWS). Payload headers are 12 bytes
  with a PTS (1 MHz); stamps from it are exactly 33.32 ms apart and pair like V4L2 (colour - depth
  -6.7 ms USB vs -6.8 ms V4L2). `decode_depth` puts Y14 on the Y10 scale ((Y14 - 8) / 16, float32; < 16 -> 0), so the calibrated law and every
  consumer are unchanged; recordings tell the formats apart by `depth_bytesperline` (1600 / 2240).
  Needs udev/70-tribuutscan.rules; otherwise, or with TRIBUUTSCAN_DEPTH=y10, depth is 10-bit V4L2.
  After a kill -9 the scanner's own reset gave /dev/video0 back in 6 s.
  Flat taped paper (measure.py, 2026-10-31), plane RMS Y10 / Y14: 0.144 / 0.138 mm at 20 cm,
  0.227 / 0.218 at 25 cm, 0.356 / 0.345 at 30 cm. The Y10 excess is exactly its rounding (step /
  sqrt 12 added in quadrature), so Y14 removes it, but the sensor's own per-frame unevenness
  (0.14-0.35 mm, growing with distance) dominates: 3-4% better on a flat face. That unevenness is
  nearly fixed over time (Y14 frame-to-frame 0.015-0.07 mm; Y10 shows 0, below one step). Not
  measured yet: sloped smooth surfaces, where Y10 steps (0.14-0.31 mm) make terraces and bad normals. The depth function also has
  a vendor extension unit (unit 4, 17 controls), not explored: a lead for the depth window.
- ffmpeg and GStreamer cannot negotiate the depth format, and camera apps pick it up and show
  black. Capture goes through raw V4L2 ioctl + mmap (struct offsets assume 64-bit).
- Some depth frames arrive truncated; drop any frame shorter than `sizeimage`.
- GARBLED ROWS (found 2026-11-01, a line across the user's hand capture): frames of the right size in
  which a run of 1-27 rows is not depth. Down each column the values step by exactly 64 raw units
  (Y10 914, 850, 786, ... wrapping at 1024; Y14 by 64 or 1024), sometimes beside a smooth ramp across
  the row; the object's own pixels in such rows often read 0. Both paths: the
  10-30 marker recording (V4L2 Y10) has them in ~50 of 581 frames, and they were merged as object
  (streaks in labels, sheets in cloud and mesh). Over USB with 8 URBs: 2 in 1500 frames idle, 100 in
  700 with a CPU-heavy job beside it. The scanner does it when the host is slow to take its data:
  under the same load (60 s, two runs each) 8 URBs gave 169-170 garbled and 693-949 broken frames
  (16-19 fps), 64 URBs 0-1 and 67-73 (29.5 fps); 128 / 256 no better. So `usbdepth.NURBS` = 64, and
  `depth.decode_depth` clears a garbled row and GARBLE_MARGIN 8 rows around it (`garbled_rows`: >= 15%
  of a row's row-to-row steps are nonzero multiples of 64, real rows 0-3%; the ramp kind is not caught,
  hence the margin; dropping the whole frame was measured worse). No hits on 614 synthetic frames.
  uvcvideo keeps only 5 URBs for bulk, so 10-bit V4L2 depth stays more exposed. The viewer logs cleared
  previews at close.
  COARSE 14-BIT DEPTH (2026-11-05): sparse, low-confidence 14-bit pixels come in whole 10-bit steps
  (values multiples of 64), often near the image's bottom edge, and rows of them looked garbled: 8041
  previews in a 20-minute viewer session on the dark-hand scene, 24 frames of the 11-04 test-block
  recording. Their flagged rows hold 26-88% such values, real 14-bit garbage 5-9% (chance). Steps
  between coarse pixels no longer count: every real hit in all recordings is still found (Y10 marker
  recording 44 of 44 frames, the four Y14 ones 5 of 5), the test-block false hits went 24 -> 1.
- Only one program can stream from the scanner at a time. Close the viewer before running
  `python3 -m tribuutscan.pointcloud` (live grab) or `tribuutscan.measure`.
- The scanner resets its USB connection 3-4 s after a program stops streaming: the kernel logs
  "USB disconnect" and it re-enumerates within a second (`journalctl -k | grep "usb 9-1"`). Seen about
  20 times on 2026-10-30; every one checked followed a viewer or capture closing. Harmless, but a
  program that opens the scanner within those seconds can lose it.
- The scene must be lit and the lenses uncovered, or all streams are black.
- `pyorbbecsdk2` 2.9.3 does not enumerate the device ("No matched usb device found").
- IR brightness is set through the depth node's standard V4L2 controls: exposure 100-6800 (default
  3000) and gain 1000-15000 (default 1000). The IR node's own exposure is read-only (EACCES). The colour
  node has the usual UVC controls (brightness, gain, white balance, auto exposure). There is no V4L2
  control for projector power. Helpers live in `v4l2.py` (`controls`, `get_control`, `set_control`).
  With 14-bit depth the depth node is gone and `usbdepth.DepthCapture` carries these two controls
  (camera terminal 1 exposure, processing unit 3 gain; same values). They only act while the IR
  node's auto exposure is 2: on 2026-10-31 it was 0, the exposure read back drifted (968-985) and
  setting it changed nothing, on both paths. With 2: exposure 500/1500/3000/6000 gave IR mean
  9.5/28.8/54.6/95.7, identical over USB and V4L2.
  AUTO IR (2026-11-05): 0 is the scanner's own auto exposure (what it powers up and resets to). Depth on
  the object (per frame / valid in 90% of 40 frames), dark hand at long range: scanner AE (it chose
  5000 / 1000) 51.6 / 49.2%, defaults 3000 / 1000 51.1 / 48.4, the old one-shot Auto IR's 1190 / 2700
  51.2 / 47.8, 800 / 1000 48.3 / 43.2. Test block at close range: it chose 581 and matched every manual
  setting (300-4500 all gave 2-6% per frame there: that scene's flicker is not exposure). So the
  viewer's Auto IR checkbox (on by default) leaves the IR node at 0 and the sliders show its values;
  moving an IR slider sets 2 (by hand). Applied again after a range switch or reconnect.
- One test with light boxes at about 25 cm in a dark room: depth stayed at about 72% valid from exposure
  800 to 3000. Clipping cost depth: gain 8000 gave 33% clipped pixels and only 47% valid. The
  projector lights the scene on its own. So the rule is to avoid IR clipping, not to make the image
  brighter. Dark objects are not tested yet.
- Markers: the standard is uncoded circular targets (white retroreflective dot in a black ring,
  e.g. 6/10 mm). `markers.py` finds them in the colour image (numpy only). On the test frames it
  found every marker that wasn't in shade, with about 1 false hit per 17 marker-free frames.
  Markers in shade on vertical faces (contrast ~40) are deliberately missed, because below ~60
  key labels, logos and glossy glints look the same. In IR they drown in the projector dot
  pattern. The live search in the viewer runs at half resolution, 5 Hz, and costs depth about
  19 -> 15 fps; there is a checkbox to turn it off.
- USB interface 6 (HID, `/dev/hidraw*`, root only) names itself "CR-Scan FerretIMU": the ICM-40608
  IMU, not the control channel. `imu.py` reads it over usbfs (same udev rule; usbhid is unbound from
  interface 6 only and rebound after). It sends only while switched on over the command channel
  (properties 2019 gyro / 2020 accel = 1; `vendor.set_property` allows only these two). Read on
  2026-10-31: 100 reports/s of 248 bytes, each 2 samples (200 Hz); header 01 07 18 02; groups of
  int16 id, accel xyz, gyro xyz, temperature + u64 device time, ~4992 units per sample (so us).
  Settings: gyro 200 Hz +-1000 deg/s, accel 200 Hz +-4 g (offered rates 100/200/500/1000 Hz).
  At rest: |a| 0.995 g, gravity on -y; gyro mean (-0.17, -0.89, -0.17) deg/s, i.e. a bias to calibrate
  (IMU calibration raw data 4036 not read yet). While the viewer streams 14-bit depth, /dev/video0 is
  gone, so commands must go through the viewer's own usbfs handle (interface 0) instead.
- Vendor commands go through UVC extension units (read 2026-10-31 with UVCIOC_CTRL_QUERY, GET only):
  - depth unit 4, GUID a55751a1-f3c5-4a5e-8d5a-6854b8fa2716 (Orbbec's common command XU; OpenCV's
    cap_obsensor uses the same GUID): selectors 1 / 2 / 3 = 512 / 64 / 1024 bytes, get+set, all
    zeros at rest; selectors 4-17 answer ENOENT although bmControls advertises 17.
  - IR unit 8 (GUID "ORBB...01"): 3 four-byte controls (1 reads 1, set allowed on 1 and 3).
  - colour unit 12 (GUID "ORBB...02"): advertises none.
  Protocol per OpenCV (not yet tried here): request = magic 47 4d, u16 size in half-words, u16
  opcode (1 get property, 2 set property, 3 get structure), u16 request id, then property id and
  value; sent with SET_CUR on selector 2, reply read from selector 2 (or 1 for big structures);
  reply bytes 8-9 = status (0 ok), data from byte 10. Candidate ids from OrbbecSDK_v2 Property.h:
  LDP 2, laser 3, min/max depth 22/23, depth precision level 75, current depth alg mode 1043.
- Read over that channel on 2026-10-31 (`vendor.py`, read opcodes only; replies valid, magic 0x4252):
  - current depth mode (structure 1043): "Close_range_mode Default" (16-byte checksum, 32-byte name,
    4 more bytes). So the near window is a named mode and others presumably exist.
  - sensors (1042): OV02K10 (colour), OV9282 (IR, global shutter), ICM-40608 (IMU); versions (1000)
    "1.2.6", "0.1", "1.0.2"; temperatures (1003) ~24 C; baseline structure (1002) floats 130.0, 7.5
  - properties: LDP 2 (0/1), laser 3 (0 while idle), 4 = 1627 of 100-6800 (default 2200, same range
    as IR exposure), depth precision level 75 = 1 of 0-1, LDP distance 100 = 0 of 0-3000; MIN/MAX
    DEPTH 22/23 unsupported; ~40 more ids answer (scan output not interpreted yet)
  - depth mode list (4034 = OB_RAW_DATA_DEPTH_ALG_MODE_LIST, read with the struct-list opcodes 29/30/31
    as OrbbecSDK_v2 does; 52-byte entries: 16-byte checksum, 32-byte name, u32 option code):
    Close_range_mode Default (0x8, ACTIVE), Long_range_mode Default (0x8), Close/Long_range_mode
    Interleave (0x18), Close/Long_range_mode Narrow_View (0x18), Grid_of_Dots Calibration (0xe),
    In_scene Calibration (0xa). The SDK switches by SET_STRUCTURE_V1_1 of 1043 with the chosen entry,
    only while nothing streams, then resets the device.
  - SWITCHED 2026-10-31 (`vendor.py switch "Long_range_mode Default"`): accepted, the scanner stayed on
    USB, the mode reads back and survives stream start/stop. Depth formats unchanged (Y14/Y10, no RLE).
    Same flat surface (close mode raw 93.4 = 299.5 mm): long mode raw 564.7; areas that were beyond
    the close window now have depth. IF long mode keeps a = 289570, b_long = 402.1 and the window is
    ~203-718 mm (raw 1023..1). Tested at a hand-measured 50 cm: centre raw 175.7 (predicted 177), i.e.
    501 mm. So long mode is z = 289570 / (v + 402.1), ~20-72 cm. Close mode at that 50 cm: 0% valid.
    SETTLED 2026-11-01 by the factory calibration: disparity offsets 218 (close) / 100 (long), so
    b_long = b - 472 exactly (401.4 with the board law; 175.7 then reads 501.8 mm).
  - The toolkit knows the modes (2026-10-31): `frames.open_depth` reads the active mode and calls
    `calib.use_mode`; `calib.Camera.set_mode` takes a "modes" block if present (a board calibration
    gets one from factory.json's disparity offsets, a factory spec has its own), else long range =
    close law with b + LONG_B_SHIFT (-472), window raw 1023..1, `law_estimate` True. Recordings, saved frames and session.json note "depth_mode"; `read_recording` switches the
    law from it (close when missing); the live worker gets it in its start command. calibrate.py tags
    views and refuses to solve non-close views until a long-mode block exists.
  - The viewer's Range menu switches Close <-> Long: streams pause (`Stream.pause/resume`), 6 s for the
    scanner's USB reset, `switch_depth_mode`, reopen. Checked: 6.4 s per switch, all streams back at
    30 fps, IR exposure control works on the new capture, long-mode centre read 49.3 cm at the user's
    50 cm. Synthetic evaluation unchanged (identical numbers before/after).
  - The mode is NOT kept across a full USB reset: killing a program that held the scanner caused a
    "USB disconnect" and the scanner came back in Close_range_mode Default. Ordinary stream stops and
    restarts kept the mode. Safe, since the mode is read at every open.
- WHY THE MODES (2026-10-31): the scanner is active stereo. Two IR cameras 75 mm apart (structure
  1002 reads 130.0, 7.5; the 7.5 is cm) see a projected dot pattern.
  - The calibrated law is raw = 4 x disparity - b: a / (fx x 75 mm) = 4.007, so a raw unit is a
    quarter pixel and a Y14 unit 1/64 px.
  - Each depth mode is the same 256-px disparity window (10 bits of quarter pixels) at another
    offset: close 218-474 px = 152-331 mm, long 100-356 px = 203-719 mm.
  - One side of the depth image has a blind strip as wide as the disparity: 241 px predicted at
    300 mm, 246 seen.
  - Specs (retail and review pages): working distance 150-700 mm; capture 820 x 560 mm at 700 mm
    (our model: 855 x 582); accuracy 0.1 mm; point distance 0.16 mm; 30 fps; object size 15 cm - 2 m.
    The 2 m is the object you walk around, not a distance: 2 m would need 36 px of disparity, and no
    mode goes below 100.
  - Depth step per raw unit grows with z^2: Y10 0.22 / 0.86 / 1.69 mm at 25 / 50 / 70 cm, Y14 / 16.
  - The depth function's 8-bit format (3, Y8) streamed all zeros in close mode (scene probably out
    of range): not a second IR image. The second camera is likely only exposed in a calibration
    mode (option bit 2 = right IR through the depth channel); not tried, since it is a write.
- PITFALL: GET_STRUCTURE 1044 got no reply (GET_CUR selector 1 timed out, -110), and ~30 s later
  the scanner dropped off USB ("device not accepting address", error -71) until it was
  physically replugged. vendor.py refuses 1044 and stops on any timeout.
- FACTORY CALIBRATION (read 2026-11-01 with the list opcodes 29/30/31, as OrbbecSDK_v2's
  G2AlgParamManager does; layouts from its InternalTypes.hpp, packed; raw bytes and factory.json in
  captures/factory/; the scanner stayed on USB). Lead from github.com/ssjrocks/ubuntu-ferret-scanner.
  - 4026 (OBDepthCalibrationParam, command version 1, 28 bytes): baseline 150.284 x unit 0.5 =
    75.14 mm, focalPix 960.51, dispOffset 218 (close mode; 100 in long, below), z0 662.16 (the far
    end in 0.5 mm). On the Y10 scale z = 4 unit baseline focal /
    (v + 4 dispOffset) = 288700 / (v + 872). Ours 289570 / (v + 873.4): 0.34-0.46 mm apart over the
    whole close window. Lateral scale a / fx 300.57 against ours 300.53.
  - 4027 (OBCameraParam_Internal_V0, 2 x 152 bytes; the 2nd is labelled 800x600, cy - 100): depth
    fx = fy 960.51, cx 646.30, cy 398.94, no distortion (rectified); colour fx 842.05 / 842.36,
    cx 637.13, cy 395.39, rational k1-k6 + p1 p2; colour = R depth + t (the SDK's convention,
    CoordinateUtil.cpp), t = (19.55, -0.03, -0.77) mm.
  - 4024 (OBD2CProfile, 22 bytes each): colour 1280x720, 1920x1080 (x1.5) and 640x360 (x0.5) use
    record 0 with 40 / 60 / 20 rows cropped top and bottom; colour 1280x800 has no record (255). So
    the colour model is 1280x720 inside a 1280x800 frame, and our 1280x800 colour stream is that
    image stretched by 10/9: y800 = (y_rec - 40 + 0.5) 10/9 - 0.5.
  - The factory frame is our depth AND colour streams mirrored horizontally (the IR stream's own
    orientation; rafik24's SDK fork turns on DEPTH_MIRROR 14 and COLOR_MIRROR 81, which explains its
    "vendor commands"). Tested on the 23 usable board views (797 corners): colour on depth 0.88 px
    median, p90 1.24, with both mirrored and R p + t; the 7 other combinations 16-360 px. Ours on the
    same views 0.34 px, but ours was fitted on them. A 24.00 mm square: factory 24.06 mm, ours 24.01.
  - Long range (read after a switch): dispOffset 100, z0 1443.50; everything else, 4027 and 4024
    byte-identical. Back in close range afterwards, the scanner stayed on USB.
  - USED since 2026-11-01. `calib.CAMERA.source` is "board" (calibration.json) > "factory"
    (captures/factory.json) > "estimate"; TRIBUUTSCAN_CAMERA=board|factory|estimate forces one.
    `frames.depth_mode` (every open, also after a range switch or reconnect) reads the lists with the
    mode and calls `calib.note_factory`: factory.json gets `calib.from_factory`'s spec in our frames
    (mirroring done there; the colour lens is the 8-coefficient rational model, which `calib.distort`
    now takes) plus "disp_offsets" for every mode read so far (reset when baseline/focal differ:
    another scanner). A failed read is logged and changes nothing. The viewer's start log names the
    source; saved sets and recordings note "calibration" in info.json.
  - Checked through `calib.Camera`: the same 0.88 px / 24.06 mm. On the scanner: close and long opens
    gave offsets {218, 100}; board long law 289570 / (v + 401.4), factory 288700 / (v + 872) and
    / (v + 400), none marked estimate. evaluate.py identical to before (board camera), 246 -> 250 ms.
  - Marker recording replayed with each camera (same code, compare SEEDS 12): board median 0.246 /
    p90 0.652 mm, 92.3% on the part, Creality surface covered 69 / 73% within 1 / 2 mm; factory 0.216
    / 0.515 mm, 92.4%, 65 / 68%. Creality's mesh comes from Creality's software, which presumably uses
    this same factory calibration, so a closer fit is expected and does not show the factory model is
    more accurate; the printed test box decides that.
  - Per-unit values differ: other units read depth fx 956.8 / 956.6, cx 645.6 / 638.9 (ssjrocks,
    rafik24; both firmware 1.2.2, ours 1.2.6).
  - Not read: IMU calibration 4036. The SDK reads it with the raw opcodes 17/18/19, and that sequence
    on 4027 knocked ssjrocks' unit off USB until a replug.
- CrealityScan (Windows, run under Bottles) ships a custom Orbbec SDK (OrbbecSDK.dll) with a
  `FerretUvcDevice` class: `switchDepthAlgMode`, `initDepthProcessParam`, and depth work modes whose
  names and checksums the SDK reads from the device. Its embedded config asks for depth 1280x800 in
  a format named "RLE" (not in our descriptors; maybe only after a mode switch), colour 1280x720
  MJPG, IR 1280x800 Y8, for both Ferret and Ferret Pro. Its algo.obconfig holds data for a stereo
  line-laser model, not the Ferret. Project files (resources.obscan) are SQLite with packed records
  (32-byte header, payload starting 40 a2; the app also ships aes256.dll): not decoded, on purpose.
  Under Wine the app never opened the scanner (its SDK logs show no device), so they hold no protocol.

## Commands

```
uv sync --all-extras                        # build .venv (Open3D, OpenCV, wgpu) from uv.lock
tribuutscan-viewer                          # launcher in ~/.local/bin, copy of tribuutscan-viewer.sh
python3 -m tribuutscan.viewer [--force-cpu] # depth (middle), colour + IR (left), panels (right, fold):
                                            # (--force-cpu: tracking and meshes on the CPU, GPU or not)
                                            # Scanner: Range close 15-33 / long ~20-72 cm (estimate),
                                            #   IR exposure/gain, colour brightness, Auto IR (the
                                            #   scanner's own, on by default; a slider: by hand)
                                            # Aim (idle): the object a scan would take is outlined
                                            #   (cyan); click depth or colour preview to pick another,
                                            #   right-click / Esc: the centre again; bar: distance,
                                            #   depth on it, IR clipping, base, markers
                                            # Capture: `s` saves captures/<stamp>/, `r` records colour +
                                            #   zlib depth + imu.csv to captures/<stamp>-rec/ at 30 or 10 fps
                                            #   (~75 / 25 MB/s, stops under 5 GB free), `m` meshes
                                            #   the latest recording (cloud.py --optimise --mesh)
                                            # Scan: `c` live scan after a 3-2-1 countdown (Esc
                                            #   cancels; Countdown box), Open 3D view
                                            # IMU: tilt horizon, turn rate; View: Find markers,
                                            #   Hold depth (calmer preview, held parts dimmer)
                                            # events: message line, console, captures/viewer.log
                                            # Status / Messages boxes under the previews (sysinfo.py)
python3 -m tribuutscan.vendor [modes|factory|scan|switch "<mode>"]   # depth modes, factory calibration
                                            # (switch WRITES); TRIBUUTSCAN_CAMERA=board|factory|estimate
python3 -m tribuutscan.imu [seconds]         # switch the IMU on, read it, switch it off
python3 -m tribuutscan.pointcloud [dir]     # .ply from a saved set, or grab a live one
python3 -m tribuutscan.measure [cm]         # flat target: plane RMS, noise, step -> captures/measure.csv
python3 -m tribuutscan.usbdepth [s] [--all] # 14-bit USB depth throughput check (--all: + colour, IR)
                                            # TRIBUUTSCAN_DEPTH=y10 forces 10-bit V4L2 depth anywhere
python3 -m tribuutscan.markers captures/<stamp>   # find markers in a saved frame -> markers.png
python3 -m tribuutscan.testblock [out_dir]       # the test block to print: captures/testblock/testblock.stl, .ply
.venv/bin/python -m tribuutscan.cloud captures/<stamp>-rec [small|medium|large] [--keep-base] [--force-cpu]
                                            # merged point cloud -> captures/<stamp>-scan/ (base removed);
                                            # tracks on the GPU when there is one (--force-cpu or
                                            # TRIBUUTSCAN_GPU=0: CPU; =cpu: WebGPU on llvmpipe)
python3 -m tribuutscan.view3d captures/<stamp>-scan    # browser 3D view at http://localhost:8765
.venv/bin/python -m tribuutscan.optimise captures/<stamp>-scan   # loop closures + pose graph + merge again
.venv/bin/python -m tribuutscan.session merge A-scan B-scan AB-scan   # merge two scans into one
.venv/bin/python -m tribuutscan.calibrate board|capture [square_mm]|solve <dir> [square_mm]
                                            # ChArUco calibration -> captures/calibration.json
.venv/bin/python -m tribuutscan.evaluate [datasets] [--tracker current|gpu|gpu-proc] [--label x] [--against report.json] [--optimise] [--mesh]
                                            # tracking metrics on saved + synthetic recordings (--mesh:
                                            # also fuse, score against the true surface; datasets in
                                            # evaluate.DATASETS, "hand", "markersgap", "handgap" among them)
.venv/bin/python -m tribuutscan.synth [out] [frames] [every] [hard] [--plate | --turntable] [--pattern MM] [--y14]
                                            # synthetic scan, ground truth poses + true_surface.ply
.venv/bin/python -m tribuutscan.bench [rec] [--frames N] [--tracker gpu] [--label x] [--against captures/bench/<run>.json]
                                            # per-stage timing table (warm, disk apart) -> captures/bench/
.venv/bin/python -m tribuutscan.bench --live [rec] [--frames N] [--tracker gpu|joint]
                                            # the live worker's fps on a recording (simulated camera)
.venv/bin/python -m tribuutscan.fusion captures/<stamp>-scan [--all] [--min-weight 4] [--force] [--open] [--force-cpu]
                                            # fuse a scan's keyframes -> mesh.ply (m, scanned surface) +
                                            # mesh.stl (mm, closed solid: solid.py; --open: as scanned)
                                            # + solid.ply (the solid with colour, metres: Blender)
                                            # (on the GPU when there is one; --force-cpu: Open3D)
.venv/bin/python -m tribuutscan.gpucheck [--frames N] [--out r.json]   # GPU self-check, no scanner
.venv/bin/python -m tribuutscan.bakeoff slice captures/<stamp>-scan [--seconds S] [--capacity N]
                                            # GPU slice timing / stability on a saved scan
.venv/bin/python -m tribuutscan.cloud captures/<stamp>-rec --optimise --mesh   # what the viewer's `m` runs
.venv/bin/python -m tribuutscan.showcase captures/<stamp>-scan [out]   # README GIFs: scan.gif (the raw copy
                                            # replayed, drawn with the viewer's previews) + model.gif (mesh.ply
                                            # turning, Open3D offscreen); needs ffmpeg. Check every frame first
.venv/bin/python -m tribuutscan.compare captures/<stamp>-scan [ref.ply] [--seeds 12] [--transform c.json]
                                            # score a cloud or mesh against a reference mesh (default
                                            # captures/reference/part/mesh.ply) -> compare[-<name>].json/.ply
```

Beware `pkill -f ...tribuutscan.viewer` inside a shell command: the pattern matches the shell
itself. Use `pgrep -f "^python3 -m tribuutscan.viewer"` and kill by PID.

## How a scan runs now (2026-11-04)

The constants named here live at the top of cloud.py, scene.py, gputrack.py, fusion.py and solid.py,
each with the measurement behind it; docs/history.md has the longer story.

Processes:
- **Viewer** (`viewer.py`, `panels.py`; system python3, numpy + tkinter):
  - Streams: colour and IR over V4L2, depth over usbfs (14-bit) when the udev rule allows, else 10-bit
    V4L2. The streams are paired by kernel capture time.
  - Previews at most 15 a second, the newest frame only, index-resampled (SIDE_W 400 / DEPTH_W 800).
    Outside a scan the depth preview holds each pixel's last depth 0.5 s while the IMU says still, 0.15 s
    while moving (`hold_depth`; display only): some surfaces lose whole regions from frame to frame.
  - The aim (`aim_loop`, 5 Hz while idle, ~18 ms each): `cloud.isolate_mask` on every 4th pixel, as the
    first scan frame does, outlined on depth and colour; a click picks a piece, followed by its 3D centre
    until Start hands its pixel to the scan (`Scanner(aim=...)`, noted as "aim" in the raw copy's
    info.json and session.json so replays take it). Start counts down 3 s (COUNTDOWN_S), the worker
    starting meanwhile. A click while scanning goes to the worker as a pick (step 5 below).
  - A range switch logs how long each step took; one still busy after SWITCH_STUCK_S 40 s is given up
    for a reconnect (one hung on 2026-11-05 after an IMU-off timeout, with no clue where).
  - While scanning, a `Feeder` thread (not the Tk tick) sends each new depth frame with its colour pair as
    soon as fewer than 2 are in flight and passes the worker's replies on; colour and IR previews drop to
    SCAN_SIDE_FPS 5 and the marker ring to 1 Hz. Live, scanner still (2026-11-05, back to back): HEAD
    13.1-13.4 fps with the worker waiting 22-23 ms a frame; the feeder alone 16.0 (6 ms); with the lighter
    previews 17.6-19.3 (1 ms). At Stop the viewer logs frames sent per second and its tick intervals.
  - Records colour + zlib depth + imu.csv. The IMU runs at 200 Hz while the viewer runs, through the USB
    depth capture's own handle.
  - The Status and Messages boxes are plain text (`sysinfo.py` reads /proc and DRM fdinfo).
  - One viewer at a time: captures/.viewer.lock, and USB claims that refuse an interface held by another
    program (DISCONNECT_CLAIM with EXCEPT_DRIVER "usbfs").
  - It reconnects after the scanner's own USB resets (App.reconnect: stops and saves a recording or scan,
    reopens with the mode read anew, the IMU back on). It asks Retry/Cancel while the scanner is unplugged
    at start.
  - `--force-cpu` (gpu.force_cpu) sets TRIBUUTSCAN_GPU=0 before anything starts, so the worker and
    meshing jobs inherit it.
  - Window 1491 x 971 with every panel open (fits 1920 x 1080 at Tk scaling 1.33). ("TkFixedFont", 9)
    is not monospace here; the boxes use a sized copy of the named font.
- **Live worker** (`livescan.py`): spawned with .venv's Python. It serves the 3D page on :8765 and exits
  when its viewer dies.
  - Frames arrive as raw depth bytes (`depth.LazyFrame`). The CPU decodes only every 4th pixel
    (`decode_strided`), and the GPU unpacks the full frame (`Engine.unpack`).
  - Two frames are in flight, and a frame is sent only when the worker is free.
  - Each reply carries the 1/8 object mask and the live-view codes. The raw copy ("keep raw frames") is
    written on its own thread.
- **Merge process** (`merger.py`, the worker's child): point-cloud merges and the keyframes' ORB, out of
  the tracking process's GIL. Every job carries the depth mode. Replays, evaluate and bench merge on a
  thread instead, with the same results bit for bit.
- **Meshing**: after Stop inside the worker ("Mesh after Stop"), or `m` (cloud.py --optimise --mesh in a
  capped systemd scope).

One frame (`cloud.Scanner.feed`, GPU tracker):
1. **Object piece.** The search runs on every 4th pixel, and the axial geometry is computed once per
   frame at the pose guess (`scene.axial`). The kept base is re-fitted once (`table_in_camera`). The
   source is the piece connected to the region's core where the guess puts it (`Scene.core`: the
   cylinder over the base, or the ball's start radius when no base is known), after the plane cut. The
   view-centre piece is used only before a region exists, or with fewer than MIN_SEEDS seeds; the first
   frame takes it at the viewer's aim when one was picked (then the scene prefers tracking's piece to
   the view centre's for the base vote and the first region).
2. **Guess.** The last motion, scaled by the time since the last frame, for at most GUESS_MAX_S 0.3 s.
3. **Solve.** The GPU runs point-to-plane ICP against a ray cast of its own tracking volume (gputrack:
   3 mm truncation; levels reach 30/12/6 mm with strides 4/2/1 and at most 3/3/8 passes). The CPU adds
   markers (sigma 1 mm) and the motion prior and runs the 6x6 Gauss-Newton (`track.gauss_newton`).
4. **Accepted** when all of these hold:
   - `solve_ok`: fitness, and an rmse within GPU_MAX_RMSE_MM 1.0 x (z / 450 mm)^2;
   - `plausible_step`;
   - `floor_agrees`: the frame's own table within 2.5 degrees of the kept base, judged only on patches
     at least 150 mm across.

   Two more checks can change an accepted pose:
   - after a gap of GAP_S 0.5 s, a shape candidate (`shape_candidate`) replaces the solve if it fits
     clearly better elsewhere;
   - with 3 or more markers in view and none used, `marker_check` places the frame by markers;
   - a solve failed only by one or two markers, on a shape that pins the pose (degeneracy >= 4 x
     track.DEGENERATE), is solved again without them (`odd_markers`; a spurious marker on a printed
     pattern, 2026-11-05).
5. **Lost.** First the frame's piece is judged against the object's look (`likeness`; `look`: size r90,
   points scaled by distance^2 and distance of the look piece's last LOOK_FRAMES 30 tracked sources, the
   look piece being the first, or the latest one a click started): "object" (r90 x0.5-2, points >= 0.35,
   distance within max(30%, 100 mm)), "other" (points < 0.1, r90 outside x1/3-3, or the distance off by
   twice that) or "unsure". An "other" frame (looking away: the shelf, a wall, crumbs) tries nothing and
   the viewer says the object is out of view. Otherwise it tries, in this order:
   - `relock`: Open3D ICP from the last pose against the object cloud, gated, then the GPU solve's checks;
     not once this loss has shown something else (`relock_due`, `away_frames`: the scanner may come back
     anywhere, and on the CPU tracker a relock 4 s after a look-away took a wrong pose nearby; a 1 s time
     limit instead cost the CPU tracker half of markersgap, whose frames are only missing). After
     AWAY_FRAMES 5 such frames (it looked away) the CPU tracker's own step from the last pose is not tried
     either (`step_due`: it accepts at fitness 0.3 / 2 mm and took a frame 14 degrees off): a new piece;
   - `refind_anywhere`: ORB against keyframes, with inliers away from the object and the pose reachable
     in the time lost;
   - `refind_shape`: FPFH + RANSAC against the object so far, at most every 0.25 s, refusing edge-on
     views.

   After a pose found anywhere, REFIND_HOLD frames must track before merging resumes. While lost, the
   viewer shows the model from the last pose (the "ghost"). While lost, the source piece is seeded by the
   object's colours instead of the region's core when the colour model can tell them (`colour.py`:
   U/V histograms of object vs the rest of the view, learned at keyframes; silent when they overlap,
   Bhattacharyya > 0.6, so depth alone decides as before).
   **Pieces.** After PIECE_AFTER_S 1 s of lost frames whose piece may be the object ("object" or "unsure"; an
   "other" frame, or one with too little depth, starts the clock again), with every recovery failing, the scan
   goes on in a new piece (`start_piece`): pose chain, scene, marker map, live cloud and tracking volume start
   afresh (the volume emptied in place, `Engine.reset`: no second ~560 MB engine), at the last good pose; its
   first frame takes the object by colour, else at the point nearest where the lost frames saw it
   (`like_source`), else nearest the last source. Before 2026-11-06 any frame with depth counted: looking
   away started pieces on the shelf and the wall, each taken where the last had its object, until one took
   the boxes behind the hand. ("unsure" must count: seen from above, devplatform's hand showed a third of
   its points, and with only "object" counting the CPU tracker never began a piece.) Re-finds look only within the current piece. `pieces` holds each one's first index, like `sources`.
   **A click** on the depth or colour preview while scanning (`Scanner.pick`, sent as ("pick", kind, (u, v));
   the raw copy notes it in picks.csv, `recording_picks` gives replays the same): on the object being
   tracked (the last frame's labels) nothing changes; on something else a new piece starts there at once;
   while lost, that frame's piece is taken at the click and found again in the scan if it can be, else a
   new piece starts there (no 1 s wait). A piece a click started becomes the look piece.
6. **Scene** (`scene.py`, mode "still"):
   - The base is found by votes: 3 must agree in world coordinates. `stands_on` requires contact within
     10 mm, and at most half the plane may lie inside the object's outline. After VOTE_LIMIT, while no
     base is known, a vote is taken on every 10th frame, except for VOTE_WINDOW 100 frames after the view
     turns VOTE_NEW_VIEW_DEG 25 degrees from every view such a window began at (a scan level for minutes,
     then a pass from 45 degrees: the dev scan of 2026-11-06 got 4-6 votes in 1000 frames before this).
   - The region grows only toward points GROW_ABOVE_MM 10 above the base, seen in the middle half of the
     view.
   - The kept plane is re-fitted only to points outside the object's footprint, and a re-fit that turns
     more than 3 degrees or moves more than 5 mm is not taken.
7. **Platform** (`check_room`, once a second at keyframes): if the room is found again where it was,
   that counts as "still"; 3 "still" verdicts set `object_turns`. From then on, ORB and Optimise's
   features take only the object's pixels, and the viewer shows "object turning, scanner standing". The
   check stops after ROOM_SETTLED 5 "moved" verdicts with no "still".
8. **Fusion and merging.** Every keyframe (3 mm / 2 degrees of motion) is fused into the GPU tracking
   volume. The point cloud, and Optimise's keyframes with their ORB, take one at most every
   MERGE_MIN_S 0.25 s. Capping the volume's fusion too made synth's ATE 0.35 -> 2.6 mm.

After Stop:
- **Join** (`optimise.join_pieces`, before the save, when there are pieces): the first piece's world is
  the scan's (where it was aimed and found its base; the biggest piece's if the first is too small); each
  other piece is placed by shared markers, ORB between keyframes, or shape (FPFH, JOIN_SHAPE_SEEDS 2 RANSAC
  seeds), each candidate refined by `track.solve` (`join_solve`) and taken only at fitness >= 0.6 and rmse
  <= 1.0 mm (JOIN_*; a weak shape join at 0.43 / 1.44 mm doubled the dev scene's hand) with markers or a
  pinned shape, one way or the other: a piece that saw much more than the others fails onto them (the
  2026-11-06 45 degree pass: 0.306 / 1.06 mm) while they lie on it (0.981 / 0.77 mm). Passing candidates
  more than JOIN_AGREE 10 mm / 5 degrees apart leave the piece out (`join_pick`: a view that fits in two
  places). A joined piece moves with all its frames; one that joins nothing, or has
  fewer than PIECE_MIN_KEYFRAMES 5 keyframes, is left out (saved as piece-N.ply; its frames "left out").
  When the first piece never found a base and a joined one did, the scan takes that piece's base and
  region (`Scene.adopt_base`; level at long range the board gives no depth, from 45 degrees it does).
  Optimise then adds no odometry edge across a piece boundary, a certain edge per join (the keyframe pair
  nearest each other once placed), and finds loops across pieces as within one; remerge blends nothing
  across a boundary. session.json keeps "piece_list" and "piece_joins".
- **Base onto a support** (`Scanner.settle_base` -> `Scene.lift_base`, after the join): the base moves up
  onto a flat layer parallel to it that the object stands on, when the layer is wide (>= 100 cm2, >= 10% of
  the object's points), what stands on it covers at most a quarter of it (cells of 3+ points), and it is
  thin compared with that object (<= 1/3 of its height). The layer's own plane is fitted (`_layer_plane`,
  within 5 degrees of the base): a board need not lie parallel to the platform, and a base voted 1.6 degrees
  off spread the board over 19-27 mm. The dev scene: the platform was voted, the board 24.9 mm above it
  became the base; mesh 1.15M -> 483k triangles, the hand and its stand, flat bottom.
  `lifted_mm` keeps how far: with Keep base everything above the voted base still shows.
  Optimise (also "Optimise again" in the 3D view) settles it too, so scans saved before get it.
- **Save**: cloud.ply, session.json, keyframes.npz, merged.npz, frame_poses.txt, labels.npz, track.csv,
  markers.json, cloud_state.npz.
- **Optimise** (`optimise.py`): loops from shared markers, ORB matches or nearby poses. Loops the shape
  cannot pin are refused unless markers held them. Then an Open3D pose graph and a re-merge. The
  message gives the largest correction at the object.
- **Mesh** (`fusion.py`):
  - It fuses the merged keyframes at Optimise's poses: 0.5 mm voxels, 1.5 mm truncation, minimum weight
    4, each frame masked by its labels. It runs on the GPU when there is one, else on Open3D's
    VoxelBlockGrid.
  - Marching cubes on the GPU writes into one 128 MB buffer (WebGPU's binding limit), which holds 2.2
    million triangles. The base and the surroundings are cut only after extraction, so a long scan can
    produce more than that before the cut. The 1613-frame platform scan of 11-03 14:48 did, and got no
    mesh. Since Phase 6, `Engine.mesh` runs over the bricks in halves until each part fits, and
    `assemble_mesh` joins the parts into the same mesh.
  - At extraction, the cloud's `output_mask` is applied to the vertices, and crumbs under 1% are dropped.
  - Outputs:
    - mesh.ply: the scanned surface, in metres;
    - mesh.stl (mm): closed by `solid.close`. That is screened Poisson, a cut and cap at each printing
      plane (the base, else square to the platform's turn axis, plus every crop face the mesh reaches),
      fan fills, weld and repair;
    - solid.ply: the closed solid with colour, for Blender.
- **3D page** (`view3d.py` + `web/`):
  - Process: Optimise again, Merge again, Make mesh, Remove/Keep base, Crop.
  - Show: cloud / mesh / solid. Export: every file above.
  - Crop box: its axes are level with the base, or the turn axis. `Scanner.crop` is part of
    `output_mask`.
- `Scanner.restore` takes the scan's depth mode, and continued or merged scans drop the crop.

**The CPU path.** It runs when there is no WebGPU adapter, or with --force-cpu / TRIBUUTSCAN_GPU=0:
- tracking is `track.solve` against Open3D levels of the cloud so far (evaluate's "current"), with ORB
  on every frame and the object search on every 2nd pixel;
- fusion runs on Open3D's VoxelBlockGrid.

TRIBUUTSCAN_GPU=cpu runs the WebGPU engine on llvmpipe instead. Without Open3D at all, frames are
merged unmoved (smeared).

Invariants:
- Replays of the raw copy reproduce live poses: the model rebuild and relock wait for pending merges.
- `cloud.py` writes next to the recording: replay a scan's own -rec through a symlink elsewhere.
- Synthetic sets always run on the estimated pinhole camera (`calib.use(calib.ESTIMATES)`).
- A scan is meshed only under the camera it was tracked with (session.json "camera"; --force overrides).

## Measuring

- **`evaluate.py`** runs a tracker over these datasets:
  - real: markers, hand, platform, platform2, devplatform (the dev scene, 2026-11-05), devaway (the dev
    scene, 2026-11-06: still 88 s, two look-aways, a 45 degree pass);
  - synthetic: synth, synth7, synthhard, synthfp, synthplate, synthturn;
  - gap variants: markersgap, handgap, synthturngap.

  Synthetic sets regenerate byte for byte when missing. The trackers are "current" (the CPU), "gpu",
  and "gpu-proc" (the live worker's merge process; must equal "gpu"). The columns are:
  - tracked, used (tracked of the frames that had depth), pieces and joined (scan pieces, U4),
    jumps (> 30 mm / 15 degrees, within a piece);
  - doubled: a same-facing sheet 2.5-8 mm along the normal;
  - marker scatter, keyframe scatter;
  - ATE / truth mm (synthetic), region mm;
  - Optimise's corrections ("corr mm" at the camera, "corr obj" at the object);
  - ms a frame.

  `--against` shows before -> after.
- **`bench.py`** gives a per-stage table, and `bench --live` the real worker's fps on a recording (frames
  as the scanner sends them).
- **`compare.py`** scores a cloud or mesh against Creality's mesh of the marker part. Use 12 seeds, and
  reuse a transform for clouds that share tracking.
- **Reference runs** (2026-11-06, all 15 datasets, `--optimise`): captures/eval/20261106-153054-g-final3-gpu
  and -153054-g-final3-cpu (the look-away plan). The code before it: 20261106-110005-g-before-gpu and
  -110839-g-before-cpu (HEAD ab893a8 with relock's no-normals guard, which HEAD's CPU tracker needed on devaway).
  Before them: captures/eval/20261105-203248-f-gpu and -203250-f-cpu, with 20261106-083850-joingate-cpu for
  devplatform and handgap after the join gate; (scan UX) 20261105-123705-u34-gpu, -122856-u34-cpu,
  -130551-u34b-gpu, -130327-u34b-cpu.
  Before them (2026-11-04, Phase 6): captures/eval/20261104-101245-p6-after-gpu
  (`--optimise`) and -101700-p6-after-cpu. The marker part's mesh lies 0.08-0.09 mm (median) / 0.33-0.35 mm
  (p90) from Creality's. Hand-held live: 20 fps through `bench --live` on hand2.
- **Datasets** live in captures/ (not in git). The hand and platform recordings are the user's;
  captures/reference/part/mesh.ply is Creality's scan of the marker part, itself a scan and not ground
  truth.

## Reference part

captures/reference/part/mesh.ply is CrealityScan's final mesh of the marker test part. It came from the
Android app, 4 scans merged, on 2026-10-30, in mm; the part is about 274 x 192 x 44 mm, the same part
as captures/20261030-194411-scan. It is a scan too, not ground truth.

The early "rim error" and the free scale of 0.970 were table points counted as part (corrected
2026-11-01). With the base left out, the cloud lies 0.22 mm (median) / 0.53 mm (p90) from it. The
scale question waits for the printed test block. compare.py's alignment (FPFH + RANSAC) varies between
point sets: use `compare.SEEDS = 12`, and the same transform for clouds that share tracking.

## Pitfalls already hit

- Open3D (0.20, in the gitignored `.venv`): the legacy `ScalableTSDFVolume` gives empty meshes on
  this data. The tensor `VoxelBlockGrid` works, with these catches:
  - its extraction keeps weights STRICTLY above the threshold (3 meant "seen in 4 frames"; fusion.py
    passes n - 0.5);
  - float depth needs float colour (or uint16 depth with uint8 colour);
  - a frame touching no block raises ("No block is touched"), so empty frames are skipped;
  - its integrate takes each voxel's depth from the pixel ROUNDED DOWN: half a pixel off along slopes.
    The GPU engine takes the nearest pixel; `nearest=False` keeps Open3D's rule for the parity checks.
- wgpu-py: the device and queue form a reference cycle, so a GPU engine is freed only by the cycle
  collector (a replay loop held 3.9 GB). Call `Engine.close()`; destroy() alone frees nothing.
  wgpu-native hands Vulkan timestamps as raw ticks (A750 ~54 ns): `calibrate_ticks`.
- The v4l2_format union starts at byte 8 on 64-bit; getting this wrong silently ignores the
  requested format.
- `np.load` of an .npz decompresses the whole array again on EVERY `f["name"]`, and a slice of it
  keeps that copy alive. `Scanner.restore` did this per merged frame (since P6): on the marker scan
  (207 merged frames, 12.2M points) it grew past 8 GB. Read each array once (`cloud.load_npz`):
  Optimise now peaks at 1.3 GB.
- That 8 GB process ran inside VS Code's systemd scope, so the OOM killer's kill made systemd stop the
  whole scope (OOMPolicy=stop) and VS Code died with it. Start heavy jobs from a session in their own
  capped scope: `systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G <cmd>` (it keeps
  the working directory and environment). Terminals no longer activate `.venv` (VS Code's Python
  terminal activation turned off, 2026-11-06), so `python3` is the system one: call `.venv/bin/python`
  for scanning code. A shell started before that may still have it active: check `$VIRTUAL_ENV`.
- Spawned processes (the live worker, the merge process):
  - A spawned process inherits the parent's `sys.path`. The worker puts its own site-packages first
    (livescan.run), but spawn re-imports the viewer module (numpy at the top) before that. So the worker
    runs on the SYSTEM numpy, with only Open3D, OpenCV and wgpu from .venv.
    - .venv must therefore resolve for the system numpy's major version (pyproject `[tool.uv]`:
      numpy<2 below Python 3.13; the lock picks OpenCV 4.11).
    - A numpy 2 worker would also pickle arrays the viewer's numpy 1.26 cannot load.
  - Spawned children re-run the starting script: test drivers need an `if __name__ == "__main__"`
    guard, and a script piped on stdin cannot spawn at all.
  - The live worker is daemonic, and multiprocessing refuses daemonic processes children. The merge
    process lifts the flag around its start only (`_config["daemon"]`), and falls back to a thread.
  - An orphaned worker once held port 8765: the worker exits when its viewer dies.
- uv: plain `uv sync` removes the extras; use `uv sync --all-extras`. `python-preference = only-system`
  keeps .venv on the system interpreter the viewer runs on. The old pip-built venv was deleted on
  2026-11-06, after live scans had run on the uv one.
- Automatic colour/depth registration by edge correlation did not converge on a flat shelf scene
  of dark boxes; a calibration target or marker is the right way, not more tuning.
- Hand-held distance readings (about +-1-2 cm) cannot tell `a/z+b` from `a/z^2+b`. A fixed rig is
  needed for the real depth law.

## Open (2026-11-06)

The GPU plan (approved 2026-11-01) is done through Phase 6 (clean-up), except for what needs the user or
was left to them. The scan UX plan (approved 2026-11-05): U1-U4 coded and measured on the datasets
(docs/history.md); U0 partly recorded. The look-away plan (approved 2026-11-06, ~/.claude/plans/virtual-
watching-kettle.md: pieces only on the object, a click while scanning, joins both ways, the base from a
joined piece or a new view) is coded and measured on the datasets; its live checks need the user.

Needs the user and the scanner:
- **Scan UX, U0 and live:** the dev scene (the dark hand on the board on the white platform, long range,
  two more hands at the left, the shelf behind) recorded with keep raw frames. Done: D1 platform turn
  (devplatform), D5 still + D4 looking away twice, back from the same side + D2 ~45 degrees down, in one
  recording (devaway). Open: D3 hand-held circle, D4 coming back from another side. Live: a click on the
  object while lost and while tracking something else, looking away without new pieces, the countdown,
  Auto IR on a light and a dark object, 10 range switches without a hang. The user offered a CrealityScan
  screen capture of its Auto IR; and the app on the white-platform scene would tell whether its depth
  there is any better.
- **The live check of Phase 5:**
  - a platform scan with one level pass and one about 45 degrees down, at 30-35 cm;
  - solid.ply opened in Blender;
  - mesh.stl in their slicer.
- **Phase 4's live marks:**
  - 20 fps sustained in the real viewer (measured only through `bench --live`; the last live scan,
    09:22 on 11-03, ran 14.5 fps before speed step 5);
  - a live scan of the marker part, held to the Phase 1 / 3 marks.
- **The 11-03 14:48 platform scan** (captures/20261103-144524-scan) has no mesh: it overflowed the GPU
  mesh buffer before the fix. Process › Make mesh in the 3D view now makes it (a copy: 56 s, 4.7 GB peak).
- **Phase 0.7:** a full-rate 14-bit recording circling a scene. It decides the GPU tracker's ~0.7 mm
  drift on synthfp (the sensor's fixed per-pixel error; the CPU tracker drifts 0.13 mm).
- **Accuracy session:**
  - print captures/testblock/testblock.stl, measure it with calipers, scan it at close and long range;
  - then a tool that aligns the scan to `testblock.sdf` and reports distances, step heights and the
    best-fit scale;
  - also a long-range board calibration.
- **The Ubuntu 26.04 checkpoint:** rebuild .venv with uv on the new system Python and numpy, run the
  tests, evaluate, and the board calibration solve (Phase 0.6 did this with Python 3.14 + numpy 2:
  identical).

Known weaknesses (each measured, in docs/history.md):
- The thin plate's edge-on turn: neither tracker holds it; markers on both faces are the way.
- Marker-map entries 2.9 mm off can make the marker check refuse a correct pose.
- Something touching the object (a stand, a cable) stays in the scan unless cropped.
- Closed solids keep a few tiny Poisson self-intersections (slicers ignore them).
- Mesh colour is a plain average over frames. Weighting by viewing angle and sharpness, with exposure
  matched between frames, was in Phase 5's original plan and left for the texture work.
- Scene modes "turntable" and "free" (scene.MODES) are no longer used by any scan or dataset: platform
  scans run in "still" and notice the turning themselves.
- GPU replays are not bit-identical run to run (rarely one frame of difference on synth).

Later (the user's):
- Speed beyond 20 fps (30 fps would need the 6x6 solve on the GPU); the user has their own plans for
  performance.
- "Tricks like creating pseudo geometry by texture" (2026-11-03): colour photos (the scanner's colour
  frames, or photos taken separately) adding surface detail the depth can't resolve: shading or
  photometric normals, normal or displacement maps for Blender, photogrammetric refinement.
- The IMU in tracking: read its calibration (raw 4036, carefully, see Device facts), check its clock
  against the depth PTS, use the gyro as the rotation guess.

## The person you are working with

Owner of 3BUUT, Ubuntu Linux (GNOME, Wayland), works with Blender and 3D printing. Prefers short,
direct answers, wants results shown not promised, and decides names, publishing and scope
themselves. They used to scan with an Android phone and finish in CrealityScan under Bottles/Wine;
that Android + Bottles route still works and is the fallback for real scans.
