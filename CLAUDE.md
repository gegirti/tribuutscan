# CLAUDE.md

Working notes for Claude Code sessions in this repo. The README has the public write-up; this file
has the working context and ground rules. This file is public, so keep it free of anything private.

## What this is

triBUUTSCAN (also written 3BUUTscan): a Linux toolkit for the Creality CR-Scan Ferret Pro, used
directly over USB through the stock `uvcvideo` driver. Made in 3BUUT, MIT licensed. It is the
beginning of a full scanning setup; today it only captures data. Status: early, and the win so far
is simply getting colour, IR and depth out of the scanner on Linux.

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
  Python 3.10+, numpy and tkinter only. The browser view (`tribuutscan/web/`) uses a vendored copy of
  three.js r160 (MIT, `THREE-LICENSE`) so it works offline; the server is Python stdlib. No pytest installed on the dev machine; the tests are plain
  functions and can be called directly.

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
  whole-frame payloads). Payload headers are 12 bytes with a PTS (1 MHz); stamps from it are exactly
  33.32 ms apart and pair like V4L2 (colour - depth -6.7 ms USB vs -6.8 ms V4L2). `decode_depth`
  puts Y14 on the Y10 scale ((Y14 - 8) / 16, float32; < 16 -> 0), so the calibrated law and every
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
  The viewer now sets the IR node's auto exposure to 2 at start and after a range switch (verified 0 -> 2).
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
- Merged clouds (`cloud.py`): each cube of side "point distance" (0.5/1/2 mm presets) keeps one
  averaged point, and a cube must be seen in 3 frames. The kettle recording went from 7.2M raw
  points to 149k kept (4.3 MB) at 0.5 mm, and 32k at 1 mm. With depth noise ~1 mm (estimate), the
  count still creeps up at 0.5 mm and flattens at 1 mm. Tracking's per-frame table test
  (`isolate_mask`) takes a plane only if it holds 10% of the points, the object stands on the
  scanner's side of it, it is not seen edge-on, it reaches the image edge, and it covers 80 cm2.
  Without those rules the kettle's lid and a slice through its handle were taken for tables. The
  base kept for merging and output is found differently (see "Object separation"). Without Open3D
  there is no tracking, and every frame is merged unmoved (smeared).
- Live scan (`livescan.py`): a spawned worker using the `.venv` Python. It merges frames
  (`cloud.Scanner`) and serves the 3D page; the viewer only sends a frame when the worker is
  free. Two pitfalls hit:
  - A spawned process inherits the parent's `sys.path`, so the old system `typing_extensions`
    shadowed the venv's and Open3D failed to import. The worker puts its own site-packages first.
  - A camera-movement limit rejected real turns of the object (8 degrees at 25 cm looks like
    35-50 mm of camera movement). The limit is now judged at the object's centre.

  Frames are thinned to about 1 point per cube (every 2nd pixel at 0.5 mm), and colour is
  converted only at those pixels. The worker takes about 74 ms per frame (13 fps); inside the
  real viewer it ran about 6 fps.
  - After a tracking loss the frame is matched against the cloud so far, from the last pose
    ("move back"); otherwise one loss used to end the scan.
  - The worker exits when its viewer dies (an orphan once held port 8765, and the page showed a
    stale scan).
  - The live page shows every cube; the saved cloud keeps cubes seen 3 times, or every cube if
    that leaves none.
  - With "keep raw frames" a live scan can be replayed exactly with cloud.py.
  - A spawned worker re-runs the starting script, so test drivers need an
    `if __name__ == "__main__"` guard.
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
- Viewer v2 (2026-10-31): `viewer.App` holds state and actions; `panels.py` has one class per right-hand
  panel (`PANELS` in viewer.py lists them; each has build/refresh). Previews are index-resampled to
  SIDE_W 400 / DEPTH_W 800. Window 1486 x 969 with every panel open (fits 1920 x 1080 at Tk scaling
  1.33). The IMU is on while the viewer runs (if the udev rule allows): its vendor commands go through
  the USB depth capture (`vendor.Channel(usb=...)`, control transfers to interface 0 / unit 4), it is
  switched off before a range switch and on close, and back on after the switch. Checked on the
  scanner: 30 fps on all streams, IMU 199-201 Hz, a 5 s recording held 51 sets and 1002 IMU rows,
  range switches ~10 s each with the IMU back, and after close the IMU switches read 0 and hidraw
  returned. Recordings: imu.csv (host_t monotonic at arrival, device_us, accel g, gyro deg/s, temp),
  `frames.read_imu`; info.json "imu" has the rates and ranges.
- Viewer robustness (2026-10-31, after the user's first session with v2):
  - Two viewers ran at once and the second silently took the scanner. The USB claims now use
    DISCONNECT_CLAIM with EXCEPT_DRIVER "usbfs" (`usbdepth.claim`): they unbind uvcvideo/usbhid
    but get EBUSY instead of taking an interface from another program. The viewer also holds
    captures/.viewer.lock (flock), and a second one shows "already running".
  - The scanner reset itself 10 s after a range switch back to close range, and the viewer froze
    silently. Now `App.tick` sees a stream that lost its device. `App.reconnect` then stops a recording
    or scan (it saves them), pauses everything, reopens with the mode read anew (a reset means close
    range) and switches the IMU back on. Retries after 10 s if the scanner is not back. Checked with a
    simulated loss: back in 4 s. A real reset was not forced.
  - The close crashed on a vanished device (STREAMOFF -> ENODEV), so it took three clicks. Now
    `v4l2.Capture.close` ignores that.
  - Retry (user's request):
    - At start, while the scanner can't be opened, `open_app` asks with Retry/Cancel
      (`scanner_problem` says whether it is unplugged or busy). App closes what it already opened,
      so a retry finds the scanner free.
    - While running, a lost scanner turns the distance bar red with a countdown and shows Retry now
      (`retry_now`).
    - The start path was checked with the scanner really unplugged. The user then used it for real:
      5 Retries while unplugged, plugged in at 23:40:43, viewer open at 23:40:47 with the IMU on.
    - The running path (a real unplug) has a unit test and the reconnect logic, but has not been
      tried by hand yet.
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
tribuutscan-viewer                          # launcher in ~/.local/bin, copy of tribuutscan-viewer.sh
python3 -m tribuutscan.viewer               # depth (middle), colour + IR (left), panels (right, fold):
                                            # Scanner: Range close 15-33 / long ~20-72 cm (estimate),
                                            #   IR exposure/gain, colour brightness, Auto IR
                                            # Capture: `s` saves captures/<stamp>/, `r` records raw sets
                                            #   + imu.csv to captures/<stamp>-rec/ (~49 MB/s), `m` meshes
                                            #   the latest recording via .venv fuse.py
                                            # Scan: `c` live scan (worker process), Open 3D view
                                            # IMU: tilt horizon, turn rate; View: Find markers
                                            # events: message line, console, captures/viewer.log
python3 -m tribuutscan.vendor [modes|factory|scan|switch "<mode>"]   # depth modes, factory calibration
                                            # (switch WRITES); TRIBUUTSCAN_CAMERA=board|factory|estimate
python3 -m tribuutscan.imu [seconds]         # switch the IMU on, read it, switch it off
python3 -m tribuutscan.pointcloud [dir]     # .ply from a saved set, or grab a live one
python3 -m tribuutscan.measure [cm]         # flat target: plane RMS, noise, step -> captures/measure.csv
python3 -m tribuutscan.usbdepth [s] [--all] # 14-bit USB depth throughput check (--all: + colour, IR)
                                            # TRIBUUTSCAN_DEPTH=y10 forces 10-bit V4L2 depth anywhere
python3 -m tribuutscan.markers captures/<stamp>   # find markers in a saved frame -> markers.png
.venv/bin/python -m tribuutscan.cloud captures/<stamp>-rec [small|medium|large] [--keep-base]
                                            # merged point cloud -> captures/<stamp>-scan/ (base removed)
python3 -m tribuutscan.view3d captures/<stamp>-scan    # browser 3D view at http://localhost:8765
.venv/bin/python -m tribuutscan.optimise captures/<stamp>-scan   # loop closures + pose graph + merge again
.venv/bin/python -m tribuutscan.session merge A-scan B-scan AB-scan   # merge two scans into one
.venv/bin/python -m tribuutscan.calibrate board|capture [square_mm]|solve <dir> [square_mm]
                                            # ChArUco calibration -> captures/calibration.json
.venv/bin/python -m tribuutscan.evaluate [datasets] [--label x] [--against report.json]
                                            # tracking metrics on saved + synthetic recordings
.venv/bin/python -m tribuutscan.synth [out] [frames] [every] [hard]   # synthetic box scan, ground truth
.venv/bin/python -m tribuutscan.fuse captures/<stamp>-rec   # ROUGH mesh.ply from a recording (Open3D)
.venv/bin/python -m tribuutscan.compare captures/<stamp>-scan [ref.ply]  # score against a reference mesh
                                            # (default captures/reference/part/mesh.ply) -> compare.json/.ply
```

Beware `pkill -f ...tribuutscan.viewer` inside a shell command: the pattern matches the shell
itself. Use `pgrep -f "^python3 -m tribuutscan.viewer"` and kill by PID.

## Tracking evaluation (P0 of the tracking plan)

`evaluate.py` runs a tracker over these datasets: tilt, standing, kettle, hand, face (private, never
rendered), synth, synth7 and synthhard. On 2026-10-30 the real ones were deleted (the user kept only
the last marker scan); the numbers below stay as the record. Datasets now: markers
(captures/20261030-194411-scan, the first recording with calibration and time-paired frames) and
the synthetic sets (regenerated byte-identically when missing). Metrics:
- pose jumps: steps over 30 mm / 15 degrees at the object centre
- double-surface ratio: share of points with a same-facing sheet 2.5-8 mm along their normal
- marker scatter: spread of each marker's world position; independent of shape tracking
- ATE/RPE: against ground truth, synthetic only

A keyframe residual and a split-half Chamfer were tried and dropped: they read the same on the
doubled and the good scan. Baseline (frame-to-frame ICP + relock):

| dataset | jumps | double | marker scatter | ATE |
|---|---|---|---|---|
| tilt | 4 | 0.60 | 3.5 mm, 32 groups | - |
| standing | 0 | 0.00 | 0.6 mm | - |
| synth | 0 | 0.23 | 1.5 mm | 4.0 mm |
| synthhard | 0 | 0.38 | 1.1 mm | 6.0 mm |

The doubled black part came from `relock()` accepting impossible jumps (117-193 mm / 42-64 degrees).
P1 (tracker "current" in evaluate.py; "baseline" keeps the first version):
- re-finds are gated (a believable step from the last good pose, at most 2x the normal limit)
- the next-frame guess follows real time
- `prep.clean` drops window/jump/grazing/edge/speckle pixels, but only for MERGING. Tracking on
  cleaned depth doubled the frame-to-frame error on the synthetic box: depth edges and steep
  faces are what hold flat objects in place.
- results: tilt 0 jumps (was 4) and 39% doubled (was 60%); face 2 jumps (was 13); synthetic ATE
  unchanged
- camera constants live in `calib.py` (`CAMERA`, estimates until captures/calibration.json exists)
- `v4l2.Capture` exposes the kernel capture time (`last_time`, monotonic) and sequence number; the
  viewer pairs streams by nearest capture time. Checked on the scanner: the stamps must be read
  BEFORE re-queueing the buffer (QBUF clears them). All three streams run at 30 fps with no dropped
  sequence numbers; the colour frame sits a constant 8.9 ms from its depth frame, and a frame
  reaches the program 27-29 ms after its stamp.
P2 (`scene.py`, scan modes still/turntable/free):
- In "still", the table plane is found once (3 of the first 5 frames), kept in world coordinates,
  and re-fitted to each frame near its kept place. The object is what lies above it, inside a
  slowly growing region.
- Tilt: plane applied in 153 of 155 tracked frames (was on/off in 88 of 168); doubled 39% -> 23%.
- Measured and dropped: tracking on the whole scene or on the kept object region. A large flat
  table lets point-to-plane ICP slide (synthetic error 4 -> 50 mm), and a plane placed by a
  predicted pose leaks table points when the prediction is a few mm off. Tracking stays on each
  frame's own piece; only merging uses the kept plane.
- Flat objects still slide under shape-only tracking (synthetic error 3-6 mm): P3/P4.
P3 (`markermap.py`):
- A world map of uncoded markers. Matching uses nearest neighbours plus pairwise-distance checks
  with a predicted pose, and triangle side lengths without one. Symmetric and collinear layouts
  are refused.
- Marker 3D comes from a ring-plane fit, falling back to the dot's own depth.
- Marker-only tracking: synthetic ATE 3.4 -> 1.0 mm (hard path 6.1 -> 2.1), doubling about 0.02.
- On the real tilt scan it fails: a median of 3 markers have 3D per frame, and only 87 of 168
  frames have 3 or more. More markers on the part are needed for marker mode; otherwise markers
  only help inside a combined solve (P4).
- With 3D markers nearly coplanar and 25 cm away, the camera position from markers alone swings
  1.5-3.7 mm with 0.3 mm marker noise, while the markers themselves land within 0.5 mm.
- A field-of-view sweep (54-66 degrees) by marker scatter was inconclusive (tracking error
  dominates), so 60 degrees stays the estimate until calibration.
P4 (`track.py`, now the default tracker):
- One Gauss-Newton solve combines point-to-plane (Tukey 3 mm, plus a distance down-weight), marker
  terms (sigma 1 mm, dropped beyond 3 sigma on the fine level) and a motion prior (20 mm / 5
  degrees). It runs against the model (the cloud so far), after stray-point removal.
- Results against the first version:
  - tilt doubling 60% -> 16% and 0 jumps (one clean shell in the renders)
  - face 13 -> 1 jumps
  - synthetic ATE 4.0 -> 1.7 mm, 1.7 -> 1.0 mm, and 6.0 -> 1.5 mm on the hard path
- Measured and dropped:
  - the model target WITHOUT markers drifts (synthetic 25 mm)
  - loose marker weighting (3 mm) lets marker noise pull poses off (tilt 78% doubled)
  - Open3D's own ICP fails with 20% clutter (390 mm), whereas this solve stays at 0.1 mm once
    stray points are removed
- Real marker positions are noisy (median 2.4 mm against the map) until the colour registration
  is calibrated.
- The flat-face flag fires on most frames of flat parts, so its threshold needs tuning before it
  can be a hint.
- ORB features (`features.py`, OpenCV 4.10 in `.venv`) give the solve its starting pose: whole
  image in "still" mode, object only otherwise. Tilt doubling 16% -> 6.7%, hard synthetic ATE
  1.5 -> 1.0 mm.
- The marker check also catches sliding: shape fits but markers disagree. Dropping it took tilt
  doubling 7% -> 26%.
- Against the first version: tilt 60% -> 6.7% doubled and 0 jumps; face 13 -> 0 jumps; synthetic
  ATE 1.0-1.5 mm (was 1.7-6.0); standing drift 0.1 mm.
P5 (live):
- Speed: normals only on merged pixels, cleanup only around the object, ORB at half resolution on
  luma (taken straight from YUYV), fewer solve iterations, `markers.detect_fast` (OpenCV blobs +
  the same checks, 8 ms against 47), and merging in a background thread. Live worker 5.6 -> about
  8 fps (simulated camera).
- On the scanner the live scan first ran 5.4 fps with a 91 ms colour/depth skew: the viewer's
  previews took the CPU from the worker (113 ms per frame alone, 161 ms beside the viewer), and the
  stream threads fell behind. The viewer now renders only the newest frame, at most 15 previews a
  second and only the shown pixels, and keeps 12 frames per stream for pairing. Live: 7.3 fps,
  126 ms per frame, skew 10 ms median.
- The object region must update on EVERY tracked frame: updating it only on merges let the region
  take in too much, the model filled with it, and tilt doubling went 1% -> 65%.
- The model rebuild waits for pending merges, so a replay of the worker's raw copy reproduces the
  live poses (0.0005 mm).
- The flat hint shows when the shape alone slides (degeneracy < 0.005) and no marker held the frame.
- Sessions also save track.csv, keyframes.npz, markers.json and cloud_state.npz (for Optimise and
  continuing a scan).
- Final against the first version: tilt 60% -> 1.1% doubled; face 13 -> 0 jumps; synthetic ATE
  1.2 / 1.3 / 0.94 mm (was 4.0 / 1.7 / 6.0).
P6 (`optimise.py`, runs automatically after Stop; Process menu: Optimise again, Merge again at
0.5/1/2 mm, Make mesh, and since 2026-11-01 Remove base / Keep base):
- Loop candidates come from shared markers, ORB matches or nearby poses, at most 3 per keyframe and
  at least 10 keyframes apart. Each is checked with `track.solve`. Then an Open3D pose graph
  (preference_loop_closure 2.0 kept a true loop and pruned a 30 mm false one; 1.0 pruned both),
  and the cloud is merged again from each merged frame's kept camera points (merged.npz).
- Keyframe ATE on the synthetic sets 1.2/1.3/1.0 -> 0.19/0.28/0.34 mm; 5-15 s for 70-110
  keyframes (every pair took 60 s for the same gain).
- The mesh is Poisson from the oriented cloud (a TSDF mesh needs depth images, which are not kept).
- Our real paths do not close a loop; a closed-loop recording is planned for the scanner session.
P7 (`session.py`; viewer "Continue last"):
- A continued scan re-finds its place in the saved one (global marker match, else FPFH + RANSAC
  against the saved cloud, checked by `track.solve`), then tracks and merges into it. Stop's
  Optimise ties old and new together.
- Tilt scan split in two and continued: re-found on the first frame; 0.14 mm median from one
  uninterrupted run.
- Merging two separate scans: 0.09 mm median from the single run, 46 s.
- Marker triangle search is vectorised over the map: the merge took 488 s with a Python loop over
  every map triangle.
- A race fixed: `relock()` read the cloud while a background merge replaced its arrays.
Scanner session preparation:
- `calibrate.py` writes an exact-size A4 ChArUco board (captures/calib/board.svg), a live capture
  window and the solver.
- On the scanner, IR shows the board only through the projector's dots, so its ArUco codes cannot
  be read at 30 cm (median, Gaussian, dilation and plain chessboard detection all failed), and corner
  refinement follows the dots (2-3 px off). The board is found in colour; IR is aligned to it: whole
  board by ECC from the colour corners + the estimated colour -> depth mapping, then each corner by a
  shift-only ECC of a one-square patch (keeps lens distortion). Still board: 0.03 px scatter;
  synthetic dot-lit views: 0.26 px median error. The synthetic end-to-end solve with dots and lens
  distortion recovered fx 1107.7 (1108), T within 0.1 mm, law a 319874 (320000).
- OpenCV's CharucoDetector corners sit about 0.5 px off; `detect()` refines them with cornerSubPix.
- The capture window keeps a view by itself when the board has been still for 0.5 s at a new place
  (so the user can hold the scanner), and pairs the three streams by capture time
  (`frames.paired`: reading them in turn left them a frame or more apart).
- A board page on a spiral pad was bowed by 6 mm: the depth map showed both ends 3-4.5 mm nearer
  than the middle. Tape the page flat on a desk.
Calibration result (captures/calib/20261030-184110, 28 views, 19-33 cm, tilts to 47 degrees):
- Reprojection colour 0.35 px, IR 0.32 px, stereo 0.34 px; even/odd halves agree within 0.2%.
- Depth camera: fx 963.5, fy 962.5, cx 631.7, cy 390.8: 67.2 degree FOV (the 60 degree guess made
  every cloud ~13% too narrow sideways). Small lens distortion.
- Depth law z = 289570 / (v + 873.4) mm, the a/z form (median 0.30 mm; a/z^2 3.5 mm). The estimate
  read 1.4% short at 25 cm and 3.6% short at 31 cm.
- Colour camera: fx 851, fy 946: non-square pixels, which fits a 16:9 sensor squeezed into the
  16:10 mode (1.11). It sits 19.6 mm sideways from the depth camera, turned 0.23 degrees.
- On frames not used for calibration: a board square measured from depth is 24.03 mm (true 24.00;
  the estimates gave 20.12), and colour lands on depth within 0.3 px (estimates 2.8 px).
- Synthetic data always uses the estimated (pinhole) camera: `calib.use()` switches the shared model,
  `synth.generate` and `evaluate.run` do it for synthetic sets.
- Tracking with the calibration: synthetic unchanged, standing and kettle as good or better. The old
  tilt recording doubled (1% -> 40%) after ONE wrong re-find (45 mm / 10.7 degrees after 2 lost
  frames, inside the 2x gate of 60 mm / 30 degrees). That recording predates the timing fix (colour
  up to ~90 ms from depth) and the tracker was tuned on it with the wrong geometry; a fresh
  recording is needed before tuning the gate.
- The solver finds colour/IR intrinsics, stereo extrinsics (`CALIB_FIX_INTRINSIC`), and the depth
  law per pixel against the board plane. It compares v = A/z + B with A/z^2 + B, which hand
  readings could not tell apart.
- On synthetic views with known truth it recovered fx 1109.0 (1108.5), baseline 25.16 mm (25.20)
  and law a 320064 / b 999.7 (320000 / 1000), at reprojection < 0.4 px.
- `calib.Camera` has the full model: lens distortion, and colour <-> depth through 3D at each
  pixel's distance. The two cameras sit apart, so the offset swings about 58 px between 18 and
  31 cm. The old fixed affine REG is only right at one distance, which likely explains the 2-5 mm
  marker noise.
- Estimate mode is unchanged (same evaluation numbers).
Markers: the retroreflective dot usually has depth on this scanner (about 90% of its pixels), and a
plane fitted to the ring around the dot is flat to about 0.3 mm.

## Reference part (2026-10-31)

captures/reference/part/mesh.ply is CrealityScan's final mesh of the marker test part (Android app,
4 scans merged, 2026-10-30; mm; ~274 x 192 x 44 mm), the same part as captures/20261030-194411-scan.
It is a scan too, not ground truth.
- First reading (2026-10-31, old code): 76% of our points on the part, median 0.77 mm, p90 3.4 mm,
  error "growing towards the rim", a free scale fitting at 0.970.
- CORRECTED 2026-11-01: that rim error was table points around the part's outline counted as part
  (the table was never removed on that scan), plus an alignment pulled by background. With the base
  found and left out, the same recording gives median 0.22 mm and p90 0.53 mm (see "Object
  separation"). Frames are flat too: a full frame of taped paper at 30 cm is flat to +-0.15 mm from
  the image centre to the edge. The scale question waits for the printed test box.
- compare.py's alignment (FPFH + RANSAC from 5 seeds) varies between point sets. For before/after
  comparisons use more seeds (`compare.SEEDS = 12`), and the same transform for clouds that share
  tracking.

## Object separation (2026-11-01)

The marker scan never found its table (plane_frames 0). The part filled the view, so the table was
only 7-9% of the points, under tracking's 10% / 80 cm2. Every frame then fell back to the per-frame
centre piece, and the table connected the part to its surroundings.
- `scene.py` finds the base with its own rules: share 3%, 30 cm2, up to 5 planes per frame, votes
  over the first 300 frames.
- `stands_on`: the centre object must come within 10 mm of the plane (a wall behind it does not),
  and at most half the plane may lie inside the object's outline seen along the normal (the
  object's own flat sides do).
- Three votes must agree in world coordinates.
- Merging takes the object and the base inside its cylinder (`Scene.masks`). The tracking model and
  relock use object-only points (`Scanner.model_points`), so tracking's target is what it was.
- Output (`Scanner.output_mask` / `output_cloud`, used by save, live view, download, mesh and
  evaluate): inside the cylinder, the base left out when `remove_base` (viewer Scan panel checkbox,
  default on; the 3D page's Process "Remove base" / "Keep base" re-saves), and only what is connected
  to the object core over 3 mm cells.
- With no base found, the object is the centre piece inside a ball that grows to at most 2x its start.
- Marker recording replayed (cloud.py), scored against Creality's mesh with a stable alignment:

  | | before | after, base kept | after, base removed |
  |---|---|---|---|
  | points on the part | 79.9% | 86.5% | 92.2% |
  | median / p90 distance | 0.69 / 3.12 mm | 0.22 / 0.53 mm | 0.22 / 0.53 mm |
  | Creality surface covered within 1 / 2 mm | 47 / 58% | 71 / 75% | 68 / 72% |

  Base found in 577 of 581 frames (from frame 4). The remaining 7.8% are the stand the part rests on.
  It touches the part, so geometry cannot tell it apart: a crop in the 3D view is the way (TODO).
- evaluate.py before/after: tracking identical or better, no lost frames or jumps, synthetic ATE
  1.19/1.29/0.94 -> 1.17/1.28/0.87 mm. The marker scan is 361 -> 234 ms per frame: merging no longer
  repeats the per-frame object search.
- Live: the worker returns each frame's object mask at 1/8 size (`livescan.view_mask`). The viewer's
  depth preview dims all but the object and tints a kept base. Checked on recorded frames (base from
  frame 4: ~4000 object and ~500 base cells of 16000). Not yet seen live, because nothing was in range.
- `cloud.py` now saves through the Scanner (output filter, session.json "remove_base" /
  "base_found", saved state); `--keep-base` keeps the base. Careful: it writes next to the
  recording. Replay a scan's own -rec through a symlink elsewhere, or it overwrites that scan.

## Pitfalls already hit

- Open3D (0.20, in the gitignored `.venv`): the legacy `ScalableTSDFVolume` gives empty meshes on
  this data. The tensor `VoxelBlockGrid` works, and its mesh extraction skips voxels seen in
  fewer than 3 frames. Plain frame-to-frame ICP lost track when the object was moved, so
  `fuse.py` uses coarse-to-fine ICP plus a check that rejects implausible jumps.

- The v4l2_format union starts at byte 8 on 64-bit; getting this wrong silently ignores the
  requested format.
- Automatic colour/depth registration by edge correlation did not converge on a flat shelf scene
  of dark boxes; a calibration target or marker is the right way, not more tuning.
- Hand-held distance readings (about +-1-2 cm) cannot tell `a/z+b` from `a/z^2+b`. A fixed rig is
  needed for the real depth law.

## Roadmap

See the README TODO. The USB-capture idea is no longer needed: vendor.py reads and switches the
depth modes directly.

GPU PIPELINE (approved 2026-11-01, the user's top priority, several sessions): a GPU-resident,
KinectFusion-style pipeline (depth maps, TSDF fusion, ray casting, frame-to-model tracking, marching
cubes on the GPU; the CPU keeps the 6x6 solve, markers, base votes, pose graph, ORB when lost).
- Ground rules from the user: reach everyone. Stock distro drivers only (no driver PPA upgrade), any
  GPU vendor (Intel / AMD / NVIDIA) plus a CPU fallback, no CUDA / ROCm / oneAPI. Ubuntu 26.04 LTS
  is coming within months: newer Python, numpy 2, newer Mesa.
- Engine by bake-off: wgpu-py (WebGPU over Vulkan / Metal / DX12; the lean, since Mesa's Vulkan is
  what ships everywhere and is the newest driver here, 25.2.8) vs pyopencl (OpenCL; here only Intel's
  23.43 runtime). open3d-xpu is out (Intel-only, needs a newer driver). Kernel rules: no float or
  64-bit atomics, no FP64, 32-bit indices, dense block index + brick pool (no hash map), 16-bit
  normalised TSDF, same results on the CPU fallback (lavapipe / PoCL).
- Phases: 0 ground work; 1 proper meshes on Open3D's CPU VoxelBlockGrid (first deliverable, no GPU
  needed); 2 bake-off; 3 GPU tracking (keeps track.solve's contract: each frame's own centre piece
  on uncleaned depth, model without the base, markers + prior on the CPU, only gated frames fused);
  4 live 20-30 fps; 5 finishing (re-fuse after Optimise, colour, per-mode voxel size); 6 clean-up;
  a checkpoint after the 26.04 upgrade.
- Risks: disk (32 GB free, 93% full: full-rate Y14 + colour + IR ~147 MB/s), the A750 also drives
  the desktop (chunk long kernels), only Intel hardware here (other vendors via testers).
- Phase 0 done so far (2026-11-01): register_color fixed (it failed with any colour model); scans
  now save frame_poses.txt (every tracked frame; Optimise's keyframe corrections spread to the frames
  between), labels.npz (each tracked frame's 1/8 object/base mask, `cloud.view_mask`), session.json
  `sources` [{rec, first_index}] (cloud.scan and the live worker's raw copy; continued and merged
  scans keep their offsets) and `camera` {source, fingerprint}. evaluate.py unchanged (identical
  metrics on all four datasets). Scans saved before this have only their merged poses.

## The person you are working with

Owner of 3BUUT, Ubuntu Linux (GNOME, Wayland), works with Blender and 3D printing. Prefers short,
direct answers, wants results shown not promised, and decides names, publishing and scope
themselves. They used to scan with an Android phone and finish in CrealityScan under Bottles/Wine;
that Android + Bottles route still works and is the fallback for real scans.
