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
- **Be honest about what is estimated.** Do not present these as calibrated or verified:
  depth units and depth law, camera field of view (60 degrees is a guess), the colour-to-depth
  registration (`REG`, eyeballed from one frame), the 17-32 cm depth window (a hypothesis from
  hand-held wall readings). Say "estimate" in docs and output.
- Keep captures out of git: `captures/` is ignored (it holds photos of the workspace). Only
  `examples/` is meant to be public; check an image before adding it.
- Match the existing style: small numpy-only modules, short docstrings, no heavy dependencies.
  Python 3.10+, numpy and tkinter only. No pytest installed on the dev machine; the tests are plain
  functions and can be called directly.

## Device facts (verified on one Ferret Pro, Ubuntu GNOME/Wayland)

- USB `2bc5:069c` (Orbbec vendor ID; strings "Creality" / "CR-Scan Ferret"). USB 3.
- Three UVC functions: `/dev/video0` depth (Y10, 1280x800), `/dev/video2` IR (GREY, 1280x800,
  shows the projector dot pattern, mirrored vs colour/depth), `/dev/video4` colour (YUYV/MJPG).
  Odd nodes are metadata. `tribuutscan/v4l2.py:discover()` finds them by USB interface.
- Depth packing is an MSB-first 10-bit bitstream, 4 pixels per 5 bytes (not MIPI Y10P).
  Values ~1..200 in the default window, nearer = larger, 0 = invalid.
- ffmpeg and GStreamer cannot negotiate the depth format, and camera apps pick it up and show
  black. Capture goes through raw V4L2 ioctl + mmap (struct offsets assume 64-bit).
- Some depth frames arrive truncated; drop any frame shorter than `sizeimage`.
- Only one program can stream from the scanner at a time. Close the viewer before running
  `python3 -m tribuutscan.pointcloud` (live grab) or `tribuutscan.measure`.
- The scene must be lit and the lenses uncovered, or all streams are black.
- `pyorbbecsdk2` 2.9.3 does not enumerate the device ("No matched usb device found").
- IR brightness is set through the depth node's standard V4L2 controls: exposure 100-6800 (default
  3000) and gain 1000-15000 (default 1000). The IR node's own exposure is read-only (EACCES). The colour
  node has the usual UVC controls (brightness, gain, white balance, auto exposure). There is no V4L2
  control for projector power. Helpers live in `v4l2.py` (`controls`, `get_control`, `set_control`).
- The vendor control interface is `/dev/hidraw*` (root only); its protocol is unknown.

## Commands

```
tribuutscan-viewer                          # launcher in ~/.local/bin, copy of tribuutscan-viewer.sh
python3 -m tribuutscan.viewer               # live colour + IR + depth; `s` saves captures/<stamp>/
python3 -m tribuutscan.pointcloud [dir]     # .ply from a saved set, or grab a live one
python3 -m tribuutscan.measure <cm>         # depth stats for a flat target at a known distance
```

Beware `pkill -f ...tribuutscan.viewer` inside a shell command: the pattern matches the shell
itself. Use `pgrep -f "^python3 -m tribuutscan.viewer"` and kill by PID.

## Pitfalls already hit

- The v4l2_format union starts at byte 8 on 64-bit; getting this wrong silently ignores the
  requested format.
- Automatic colour/depth registration by edge correlation did not converge on a flat shelf scene
  of dark boxes; a calibration target or marker is the right way, not more tuning.
- Hand-held distance readings (about +-1-2 cm) cannot tell `a/z+b` from `a/z^2+b`. A fixed rig is
  needed for the real depth law.

## Roadmap

See the README. Next: USB capture of Creality's app traffic (usbmon, needs sudo, run under Wine/
Bottles or Android) to find the commands that may widen the depth window and where calibration
lives; then proper calibration, multi-view fusion (Open3D in a separate venv), geometry, markers,
texturing, and a better interface (planned in VS Code).

## The person you are working with

Owner of 3BUUT, Ubuntu Linux (GNOME, Wayland), works with Blender and 3D printing. Prefers short,
direct answers, wants results shown not promised, and decides names, publishing and scope
themselves. They used to scan with an Android phone and finish in CrealityScan under Bottles/Wine;
that Android + Bottles route still works and is the fallback for real scans.
