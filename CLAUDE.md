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
  count still creeps up at 0.5 mm and flattens at 1 mm. The table/turntable is removed only if a
  plane holds 10% of the points, the object stands on the scanner's side of it, it is not seen
  edge-on, it reaches the image edge, and it covers 80 cm2. Without those rules the kettle's lid and
  a slice through its handle were taken for tables. Without Open3D there is no tracking, and
  every frame is merged unmoved (smeared).
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
- The vendor control interface is `/dev/hidraw*` (root only); its protocol is unknown.

## Commands

```
tribuutscan-viewer                          # launcher in ~/.local/bin, copy of tribuutscan-viewer.sh
python3 -m tribuutscan.viewer               # live colour + IR + depth; `s` saves captures/<stamp>/
                                            # `r` records raw sets to captures/<stamp>-rec/ (~43 MB/s),
                                            # Auto IR button, distance guide (estimate),
                                            # `m` meshes the latest recording via .venv fuse.py;
                                            # events print to the console and captures/viewer.log
                                            # `c` live scan (merged cloud, worker process), Open 3D view
python3 -m tribuutscan.pointcloud [dir]     # .ply from a saved set, or grab a live one
python3 -m tribuutscan.measure <cm>         # depth stats for a flat target at a known distance
python3 -m tribuutscan.markers captures/<stamp>   # find markers in a saved frame -> markers.png
.venv/bin/python -m tribuutscan.cloud captures/<stamp>-rec [small|medium|large]
                                            # merged point cloud -> captures/<stamp>-scan/
python3 -m tribuutscan.view3d captures/<stamp>-scan    # browser 3D view at http://localhost:8765
.venv/bin/python -m tribuutscan.optimise captures/<stamp>-scan   # loop closures + pose graph + merge again
.venv/bin/python -m tribuutscan.evaluate [datasets] [--label x] [--against report.json]
                                            # tracking metrics on saved + synthetic recordings
.venv/bin/python -m tribuutscan.synth [out] [frames] [every] [hard]   # synthetic box scan, ground truth
.venv/bin/python -m tribuutscan.fuse captures/<stamp>-rec   # ROUGH mesh.ply from a recording (Open3D)
```

Beware `pkill -f ...tribuutscan.viewer` inside a shell command: the pattern matches the shell
itself. Use `pgrep -f "^python3 -m tribuutscan.viewer"` and kill by PID.

## Tracking evaluation (P0 of the tracking plan)

`evaluate.py` runs a tracker over these datasets: tilt, standing, kettle, hand, face (private, never
rendered), synth, synth7 and synthhard. Metrics:
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
  viewer pairs streams by nearest capture time. This is not yet checked on the scanner.
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
0.5/1/2 mm, Make mesh):
- Loop candidates come from shared markers, ORB matches or nearby poses, at most 3 per keyframe and
  at least 10 keyframes apart. Each is checked with `track.solve`. Then an Open3D pose graph
  (preference_loop_closure 2.0 kept a true loop and pruned a 30 mm false one; 1.0 pruned both),
  and the cloud is merged again from each merged frame's kept camera points (merged.npz).
- Keyframe ATE on the synthetic sets 1.2/1.3/1.0 -> 0.19/0.28/0.34 mm; 5-15 s for 70-110
  keyframes (every pair took 60 s for the same gain).
- The mesh is Poisson from the oriented cloud (a TSDF mesh needs depth images, which are not kept).
- Our real paths do not close a loop; a closed-loop recording is planned for the scanner session.
Markers: the retroreflective dot usually has depth on this scanner (about 90% of its pixels), and a
plane fitted to the ring around the dot is flat to about 0.3 mm.

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

See the README. Next: USB capture of Creality's app traffic (usbmon, needs sudo, run under Wine/
Bottles or Android) to find the commands that may widen the depth window and where calibration
lives; then proper calibration, multi-view fusion (Open3D in a separate venv), geometry, markers,
texturing, and a better interface (planned in VS Code).

## The person you are working with

Owner of 3BUUT, Ubuntu Linux (GNOME, Wayland), works with Blender and 3D printing. Prefers short,
direct answers, wants results shown not promised, and decides names, publishing and scope
themselves. They used to scan with an Android phone and finish in CrealityScan under Bottles/Wine;
that Android + Bottles route still works and is the fallback for real scans.
