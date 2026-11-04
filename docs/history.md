# triBUUTSCAN: working history

The dated working log, moved out of CLAUDE.md in Phase 6 (2026-11-04) so the working notes stay short.
Everything below is as it was written on the day: numbers are as measured then, and later entries
sometimes correct earlier ones. CLAUDE.md has how things work now.

Code named here that no longer exists: the evaluate.py tracker variants baseline, p2, frame, nomarkers,
nofeatures, markers, inline, gpu-p3, gpu-seg, gpu-allmerge and gpu-noverify, and the cloud.Scanner options
behind them (clean_depth, gate_refind, motion_dt, target, use_features, threaded_merge, segment_stride, orb,
verify_gap; track_with "icp", "jointnm", "markers"), removed in commit 5421e39; check out an earlier commit
to run them again. setup-venv.sh (pip) gave way to uv (`uv sync --all-extras`).

## Device notes that described the pipeline at the time (from CLAUDE.md's Device facts)

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
- Status and Messages boxes (2026-11-02, the user's request): plain text in the room under the previews.
  Status (every 0.5 s): what runs, the scanner's depth / range / calibration, stream rates, the worker
  (GPU from its "tracking on the GPU" line, merge process found as its spawn_main child), and per process
  CPU, memory and GPU from /proc (`sysinfo.py`, stdlib; reading the worker's DRM fdinfo from the viewer
  works, ~1 ms). Messages: everything `App.log` gets, plus the meshing job's output line by line (a
  reader thread: it also never fills the pipe). ("TkFixedFont", 9) is NOT monospace here (a family name:
  Noto Sans); the boxes use a sized copy of the named font. Window 1491 x 971 with every panel open.
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

## Tracking evaluation (P0 of the tracking plan)

`evaluate.py` runs a tracker over these datasets: tilt, standing, kettle, hand, face (private, never
rendered), synth, synth7 and synthhard. On 2026-10-30 the real ones were deleted (the user kept only
the last marker scan); the numbers below stay as the record. Datasets now: markers
(captures/20261030-194411-scan, the first recording with calibration and time-paired frames) and
the synthetic sets (regenerated byte-identically when missing; synthfp and synthplate since 11-01,
see the GPU roadmap). Metrics:
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
- LEAKS CLOSED (2026-11-01 evening, after Phase 1's mesh showed 5 sheets of surroundings). A replay
  logging the region per frame (radius, top, re-fit angle) and the labels drawn over colour frames
  showed three causes:
  - garbled depth rows (see Device facts) merged as object: horizontal streaks in the labels.
  - `plane_in_camera` re-fitted the kept plane to everything within 10 mm of it. Where the part's
    lower edge or a low face outnumbered the table, the re-fit tipped onto the part: 276 of 581
    frames 20-64 degrees off, half the part's face labelled base, and 103 frames with an empty or
    sheet-only object (the 54 empty-label frames of the Phase 1 notes). Now only points outside the
    object's footprint (radius - margin) re-fit it, and a re-fit turning > 3 degrees or moving > 5 mm
    (at the axis) is not taken.
  - the region grew toward anything connected to anything inside it (its 20 mm margin included), 15
    mm per frame: radius 148 -> 317 mm, top 183 -> 225 for a part 145 mm in radius. 3 mm instead of
    6 mm connectivity changed nothing (the stand touches the part). Now: the object is the frame's
    `centre_piece` (what the middle of the view hits, the piece tracking uses) when it reaches the
    region's core (radius and top minus the margin), else everything connected to the core
    (`Scene.off_centre` counts those; 0 on the marker replay); it grows only toward points in the
    middle half of the view (`GROW_WINDOW` 0.5); merged points stay inside the region.
  Marker recording, board camera, against the baseline (12 seeds; captures/baseline/separation-scan,
  separation-run.txt):

  | | baseline | now |
  |---|---|---|
  | cloud after Optimise: on the part | 92.4% | 98.6% |
  | cloud: median / p90 | 0.229 / 0.562 mm | 0.236 / 0.576 mm |
  | cloud: Creality within 1 / 2 mm | 66.5 / 68.9% | 66.5 / 68.9% |
  | mesh: on the part | 76.9% | 95.0% |
  | mesh: median / p90 | 0.083 / 0.356 mm | 0.090 / 0.340 mm |
  | mesh: Creality within 1 / 2 mm | 66.2 / 68.7% | 66.1 / 68.5% |

  Region 182 mm / top 183; 581 tracked, 0 lost, 206 merged (all with labels; was 207, 28 empty), mesh
  from 206 frames. Without Optimise (replay only): 98.5% on the part, 0.219 / 0.528 mm (baseline
  0.246 / 0.652), 68.2 / 72.1% (69.0 / 72.7). What is left off the part: two small pieces against
  its edge, the stand it leans on (touching: the crop TODO).
  evaluate.py `--against` the baseline (captures/eval/20261101-225354-separation): markers 581
  tracked, 0 jumps, doubled 0.0001, marker scatter 0.938 -> 0.947 mm, 2 re-finds (was 1); ATE synth
  1.17 -> 1.14, synth7 1.28 -> 1.27, synthhard 0.87 -> 0.93, synthfp 0.548 = 0.548, synthplate
  10.8 -> 6.3 mm (160 of 160 tracked); truth mm equal or better.
  MEASURED AND DROPPED: dropping a garbled frame whole instead of clearing its rows. On the marker
  recording (3.3 fps) the 44 dropped frames let tracking double the part: coverage 66.5 -> 40.8%.
- The user's hand-model capture (2026-11-01, captures/20261101-220331, one saved frame, long range,
  ~44 cm, dark room): the floor is found as the base, the hand (about 22 cm tall, 90 mm region) is
  the object, the cable and boxes around it are left out. It also showed the first garbled row seen.

## Pitfalls already hit

- Open3D (0.20, in the gitignored `.venv`): the legacy `ScalableTSDFVolume` gives empty meshes on
  this data. The tensor `VoxelBlockGrid` works, with three catches: its extraction keeps weights
  STRICTLY above the threshold (3 meant "seen in 4 frames"; fusion.py passes n - 0.5), float depth
  needs float colour (or uint16 depth with uint8 colour), and a frame touching no block raises
  ("No block is touched"), so empty frames are skipped. Plain frame-to-frame ICP lost track when the object was moved, so
  `fuse.py` uses coarse-to-fine ICP plus a check that rejects implausible jumps.

- The v4l2_format union starts at byte 8 on 64-bit; getting this wrong silently ignores the
  requested format.
- `np.load` of an .npz decompresses the whole array again on EVERY `f["name"]`, and a slice of it
  keeps that copy alive. `Scanner.restore` did this per merged frame (since P6): on the marker scan
  (207 merged frames, 12.2M points) it grew past 8 GB. Read each array once (`cloud.load_npz`):
  Optimise now peaks at 1.3 GB.
- That 8 GB process ran inside VS Code's systemd scope, so the OOM killer's kill made systemd stop the
  whole scope (OOMPolicy=stop) and VS Code died with it. Start heavy jobs from a session in their own
  capped scope: `systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G <cmd>` (it keeps
  the working directory and environment). Note too that this machine's shells have `.venv` activated, so plain `python3`
  is the venv: use `/usr/bin/python3` to test without Open3D.
- Automatic colour/depth registration by edge correlation did not converge on a flat shelf scene
  of dark boxes; a calibration target or marker is the right way, not more tuning.
- Hand-held distance readings (about +-1-2 cm) cannot tell `a/z+b` from `a/z^2+b`. A fixed rig is
  needed for the real depth law.

## Roadmap

See the README TODO. The USB-capture idea is no longer needed: vendor.py reads and switches the
depth modes directly.

LATER (the user, 2026-11-03): "tricks like creating pseudo geometry by texture": colour photos (the scanner's
colour frames, or photos taken separately) adding surface detail the depth can't resolve: shading or
photometric normals, normal or displacement maps for Blender, photogrammetric refinement. After Phase 5.

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
- Risks: disk (46 GB free on 11-01; recordings now ~75 MB/s at full rate, Phase 0.3), the A750 also drives
  the desktop (chunk long kernels), only Intel hardware here (other vendors via testers).
- Phase 0 done so far (2026-11-01): register_color fixed (it failed with any colour model); scans
  now save frame_poses.txt (every tracked frame; Optimise's keyframe corrections spread to the frames
  between), labels.npz (each tracked frame's 1/8 object/base mask, `cloud.view_mask`), session.json
  `sources` [{rec, first_index}] (cloud.scan and the live worker's raw copy; continued and merged
  scans keep their offsets) and `camera` {source, fingerprint}. evaluate.py unchanged (identical
  metrics on all four datasets). Scans saved before this have only their merged poses.
- Phase 0.3 disk (2026-11-01), measured on the marker recording and Y14 frames: IR is 1.02 of 4.86 MB
  per set and nothing replays it; zlib level 1 packs Y10 depth 5.5x and Y14 4.0x (at 30-37% valid
  pixels; 172-240 MB/s on one core, the GIL is released); colour YUYV only 1.4x for ~0.85 core, and
  MJPG would need a JPEG decoder (not numpy/tkinter), so colour stays raw. Recorder now keeps colour +
  depth.zlib (4-byte length + data per frame; a cut-off last record is dropped), ~2.3-3 MB per set,
  ~75 MB/s at 30 fps (was 146 with IR at full rate); old recordings read as before. Stops (DiskFull)
  under 5 GB free, checked every 15 sets; the viewer records at 30 or 10 fps (Capture panel) and shows
  the free space; the live worker's raw copy stops but the scan goes on. Re-recorded marker frames
  decode identically. The record loop writes every depth frame from the stream's kept frames (~0.4 s),
  each paired once colour has a frame after it; simulated 30 fps with jittered arrival and slow
  writes: every frame kept and paired right while a write takes less than a frame; behind that, frames
  are skipped and counted ("missed" in the status line), never mispaired.
- Phase 0.4 measuring (2026-11-01): `bench.py` (stage table: inclusive wall clock per stage, merge
  thread apart, first 10 frames left out, read + decode its own row; peak RSS; GPU memory and busy
  time from /proc/self/fdinfo drm-total-* / drm-engine-*; machine and versions in the JSON).
  compare.py takes meshes (sampled ~1 point per 0.25 mm2), `--seeds`, and `--transform` (reuse a
  placement). ONE BASELINE, board camera, marker recording replayed through captures/baseline/ (a
  symlink to the -rec):
  - replay 179 s, 581 tracked / 207 merged, base found; compare (12 seeds) median 0.246 / p90 0.652 mm,
    92.3% on the part, Creality surface 69.0 / 72.7% within 1 / 2 mm (same as the 11-01 reading)
  - after Optimise (83 s, 1.3 GB, 587 loops kept: 362 markers + 225 features, largest correction
    4.3 mm): median 0.229 / p90 0.562 mm, 92.4%, 66.5 / 68.9%
  - bench captures/bench/20261101-194222-baseline.json: 259 ms per frame; frame levels 85, solve 49
    (tracking step 58), ORB 33, object piece 27, markers 18, base masks 14, depth -> points 11, model 6
    + 15; merge thread 78 per frame (cloud add 31, depth -> points 24, clean 17); read + decode 30;
    peak 3.5 GB
  - evaluate captures/eval/20261101-194223-baseline/report.json: the four older datasets identical to
    the morning's run (to 1e-13); markers 0 jumps, doubled 0.0001, marker scatter 0.94 mm; synthetic
    ATE 1.17 / 1.28 / 0.87 mm
- Phase 0.5 synthetic (2026-11-01): synth.py writes true_surface.ply (object + its markers, first
  camera's frame, cloud.ply convention; older sets get it on first use) and takes `pattern_mm` (a fixed
  per-pixel disparity error, RMS at 20 cm, growing with z^2; pattern of 4 and 24 px blurred noise is
  an estimate), `noise_mm`, `bits=14` and kind "plate" (1.8 mm plate standing, markers both faces,
  path -40..220 degrees round it). The three old sets regenerate byte for byte (21 files) and their
  true surfaces match. evaluate.py: datasets synthfp (box, 14-bit, 0.144 mm pattern, 0.05 mm noise)
  and synthplate, and "truth mm" (cloud to true surface, in place, within 5 mm). Frame 0 alone lies
  0.10-0.13 mm from the true surface with the pattern, 0.25 mm with the old 0.5 mm noise. Baseline:
  synth / synth7 / synthhard truth 0.37 / 0.35 / 0.34 mm; synthfp ATE 0.55 mm, truth 0.12 mm;
  synthplate ATE 10.8 mm, 3 re-finds, 86% on the object: tracking slips where the plate is edge-on (a
  thin-object case for Phase 3; Phase 1's both-faces check fuses it with the true poses).
- PHASE 1 DONE (2026-11-01): meshes on the CPU engine. `fusion.py`: `Engine` (Open3D VoxelBlockGrid,
  0.5 mm voxels, 8^3 blocks, tsdf f32 + weight/colour uint16; depth as float mm, pinhole K) knows
  nothing about the scanner; the Ferret layer removes the lens distortion by a fixed nearest-pixel
  lookup (`Pinhole`), cleans like the merge (prep.clean, only around the label mask), takes colour per
  pixel at its distance, masks by the frame's labels (1/8, grown one cell), skips empty frames. At
  extraction: the cloud's output_mask (cylinder, base, connected to the core) on the vertices, then
  crumbs under 1% of the largest piece dropped. mesh.ply (own writer: float32, colour, normals, metres,
  y up) + mesh.stl (mm) + mesh.json. `cloud.py --optimise --mesh`, Process "Make mesh" (Poisson only
  as a fallback when the recordings are gone), viewer `m` (cloud.py in a capped scope), the 3D page's
  Show Cloud/Mesh + STL download (checked in headless Chrome: renders), `evaluate --mesh`. fuse.py
  keeps only the ICP helpers.
  - Settings measured on the marker scan (compare.py from the optimised cloud's placement):
    truncation 1.5 mm (2 mm shifts the 1.8 mm plate's faces: median 0.034 -> 0.076 mm); keyframes
    only (all 527 frames: p90 0.55 mm; with corrections blended between keyframes 0.53; keyframes
    0.34), so frames between keyframes carry tracking error Optimise does not fix; min weight 4 frames
    (3: p90 0.36, 6: 0.32 but less surface).
  - Pass marks: marker mesh median 0.094 / p90 0.336 mm against the cloud's 0.229 / 0.562: PASS.
    Creality surface within 1 mm 66.1% against 66.5%: MISSED by 0.4 points, all of it a thin band at
    the edge of the scanned area (a fused surface stops about a voxel before the last points; mapped).
    Synthetic, true poses: 0.018-0.025 mm median (<= 0.1: PASS); plate faces 100% / 99.9% within
    0.5 mm (PASS). Time: 58 s for 179 keyframes (prepare 154 ms per frame, integrate 17, reading every
    recorded frame 52).
  - With tracked poses: synthetic meshes 0.055-0.10 mm; the plate loses its back face (78% / 2%):
    tracking, Phase 3.
  - Found on the way: Optimise's spread of keyframe corrections now blends between keyframes within a
    recording (was held constant); 54 tracked frames (256-389) of the marker scan have EMPTY labels
    (scene.masks found no object in them), and the object region's cylinder grew to 317 mm for a 274 mm
    part, so sheets of the surroundings stay in cloud and mesh (5 pieces, 4-7% of the area each;
    visible in the 3D page). FIXED the same evening: see "Object separation", LEAKS CLOSED.
- Phase 0.6 platform (2026-11-01): pyproject.toml has the manifest (core numpy; extra "scan" open3d,
  opencv-python-headless; a "gpu" extra comes with the bake-off) and setup-venv.sh builds a venv from
  it on any distro (`pip install -e .[scan]`; PYTHON=... picks the interpreter; leaves an existing
  directory alone). The user's .venv (made by hand, --system-site-packages) is unchanged. Throwaway
  check with uv's standalone Python 3.14.5 (Tk 9.0) + numpy 2.5.4 + open3d 0.20.0 + OpenCV 5.0.0: all
  tests pass; evaluate on all six datasets equal to the baseline within 1e-11 (markers 244 -> 218 ms
  per frame); the board calibration solved from the 10-30 views agrees within 0.002% (fx 963.53 /
  963.51, law a 289570 / 289564, same 28 views, reprojection 0.346 / 0.319 / 0.340 px vs 0.346 /
  0.321 / 0.340); the viewer's PPM previews and a panel render the same under Tk 9.0. So Ubuntu 26.04's
  likely stack needs no code change today; the checkpoint re-runs this on the real install.
- uv (2026-11-04): the user's .venv is now built by `uv sync --all-extras` from uv.lock (plain
  `uv sync` removes the extras), on the system /usr/bin/python3.12, no --system-site-packages; the
  hand-made one sits in ../tribuutscan-venv-pip-backup until a real scan has run on the new one.
  pyproject `[tool.uv]`: python-preference only-system, and numpy<2 below Python 3.13. Why: the live
  worker (.venv Python) imports the SYSTEM numpy (apt 1.26): spawn re-imports tribuutscan.viewer (numpy
  at the top) before livescan.run reorders sys.path; only Open3D/OpenCV come from .venv. So .venv's
  packages must resolve for numpy 1.x (unpinned, the lock picks OpenCV 5.0, which needs numpy 2). And if
  the worker ever did load a numpy 2, its pickled arrays would not load in the viewer's numpy 1.26
  (checked: "No module named 'numpy._core.numeric'"). The "gpu" extra got `python_version >= '3.11'` (wgpu 0.32 needs
  3.11; 3.10 gets the CPU fallback): uv's lock found it, `pip install .[scan,gpu]` failed on 3.10.
  Lock on 3.12: numpy 1.26.4, OpenCV 4.11.0, Open3D 0.20.0, wgpu 0.32.0; all 137 tests pass (same as
  the old .venv). pyusb (installed by hand before, no code imports it) is not in the new .venv.
- PHASE 2 DONE (2026-11-01 night, the user away): ENGINE = WEBGPU (wgpu-py 0.32, `gpu_wgpu.py`). Shared layout,
  inputs and numpy references in `gpu.py`; the bake-off harness `bakeoff.py` (dump / run / slice); a
  no-scanner self-check for testers `gpucheck.py`. Installed into .venv: wgpu (+ cffi, rendercanvas,
  pycparser). The OpenCL engine (gpu_cl.py; pyopencl + pocl-binary-distribution) was removed after
  the decision, code in commit 401cdf9, packages uninstalled.
  - Layout (both engines, the portability rules): dense u32 brick index over the scan's box -> pool of
    65536 bricks of 8^3 voxels; voxel = u32 TSDF int16 (normalised by the truncation) | weight uint16
    << 16, plus u32 RGB8; 32-bit integer atomics only (CAS allocation, visible-brick list); ICP sums
    reduced per work-group in f32, finished on the CPU in f64; <= 8 storage buffers and <= 128 MB per
    binding (WebGPU's defaults; llvmpipe allows no more). Bricks are allocated as Open3D's DepthTouch
    does: every 4th pixel each way (its down_factor), 4 depths along the ray within the truncation.
    Voxels at origin + index * voxel, x fastest in a brick (= Open3D's order).
  - Bake-off, marker recording's 206 keyframes (board camera), both engines on the A750 and on their
    CPU fallbacks (llvmpipe / PoCL): the same volume as Open3D (10523 of 10532 bricks, the 9 outside
    the box; TSDF median 4e-6, 0.006% of voxels > 0.01; weights differ on 0.06% of 3M voxels), frame
    maps = numpy to 2e-5 mm, ICP sums = numpy to 1.3e-7. Kernel ms on the A750 (30 frames; WGSL after
    unrolling): integrate 0.14 / 0.10 (WebGPU / OpenCL), ray cast 2.65 / 1.10, ICP 0.16 / 0.12, maps
    0.07 / 0.09; ICP pass + readback 0.63 / 0.48; llvmpipe ray cast 96, PoCL 165. OpenCL is faster on
    Intel (IGC vs Mesa's backend; WGSL work-group size and unrolling closed the ICP gap, not the ray
    cast's), but a stock Ubuntu desktop has Vulkan (xserver-xorg / libvulkan1 recommend
    mesa-vulkan-drivers: anv, radv, nouveau/nvk, llvmpipe) and no OpenCL (intel-opencl-icd was
    installed by hand here; AMD needs mesa-opencl-icd + RUSTICL_ENABLE; nouveau none). WebGPU also
    covers Windows (DX12) and macOS (Metal). Decided on reach, the user's first rule.
  - The full slice in WebGPU: frame preparation on the GPU (`set_camera`, `fuse_frame`: prep.clean's
    window / jump / grazing / erosion / 7x7 speckle on the cut-out frame_images uses, the 1/8 labels
    grown one cell, colour through the calibrated colour camera from raw YUYV, the Pinhole lookup) =
    the CPU path pixel for pixel on the marker frames (same pixels kept, depth 3e-5 mm, colour 0.01%
    one-pixel rounding); marching cubes on the GPU (table generated in `gpu.triangle_table`: closed and
    outward on a sphere and a random field) = Open3D's mesh (559734 / 559722 triangles, 0.00003 mm
    median apart). `fusion.open_engine` takes the GPU when wgpu has an adapter (TRIBUUTSCAN_GPU=0: Open3D,
    =cpu: llvmpipe), volume = the scan cloud's box + 20 mm (`scan_bounds`); `read_recording(yuyv=True)`
    leaves colour raw; `scan_frames` yields 1/8 labels now.
  - Pass marks (A750, marker scan, `bakeoff slice`): prep + maps 0.49 ms (<= 3), integrate 0.22 + ray
    cast 2.5-4.3 (<= 10), ICP 0.24 (<= 2), mesh 0.17 s (<= 2); 206 frames in 6 s with reading
    (8 ms a frame on the CPU); `fusion` mesh 17.7-24 s (was ~57; the rest is restoring the scan);
    compare vs Creality identical to Phase 1 (0.090 / 0.343 mm, 95% on the part). Full pool (4000
    bricks): 6517 refused and counted, mesh from what fitted. Compile 0.05 s. Stability: a 10-minute
    loop of 127 full replays, no error, the same 559736 triangles every pass (GPU memory then peaked
    at 4.5 GB: see close() below; with it, 67 replays in 5 minutes held 735-809 MB, host 3.6 GB); the
    desktop was idle (the user away) and the 3D page not open: that part of the mark is not done.
  - FOUND: wgpu-native hands Vulkan timestamps as raw ticks (A750: ~52 ns, calibrated 56-57 with
    submission; llvmpipe ns): `calibrate_ticks`. GPU memory: wgpu-py's device and queue form a
    reference cycle, so engines were freed only by the cycle collector (a replay loop held 3.9 GB);
    `close()` drops the references and collects (back to 1 MB), destroy() alone frees nothing.
  - FOUND: Open3D's integrate takes each voxel's depth from the pixel it falls in ROUNDED DOWN: half a
    pixel off along every slope (synthetic sphere: mesh 0.057 mm median from the truth; a tilted wall
    0.083 mm, exactly 0.5 px x pixel size x slope). The GPU engine takes the NEAREST pixel by default
    (sphere 0.018 mm); `nearest=False` keeps Open3D's rule for the parity checks. Marker mesh vs the
    Phase 1 mesh: 0.0000 mm median with Open3D's rule, 0.020 (p90 0.10) with the nearest pixel (the
    0.05 mark holds either way); vs Creality: median 0.090 -> 0.083, p90 0.343 -> 0.353
    (inconclusive against another scan; the printed box decides).
  - Not done in Phase 2: S1 decode of the raw Y14/Y10 bytes and the garbled-row check stay on the CPU
    (decode_depth); GPU tracking (Phase 3) will use `raycast` / `icp` as they are.
- PHASE 3 DONE (2026-11-02, the user asked for it "without testing": no scanner, offline data only):
  tracking on the GPU. `gputrack.py` (Tracker: its own volume; `kind()`), Scanner track_with="gpu"
  (`gpu_source`, `gpu_step`, `close()`; `judge` and `marker_pairs` now shared with joint_step), track.py's
  solve split (`prior_weights`, `pose_terms`, `gauss_newton`, `summary`; CPU results identical: evaluate
  equal to 1e-14), evaluate / bench `--tracker gpu`. cloud.py (so the viewer's `m`) tracks on the GPU
  when there is one; the live worker and restored / continued scans stay on the CPU tracker (Phase 4;
  session.refind needs Open3D levels). session.json notes "tracker" and "gpu_ms".
  - Contract kept: source = the frame's centre piece (isolate_mask), uncleaned, on the pinhole grid
    (kernel prep_source: raw -> mm where the stride-2 cells are kept, through the Pinhole lookup); model
    = ray cast from the predicted pose of a volume of the MERGED frames, fused as for the mesh
    (fuse_frame: cleaned, labels; colour grey without YUYV or a colour model), the base left out in the
    ICP kernel by the kept plane (height <= 3 mm); markers, prior, LM, gates, ORB start and the Open3D
    re-find unchanged; the VoxelCloud is still merged (doubling stays measurable, outputs unchanged).
  - The solve: gputrack.LEVELS = fuse.ICP_LEVELS (reach 30/12/6 mm), source stride 4/2/1, track's
    iterations; the GPU sums J about the object's centre (float32 stays accurate), the CPU turns them
    into track.solve's origin-based twist and adds markers + prior. Each pixel weighs its footprint
    z^2 / (fx fy) over the level's voxel area: the shape term ends up with about half the CPU's
    information (projected, not surface area; normal gate), and the flat-face count is the same (12 on
    the marker recording), so DEGENERATE and the fitness limits stay.
  - Kernel changes: params 336 bytes (trk: footprint scale, reach, normal step, stride; base plane);
    ICP_VALUES 32 (+ frame pixels, matched pixels; sum r^2 unweighted for the RMSE; 16 kB of work-group
    memory, WebGPU's guaranteed limit); maps' normal step; set_camera works without a colour model.
  - FOUND: after ONE fused frame the ray cast hit only 58% of it: prep.clean's edge-on test (+-1 px)
    drops scattered pixels on noisy depth, erosion widens the holes, and the ray cast wanted all 8 voxels
    seen. Samples now average the seen voxels when they hold >= half the trilinear weight: 100% (the
    Phase 2 wall test: 99.9% within 0.2 mm, max 0.28 at the rim). Tracking volume truncation 3 mm
    (gputrack.TRUNC_VOXELS 6; the mesh keeps 1.5): plate ATE 6.8 mm, 2 mm 19, 1.5 mm 36.
  - Results (evaluate --tracker gpu --against the separation run, captures/eval/20261102-032851-phase3-gpu-final):
    markers 581 tracked, 0 jumps, doubled 0.0001, marker scatter 0.947 -> 0.929 mm, keyframe scatter
    0.868 -> 0.78, re-finds 2 -> 1; ATE synth 1.14 -> 0.37, synth7 1.27 -> 0.34, synthhard 0.93 -> 0.40,
    synthfp 0.55 -> 0.34 mm: every Phase 3 mark met (the full-rate recording does not exist yet).
    End to end on the marker recording (replay + Optimise + mesh, 12 seeds vs Creality): 222 s (was 314),
    largest Optimise correction 4.1 -> 3.4 mm; cloud 98.5% on the part, 0.235 / 0.570 mm, 66.6 / 69.1%;
    mesh 0.083 / 0.346 mm, 66.2 / 68.7% (separation run 0.236 / 0.576, 66.5 / 68.9; mesh 0.090 / 0.340,
    66.1 / 68.5): the same. bench (captures/bench/20261102-024423-gpu-tracking-nostamps.json): 189 ms a
    frame (259); GPU tracking step 20.6 (frame maps 3.1, ray cast 8.3 wall, ICP 8.5 for 9.2 passes, fuse
    1.3; in evaluate's run ICP 10.1 for 9.4: the 10 ms mark is met only just, by wall clock);
    alone a pass is 0.41-0.47 ms and the ray cast 2.7 ms, the rest is waiting on the GIL beside the
    merge thread. Engine(timestamps=False) for tracking: each timestamped submission read the stamps
    back (0.2 ms a pass). The CPU stages left (object piece 30, markers 25, ORB 37, base masks 24 ms)
    are Phase 4's.
  - OPEN: the thin plate (synthplate): while it holds the GPU is far closer (0.1-0.2 mm at the plate's
    centre over frames 0-44, the CPU 0.2-1.1), but twice (frames 38, 48) a marker-map entry 2.9 mm off
    made the marker check reject a correct pose (0.44 / 0.32 mm from the truth), the Open3D re-find then
    took one 2.4 mm off, and at the edge-on turn (frames 78-89, the far face new surface) the re-finds
    were up to 12 mm off and were merged: 157 tracked, ATE 6.8 (CPU 6.3), cloud 56% on the object (CPU
    90%). Neither tracker holds that turn; markers on both faces are the way.
  - OPEN: the sensor's fixed per-pixel error (synthfp): the GPU drifts ~0.7 mm over its first 16 frames
    and keeps that offset (the CPU 0.13), so the cloud's p90 to the true surface is 0.70 mm (CPU 0.33);
    with 0.05 mm noise and no fixed error the GPU is the better one (0.18 vs 0.23 mm). Not the
    iterations (converged), not the frame normals' step, not the balance (the GPU's shape weighs less).
    Measured, off by default (gputrack constants): depth blur in frame and fused frames 3 px -> p90 0.63;
    model normals from ray-cast hits 8 px apart -> 0.42 but the noisy box 0.37 -> 1.74 mm. The real
    marker recording (Y10) shows nothing of it; the full-rate Y14 recording (0.7) will say.
- PHASE 4 GROUNDWORK (2026-11-02, offline, before the user's live session): the live worker tracks on
  the GPU ("start" takes an optional tracker, default "gpu", falls back to the CPU; Stop frees the GPU
  before saving). `bench.py --live` replays a recording through the real worker process (frames.read_raw:
  the capture's raw buffers; two in flight as the viewer, none dropped). Marker recording, live worker:
  CPU tracker 3.7 fps (270 ms), GPU as Phase 3 left it 5.4, now 9.3 (107 ms), all 581 frames tracked.
  Moving frames costs nothing worth fixing (decode 6 ms, waiting 3): the shared-memory ring is not needed.
  - GPU tracker defaults (Scanner segment_stride / orb; the CPU tracker unchanged, evaluate trackers
    "gpu-p3" and "gpu-seg" keep the steps): the object / base search on every 4th pixel (the masks cover
    their 4 x 4 blocks for the merge and the GPU source), and ORB only for keyframes (extracted in the
    merge thread, in order; a retry waits for it, so replays stay identical) and to retry a frame the
    motion guess did not hold (from the last frame with ORB). The marker recording, at 3.3 fps (more
    motion per frame than live), never needed the retry. Evaluate (captures/eval/20261102-044300-p4-orbthread
    against Phase 3): markers 581 tracked, 0 jumps, 0 re-finds, scatter 0.929 -> 0.932, 171 -> 70 ms;
    ATE synth 0.37 -> 0.36, synth7 0.34 -> 0.42, synthhard 0.40 -> 0.33, synthfp 0.34; synth lost 1 of 160.
    The plate is now tracked up to its edge-on turn (81 of 160) and stays lost after it instead of being
    re-found wrong: its cloud is clean (0.13 mm, no doubling; Phase 3 0.82, 2.5% doubled).
  - bench --tracker gpu (captures/bench/20261102-044143-gpu-p4a.json, before the ORB move): 93 ms a frame
    (189); left: markers 23, GPU step 20, object piece 14, keyframe ORB 12 (now in the merge thread),
    base masks 8. Live the merge thread (VoxelCloud merge ~130 ms a keyframe, + ORB) and the main thread
    fight for the GIL: the ray cast and ICP wall times rise with it.
  - End to end (cloud.py --optimise --mesh, 12 seeds vs Creality): 156 s (Phase 3 222, CPU 314); cloud
    98.6% on the part, 0.230 / 0.562 mm, 65.7 / 68.1% within 1 / 2 mm; mesh 0.088 / 0.333 mm, 65.3 /
    67.8% (Phase 3: 0.235 / 0.570, 66.6 / 69.1; mesh 0.083 / 0.346, 66.2 / 68.7): 0.9 points less of
    Creality's surface covered, probably the coarser object masks at the outline (not checked).
  - FOUND: Scanner.merge broke on YUYV frames (only live ones have them) after its pixel-block change;
    test_frames_as_the_live_worker_gets_them_are_tracked_and_merged covers it now.
  - Measured and dropped: marker detection in a helper thread beside the object search (9.3 -> 9.6 fps).
  - Next for 20 fps: the merge out of the tracking process's GIL (or cheaper), the object / base search
    on the GPU; then the live view of the model.
- FIRST LIVE GPU SCAN (2026-11-02, the user, captures/20261102-081622-scan with its raw frames): hand
  model (~22 cm) standing on a tiled floor, LONG range, 35-50 cm, 55 s, no markers, mostly the front.
  The user: result impressive, scanning experience not.
  - Result: clean one-sided hand, 118k points, doubled 0.0019, Optimise 174 loops, largest correction
    1.4 mm.
  - Speed: 483 frames, 9 fps (frame gaps median 100 ms, p90 200, max 1.7 s). Worker per frame:
    tracking 62 ms, raw copy 34 ms (keep raw on), decode 9. GPU wall: ray cast 10.1 ms (2.7 alone),
    ICP 9.9 for 5.9 passes.
  - Tracking: only 132 of 483 frames tracked, 79 merged.
    - 244 frames were "too few points". The source is the piece at the image centre, and the hand is
      narrow: whenever the centre hit the floor beside it (removed as the plane), the piece was a
      1-25 point crumb.
    - Once lost, only "move back" re-finds (relock from the last good pose). The motion guess keeps
      extrapolating the last motion over the lost time (the guess's step at the centre grew 7 -> 177
      mm over 20 lost frames).
  - Offline replays of its raw frames (scratch code, nothing changed in the repo):
    - Crumbs skipped (nearest piece >= 30 points): the wrong piece (wall, box) when it was nearer the
      centre than the hand.
    - Source seeded from the object's region as predicted by the pose guess (cylinder core, the
      frame's own plane cut, centre piece when nothing is in the core): 466 of 483 tracked, 284
      merged.
    - That exposed the next leak, the long-range base: tile / floor points 3-10 mm above the base
      plane (OBJECT_ABOVE_MM 3, set for close range) connect to the hand's foot. The region grew to
      radius 523 mm (live 98), and floor speckle got into the cloud. Cropped to the hand: doubled
      0.009, same shape.
    - Patching scene.py's centre piece as well (crumb skip) let the region grow into the background:
      21% doubled.
  - GPU replays are not bit-identical run to run (the same variant ended x6 / x22 lost). Not traced;
    relock reads the cloud without waiting for pending merges (a suspect). It now flushes first.
- TRACKING ROBUSTNESS (2026-11-02, the user: "follow your schedule", item 1). Offline, on the hand
  recording (evaluate dataset "hand", plus "markersgap" / "handgap": 80 / 100 frames left out, a jump
  to find again; `GAPS`). evaluate has a "region mm" column (the object region's radius: leaks show
  there).
  - Region growth (scene.py GROW_ABOVE_MM 10): only toward points 10 mm above the base. The floor's own
    scatter at 35-75 cm is +-3 mm (p1/p99); cardboard and cables on it 3-25 mm.
  - Tracking source (Scanner.feed, isolate_mask `seeds`): the piece connected to the region's core
    where the pose guess puts it (Scene.core: inside radius - margin, 10 mm up, below top - margin),
    after the frame's own plane cut; the centre piece only until the region exists or with < 20 seeds.
    When the frame shows no table of its own, the kept base (plane_in_camera at the guess, computed
    lazily) cuts instead (`known`): the whole floor was in the piece, and the solve matched 10-28%.
  - The motion guess extrapolates at most GUESS_MAX_S 0.3 s.
  - Pose checks (accepted = solve_ok + plausible_step + floor_agrees):
    - floor_agrees: the frame's own table (table_normal: least squares, narrower extent) must lie
      within 2.5 degrees of the kept base at the solved pose. No verdict when the patch is under
      150 mm across or more than 15 degrees off. Hand: good poses 0.1-1.1 degrees, slid ones
      3.4-8.2. Marker scan: its rim, 80 mm across, read 1.6-3.7 (hence the span rule), and most of its
      frames' own "planes" are the part or the backdrop (57-60 degrees). Tilt only: an object's flat
      top is parallel to the table.
    - relock (Open3D, from the last pose) must also pass the GPU solve's checks (markers): synth
      replayed once re-found a box 8.3 mm off. The marker check rejected the right pose there (a
      map entry 2.9 mm off, the known weakness).
  - refind_anywhere (GPU only) after relock fails: the frame's ORB (whole image in still mode) against
    16 keyframes spread over the scan plus the last 4; the best 3 tried. A candidate needs:
    - 50 RANSAC inliers, 25 of them away from the object (Scene.away);
    - solve_ok, fitness 0.6 and floor_agrees;
    - `reachable` in the time lost: 300 mm/s and 60 degrees/s at the object, plus 50 mm / 15 degrees.
    Then 2 more tracked frames before merging resumes (REFIND_HOLD).
    - Why each rule: the synthetic plate (the same marker grid on both faces, a plain table) was
      re-found mirrored, 180 degrees off, on 31 / 54 inliers with 8-12 off the plate, and with a
      true turn of ~100 degrees the mirror looked like a reachable 79. The real re-finds: hand
      141-245 inliers (49-215 away), 18-31 degrees after 19-27 s.
    - The marker part fills the view (0 away), so markersgap is re-found by relock instead.
    - 200 keyframes instead of 16 recovered one frame sooner at 150 ms instead of 49.
  - Measured and dropped: a residual check against the scan's median (rmse / z^2, 2.5x). The first
    frames match a model of nearly the same view, so the median starts low and the check rejected
    the hand's frames from frame 13 on.
  - Results (evaluate --tracker gpu, captures/eval/20261102-102002-robust-final against
    20261102-085234-robust-before):
    - hand: 133 -> 480 tracked, 1 -> 0 jumps, region 98 -> 121 mm. Doubled 0.0017 -> 0.0185, all at
      the stand's stepped rim, 10-40 mm up (6.8% there, likely real geometry); the hand above
      40 mm 0.1-0.3%. Floor tilt <= 1.15 degrees on every frame. Coverage now the palm side and
      the base too.
    - markers: identical (581, 0 jumps, scatter 0.931).
    - synth sets: equal or better (ATE 0.351 / 0.373 / 0.327 / 0.339); synthplate ATE 7.46 -> 4.11.
    - markersgap 500 of 501, handgap 275 of 383 (without refind 249; it waits for a view it has
      seen).
    - CPU tracker (`current`): synth and synthhard identical, markers 2 -> 1 re-finds; hand 470.
  - Cost: Scene.core 2.7-3.9 ms, table_normal 0.2-1.8, plane_in_camera 1.9-8 (only frames without
    their own table). Whole-run timings swung +-15% with the desktop busy (load 11-15).
  - LIVE SPEED, the next item: bench --live on the hand recording gives 6.3 fps with 480 tracked (old
    code: 12.2 fps with 133). Nearly every frame is now tracked and 60% merged; the VoxelCloud merge
    thread fights the main thread for the GIL; ICP 17.6 ms for 10 passes.
  - Viewer: while scanning, the depth preview shows a centre mark (the first frame's object is what
    it hits) and a red frame while tracking is lost. The hint now says "aim at a part already scanned,
    or move back".
- SECOND LIVE GPU SCAN (2026-11-02 12:13, captures/20261102-121318-scan with raw frames, after 071cdf2 and
  81332d8). The same hand, brighter light, long range 33-39 cm (median 37), azimuth span 88 degrees
  (front and both sides), looking down 47-64 degrees. The user: "really impressed".
  - Tracking: 441 frames in 53 s (8.3 fps), all tracked, 0 lost, 0 re-finds; fitness p5 / median
    0.94 / 0.97, rmse median / p95 0.35 / 0.42 mm; no markers.
  - Cloud: 131 merged, 141,772 points, doubled 0.0062. Optimise: 360 loops (features), largest
    correction 5.4 mm.
  - Mesh (fusion on the GPU, from the scan dir): 219,600 triangles, 177 cm2, 14 s (read 41 ms a frame,
    integrate 4.2). The hand is 198 mm above the floor.
  - Consistency: the morning scan's cloud against this mesh (compare.py, 12 seeds): median 0.233 /
    p90 0.527 mm, all on the part, 91.2% of this mesh within 1 mm.
  - GPU use, a replay of its frames: 441/441 tracked, 131 merged, as live. Measured with timestamps
    (calibrate_ticks: 54.2 ns a tick) and DRM fdinfo sampled every 0.25 s. Per frame, kernels 6.7 ms:
    ray cast 4.3 (windowed), ICP 1.75 (0.21 a pass), prep ~0.4, integrate 0.26 a fused frame. Render
    engine busy 12% of the run; VRAM 1.07 GB (local0), 75 MB system. The worker's 94 ms a frame is
    CPU / Python plus ~25 ms of GPU round trips (wall).
  - README: a section "A live scan on the GPU" with examples/hand-gpu-scan/ (mesh renders only, no
    workspace; numpy z-buffer rasteriser, kept in the scratchpad only).
- LIVE MODEL VIEW (2026-11-02, the user's order: 1 live model view, 2 speed step 2 with a plan first, 3
  accuracy session, 4 Phase 5).
  - GPU kernel PREVIEW (gpu_wgpu.Engine.preview, reference gpu.preview_codes + gpu.voxel_weight). It
    runs after each frame's solve and before its fusion, from the ray cast made for tracking and the
    frame's source maps. One byte per 2 x 2 block (640 x 400; 4 to a word, a 256 KB readback):
    - 0: nothing;
    - 1: the frame's piece with no model ("missing", orange);
    - 2-127: model seen fewer than min weight times (yellow);
    - 128-255: seen enough (grey); the low bits are the facing |n . v| (0.6 without a normal).
    Model hits within OBJECT_ABOVE_MM of the base are not drawn.
  - "Seen enough" is the most of the 8 voxels around the hit (the single nearest voxel speckled), and a
    block takes its first hit among its 4 pixels (pinholes speckled orange).
  - Threshold Scanner.preview_weight = fusion.MIN_WEIGHT x (GPU-fused keyframes / merged frames): the
    mesh fuses only merged frames (at most 4 a second), the tracking volume every keyframe (hand:
    272 vs 131, so ~8). A 0.6 s synthetic scan merges 3 frames, so nothing shows as done (the mesh
    would have nothing either); its test uses merge_min_s=0.
  - Scanner(live_view=True) (the live worker only) sets view_codes on GPU-tracked frames.
    gputrack.Tracker.preview, Tracker.ref = the last ray cast's pose, timing "preview" 0.8-1.0 ms a
    frame.
  - livescan's frame_done now ends with (model, t_depth). The viewer keeps the depth buffers it sent
    by capture time (live["raw"]), and scan_view["model"] = (codes, that buffer). render_depth then
    shows THAT frame (model and depth agree; ~100-200 ms behind, at the worker's rate) and
    paint_model colours it. The stats line carries the legend.
  - Checked: on the hand recording the hand shows grey with yellow along grazing edges. End to end
    with the scanner (no window: capture as the viewer, the worker spawned as the viewer does,
    replies paired by time, render_depth), static at the hand: 190 frames in 14 s, 0 lost, yellow
    turning grey by frame 50; the scanner stayed on USB. bench --live on the second hand recording
    10.9 fps. Not yet seen by the user in the real window.
- TEST BLOCK for item 3 (2026-11-02, prepared while the user is away; it needs their printer and scanner):
  - `testblock.py`: an 80 x 55 x 25 mm block, printable without supports. Steps 8 / 7 / 15 mm, a 45°
    ramp, a dome (R 12, 6 high), bosses Ø16 x 12 and Ø6 x 8, blind holes Ø10 x 8, Ø4 x 5, Ø6 x 6,
    ribs 1 / 2 / 3 mm; asymmetric.
  - The shape is an exact sdf, meshed by marching cubes (`gpu.triangle_table`, numpy, vectorised by
    case) on a 0.25 mm grid offset by 0.37 of a step: grid points exactly on its faces made coincident
    vertices and an open mesh.
  - Result: closed and manifold, faces within 0.10 mm, 536,676 triangles, a 27 MB STL plus
    testblock.ply in captures/testblock/; analytic volume 68,389 mm3 checked in the test.
  - Measured and dropped: Open3D's quadric decimation to 150k triangles. It left the mesh open and cut
    corners by up to 1.9 mm (only the vertices stayed on the shape).
  - One microscopic fold at the dome / base crease (Open3D self-intersection, 2 triangles); slicers
    don't care.
  - Next in item 3: the user prints it, measures MEASURE with calipers and scans it (close and long
    range). A tool then aligns the scan (compare.py's placement) and measures it against the sdf:
    distances, per-face offsets (step heights), best-fit scale.
- PLATFORM SCAN (2026-11-02 17:07, captures/20261102-170723-scan with raw frames, ~46 s, 505 frames). The hand on
  a round white platform, LONG range ~30 cm. The scanner stood STILL and the platform was turned by hand: the
  shelves behind never move (contact sheets), and tracked poses say ~1.1 turns. The live scan ran as "still" (the
  viewer has no mode switch).
  - Colour features came from the static background:
    - the re-find after the edge-on view (lost 92-126) was built on the shelves ("did not move");
    - so were Optimise's 102 loops (all "features"), so the largest correction was 196 mm (a replay: 260).
  - Still a full 360 degree hand (cloud 302k points). But the fingers are doubled on the sides, with a seam where
    the turn closes.
  - The platform top had NO depth at this low angle (white, likely glossy, grazing). So no base was found
    (plane_frames 0), and the object was the centre piece in a ball.
  - Replayed in the P2 turntable mode (Scanner(mode="turntable")): lost 92-262, 82 merged, correction 97 mm. The
    hand has too few colour features of its own to be re-found by features; relock found it a turn later.
  - Turntable scanning needs: a mode switch in the viewer, a shape re-find (as session.refind), object-only
    loops, and markers on the platform at best.
  - SECOND PLATFORM SCAN (17:42, captures/20261102-174213-scan, 94 s, 1109 frames, 11.8 fps). The platform was
    turned slowly; the user said tracking "stopped" at ~90 degrees, where the hand gets thin.
    - The data: all 1109 frames tracked, 0 lost, 170 merged; the tracked turn was ~123 degrees.
    - The hand stopped turning at frame ~1024. The depth outline's overlap with frame 960 levels off there, and
      so does the tracked angle.
    - What got weak: shape degeneracy fell 0.036 -> 0.0036 as the hand turned edge-on. "Flat" was on for 32% of
      frames 900-999 and for all frames from 1000, so the viewer showed FLAT AREA.
    - Cloud doubled 0.45% (first platform scan 9.2%, the 12:13 hand-held one 0.62%). Optimise kept 468 feature
      loops, largest correction 24 mm.
- BRING BACK THE SHAPE (2026-11-02 evening; the user's idea after the platform scans, plan approved): "when tracking
  is lost, bring back the shape: the program fits the already scanned points".
  - Measured first, on replays of the first platform scan, fitting lost frames to the shape so far:
    - edge-on frames fit anywhere (fitness 1.00 at contradictory angles);
    - the hand's unscanned back was placed 10-15 degrees off (0.52-0.68 fitness, 1.4-1.6 mm);
    - in the thin view the weakest direction IS the platform's turn (overlap 0.85-0.92), so tying the motion
      to the platform's axis would not help. Not done.
  - `Scanner.refind_shape` (GPU tracker), after relock and the colour re-find fail:
    - `session.shape_match` (FPFH + RANSAC; `shape_pose` split into `shape_features` / `shape_match`, the
      model's features computed once per merge) against the object so far, refined by the GPU solve;
    - checks: solve_ok, RELOCK_FITNESS, degeneracy >= track.DEGENERATE (edge-on views refused), floor_agrees,
      reachable; then REFIND_HOLD;
    - at most one try per SHAPE_EVERY_S 0.25 s while lost; status "re-found by shape", stats "shape_refinds",
      the worker's end line ("lost N, found again M (K by shape)").
    - The coarse RANSAC placement is 1.7-2.5 mm / 0.9-3.7 degrees (test block); the solve refines it.
  - FOUND: a turn about a vertical axis hides in horizontal faces.
    - The synthetic test block on a platform with 3 s left out (38 degrees): the next frame's solve started at
      the last pose and was ACCEPTED 37 degrees off, and so was every later frame (synthturngap: ATE 50.8 mm,
      17.7% doubled).
    - Fix: the first frame tracked after GAP_S 0.5 s without one is checked by shape (`shape_candidate`, no
      throttle). A candidate that fits clearly better (`fits_better`) and lies elsewhere (`same_pose`)
      replaces the solve.
    - `verify_gap=False` (evaluate tracker "gpu-noverify") leaves the check out: every other dataset came out
      the same, and synthturngap got worse (ATE 110 mm).
  - `Scanner.check_room` ("still" mode, at a keyframe, at most once per second of frame time): is the room
    (neither object nor base, on an 8-pixel grid) of an earlier keyframe found again where it was (it stood),
    or where the tracked motion takes it?
    - `room_verdict`: Open3D nearest distances, ROOM_TOL_MM 4. Only points the two hypotheses tell apart count:
      a point landing on the room as it was says nothing. Without that, a table under a turntable (a plane that a
      vertical turn maps onto itself) let "moved" match half the points of a still room.
    - Sampling: the reference is thinned to 5000 points, the current room is kept whole, both on the same pixel
      grid. Thinning both by index made "still" depend on which samples survived.
    - Cost: run at every keyframe on the whole stride-4 room, it took 10-25% of the live worker's frame rate on
      the hand-held hand2 recording (the camera moves, so nearly every keyframe gave a verdict). After the fix,
      hand2 live (alternating, load 6-14): 12.7 / 12.6 fps against the old code's 12.3 / 11.5; 441 tracked
      against 420.
    - 3 "still" verdicts set `object_turns`:
      - `orb_mask` takes colour features on the object's pixels (keyframe ORB, the ORB start/retry,
        refind_anywhere, whose "away" rule goes too);
      - Optimise matches features on the object only (`optimise.object_features`, `Scene.on_object`: the
        cylinder, or the ball when no base was found);
      - session.json gets "object_turns", and it is restored with the scan;
      - the viewer shows "object turning, scanner standing", and FLAT becomes "THIN VIEW: turn slowly here".
    - Both platform recordings: 3 still / 0 moved verdicts. The hand-held box: no "still" verdict.
  - Optimise refuses loops the shape cannot pin, unless markers held them.
    - On the first platform scan the "near" loops between edge-on views were 8-44 degrees off tracking, at
      degeneracy 0.0003-0.004; the 42 good ones were 0.013 and up. The graph had kept them all.
    - The "largest correction" Optimise reports is the CAMERA's move: with the platform's hand ~330 mm away,
      22 mm there is a ~3.7 degree turn.
  - The shape while lost:
    - `gputrack.Tracker.ghost`: a full-view ray cast at a pose, with the preview codes.
    - `Scanner.show_ghost` on lost frames, including "too few points" ones (their small piece is sourced).
    - After a gap check, the preview is ray cast at the accepted pose.
    - Hint: "TRACKING LOST: turn (or move) back until the live view lines up with the grey shape".
    - Correction to the plan: frames with too few points showed NO model (view_codes cleared), not stale codes.
  - synth kind "turntable" (`--turntable`): two scenes cast from one camera, the nearer hit kept; the platform
    is NO_DEPTH like the real one; long range (the estimated law's window ~204-600 mm); the turn tied to time
    (TURN_DEG in TURN_S). The old sets regenerate byte for byte (box, plate checked).
  - evaluate datasets: platform, platform2, synthturn, and synthturngap (frames 150-174 left out). Gap
    variants of synthetic sets keep their truth (`base_of`). New "corr mm" column.
  - Results: evaluate --tracker gpu --optimise, captures/eval/20261102-195748-turn-final3 against
    20261102-182449-turn-before.

    | dataset | tracked | doubled | Optimise's largest correction | other |
    |---|---|---|---|---|
    | platform (17:07) | 456 -> 325 | 6.0 -> 0.40% | 80.9 -> 0.84 mm | marker scatter 2.86 -> 1.61 mm |
    | platform2 (17:42) | 1109 = | 0.19 -> 0.20% | 24.1 -> 10.9 mm | |
    | synthturngap | 275 = | 17.7 -> 1.5% | 195 -> 2.5 mm | ATE 50.8 -> 1.67 mm, truth 0.50 -> 0.19 mm |
    | synthturn | 300 = | 1.54 -> 1.49% | 2.4 -> 2.6 mm | ATE 1.74 -> 1.68 mm, truth 0.21 -> 0.19 mm |
    | handgap | 315 -> 382 | 0.27 -> 0.39% | 1.7 -> 2.2 mm | the shape re-find comes back sooner |

    - Everything else (markers, hand, the five older synthetic sets, markersgap) is identical in tracking.
    - The loop check drops loops on markers (580 -> 508) and markersgap (493 -> 442), with marker scatter and
      corrections unchanged (2.93 -> 2.85, 2.49 -> 2.55 mm).
    - platform tracks fewer frames because the old code went on with the room's features, at wrong poses.
      The back of the hand stays missing: it was turned past while tracking was lost.
    - Room verdicts:
      - the hand-held sets got "moved" only (markers 32, hand 37, synth sets 6-7);
      - synthplate got 1 "still" and 1 "moved";
      - all three platform sets got 3 "still" against 0.
    - Live worker (bench --live) on the first platform recording: 330 of 505 tracked at 8.7 fps (lost frames
      cost relock, the re-finds and the full-view ghost), Optimise 42 loops, 0.9 mm. The old code: 485 tracked
      at 12.1 fps, 69 mm.
- THIRD PLATFORM SCAN (2026-11-02 20:31, captures/20261102-203145-scan, raw frames kept; the first with "bring back
  the shape"). The user: "WOW! Look at what I did!"
  - The scan: 730 frames in 79 s; the platform noticed 6 s in; 362 degrees of turn. Cloud 218k points, 0.12%
    doubled (the cleanest hand yet). Optimise 95 loops, largest correction 4.5 mm.
  - How it was used: lost at about -110 degrees (the thin view); the user turned back to the grey shape (found
    again at -112), went back to -52, then forward through the thin view without a loss (260 degrees in one
    stretch) and ended where the scan started. At about frame 540 the scanner was lifted for a pass from above
    (the platform top then has depth).
  - Mesh (fusion, in a scratch copy): 483k triangles from 124 frames, 13 s on the GPU. Palm and stand clean;
    small holes in the finger backs, ragged fingertip edges in the side views (the thin view).
  - STILL LOST: frames 123-183 (edge-on) and 545-616 (the pass from above). No base was ever found (the platform
    top had no depth during the first VOTE_LIMIT 300 frames), so the tracking source is the view-centre piece
    (`Scene.core` returns None without a base). From above the centre hit the room (~665 mm) or nothing, leaving a
    crumb of 2-36 points. Edge-on, it fell between the fingers.
  - DONE (2026-11-03, the user's "Continue"): `Scene.core` without a base returns the points within the ball's
    start radius (ball[2]) of its centre at the pose guess. feed seeds isolate_mask with them, after the
    frame's own plane cut as before, and falls back to the view-centre piece under MIN_SEEDS.
    - This scan replayed: 729 of 730 tracked (live 585), 165 merged (124), Optimise 206 loops, largest
      correction 5.2 mm, 258k points, 0.04% doubled.
    - evaluate (captures/eval/20261103-085856-ballcore-only against 20261102-195748-turn-final3):
      - platform: 325 -> 492 tracked, doubled 0.40 -> 0.0%, Optimise 42 -> 115 loops (correction 3.1 mm);
      - markers: keyframe marker scatter 0.785 -> 0.748 mm (its first frames are in ball mode);
      - every other dataset identical or within run-to-run noise (synthhard 0.468 mm, unchanged).
  - MEASURED AND HELD BACK: the ball's MERGE labels following the tracking piece (`_ball_mask` given `masks`'s
    `piece`, instead of its own centre piece).
    - It merged the pass from above too (286k points, a fuller mesh: 511k triangles from 163 frames), but
      left a ragged ring of platform top around the stand's foot in the mesh. Without a base nothing removes
      the platform once it shows depth. Candidate fix: look for the base after VOTE_LIMIT while none is
      known.
    - It also skipped `_ball_mask`'s own isolate_mask, so the scene's RNG draws changed, and synthhard ran
      away (ATE 0.47 -> 122 mm, reproducible). The same code with the old draws: 0.468 again.
  - FOUND: synthhard's slide (frames 54-72, the scanner coming back from beyond the window), a fragility that
    other RANDOM DRAWS can tip.
    - The true motion slows to ~0.5 mm a frame, but each solve keeps taking the last motion's ~14 mm (the
      guess plus the prior). On the box's flat faces the shape cannot push back, and fitness ~0.6 passes.
    - 8-9 markers are detected per frame but none are used: at the drifted pose none match the map, so the
      marker check never fires.
    - Not fixed. Candidate: refuse a frame that shows >= 3 markers when the map has some in view yet none
      match, then re-find it by markers.
- FOURTH PLATFORM SCAN (2026-11-03 09:22, captures/20261103-092240-scan). 1290 frames at 14.5 fps, 0 lost, 380
  degrees, nearly level (2-9 degrees up, a brief raise to ~31 at frame ~1100), the hand ~40 cm away. Optimise
  largest correction 0.6 mm. Mesh 515k triangles. Fingertips open (no view from above); ragged platform bits at
  the stand's foot; a softer palm than at 34 cm. The user: fast and good models first, accuracy after the phases.
- COMPLETE, CLEAN MODELS RIGHT AFTER STOP (2026-11-03, plan approved):
  - `Scanner.marker_check` (GPU tracker; MARKER_CHECK_MM 5 / DEG 2 / MIN 3). When an accepted solve used none
    of the >= 3 markers the frame shows, `marker_map.match_global` places the frame by them. If that disagrees,
    the solve is redone from the marker pose: taken ("re-found by markers", motion reset) when it holds >= 3
    markers, else the frame is lost. It stops synthhard's slide (frames 54-72: the guess carried on ~14 mm a
    frame after the scanner stopped, the box's flat faces did not resist, 8-9 markers in view matched nothing).
    Now synthhard holds under other random draws.
  - Base votes go on after VOTE_LIMIT while none is known, on every VOTE_LATE_EVERY (10th) frame. When the
    centre piece doesn't touch the plane, `stands_on` takes tracking's piece (from above, the middle of the
    view missed the hand). A late base fits the cylinder on tracking's piece.
  - While no base is known, `_ball_mask` takes tracking's piece (`masks(piece=)`, as the cylinder path
    already did), so the pass from above is merged.
    - It also saves a plane search + flood fill a frame: bench --live, the 09:22 recording, 500 frames,
      alternating with 1c2aab1: 20.4 / 20.1 fps against 15.9 / 15.1.
  - The live worker makes the mesh after Optimise: ("stop", mesh); "optimised" says a mesh follows, then
    "meshed". The viewer's Scan panel has "Mesh after Stop" (default on) and the state "meshing".
  - Replays:
    - 20:31 scan: base found at frame ~550 (the pass from above), 730/730 tracked, platform ring gone from the
      mesh. Fingertips still a little open (a short, shallow pass).
    - 09:22 scan: base found at ~1080 (its brief raise), 1289/1290, Optimise 3.6 mm, doubled 0.31%, the foot
      clean.
  - evaluate (captures/eval/20261103-094020-topclean against 20261103-085856-ballcore-only):
    - synthhard 0.468 -> 0.457 mm (1 marker re-find);
    - synth 0.356 -> 0.333 (kf 0.45 -> 0.35);
    - platform2 Optimise correction 11.1 -> 9.0 mm;
    - the rest identical or within noise.
  - End to end with the scanner (still scene): 187 frames in 12 s; Stop gave saved, optimised (1 s) and meshed
    (3 s: 133k triangles, mesh.ply / mesh.stl).
- PRINTABLE MESHES (2026-11-03, plan approved; Phase 5 part 1): mesh.stl is now a CLOSED solid (`solid.py`);
  mesh.ply stays the scanned surface.
  - Why: the fused mesh is open where nothing looked: fingertips, the stand's bottom, the wrist/rim gap, ~100
    pinholes. The 09:22 replay had 102 boundary loops (11 over 10 mm) and non-manifold vertices.
  - Measured and dropped: Open3D's fill_holes (VTK): 110 s, still open, non-manifold and self-intersecting.
  - `solid.close`:
    1. screened Poisson (depth 10) on the fused mesh's own vertices and normals: the scan lies a median 0.007
       mm from it (p95 0.04);
    2. its largest piece after Open3D's clean-ups;
    3. with a base plane (`scene.plane`, when the base is left out), cut there (crossing triangles split) and
       each cut loop capped flat by ear clipping. Poisson otherwise bulges below the stand into a toothed
       fringe, where most of its defects were;
    4. fan fills for loops up to 200 edges;
    5. `weld`: vertices float32 cannot tell apart are merged, as an STL reader joins them by position.
       ON_PLANE_MM is 1e-3: at 1e-6 a cut point fell beside a vertex, and the STL had 2 collapsed
       triangles;
    6. `repair`: a pinched spot is opened a little wider and filled again (one Poisson spot 136 mm up the
       hand: 6 open edges);
    7. `edges_closed`: every edge in two triangles, once each way.
  - Report in mesh.json "solid": the added area beyond 1 mm of the scan, and its largest distance.
    `fusion.make_mesh(close=True)`, `fusion --open`; the viewer's "meshed" line says closed or not.
  - The two platform replays: closed (the STL read back as a slicer joins it), 693-705k triangles, +8-9 s.
    11-12% of the area added (bottom cap, fingertips, rim gap), at most ~24 mm out. Volumes 392.7 / 398.6 cm3
    (the same hand, two scans, 1.5% apart).
  - Left: 37 tiny self-intersections (71 triangles, 3.6 mm2; Poisson folds between close parts). Slicers
    ignore them. Open3D needs 199 s to find them, so they are not checked routinely.
  - evaluate --mesh (synthetic, captures/eval/*-closed-mesh): every closed mesh is edges_closed. On observed
    parts it is as close to the truth as the open mesh (median 0.055 / 0.052 / 0.118 / 0.106 mm against
    0.061 / 0.049 / 0.113 / 0.099). The added surface is a smooth guess. synthturn found no base, so its
    bottom is Poisson's bulge.
  - FOUND: evaluate's true-pose fusion (whole frames, Open3D CPU grid) SEGFAULTED on synthturn: the turntable's
    still room smeared round the object and overran the voxel blocks. For turntable sets it now fuses only the
    tracked run's labels.
  - End to end with the scanner: Stop, then meshed 6 s later, "closed for printing".
  - With no base on a platform scan, the cut is square to the axis the object turned about
    (`Scanner.turn_axis`: the mean rotation axis of the tracked poses against the first, turns of 20-150
    degrees, pointing the way the cameras' up does), at the lowest scanned point (`fusion.printing_plane`).
    - synthturn's closed solid: 98.5% of the true block within 1 mm (Poisson's own bottom: 71.7%), added 29%
      (51%), median 0.104 mm.
  - solid.ply (Phase 5, 2026-11-03): the closed solid with colour (Poisson's interpolated vertex colours), metres
    and axes as mesh.ply, for Blender; only when the solid closed (a stale one is removed). The 3D page: Show
    Solid, Export "Download solid.ply" (/solid.bin, /solid.ply; info.json "solid" = its mtime). The 09:22
    platform replay: 18.7 MB, colours as on the open mesh (grey fingertips are what the scan saw).
  - CROP (Phase 5, 2026-11-03): the 3D page's Crop › Crop box: a box (Box3Helper wireframe) with six sliders
    and a live preview (three.js clipping planes on the cloud and mesh materials); Apply posts
    /process/crop?box=lo xyz,hi xyz (mm along the box's axes), Clear crop /process/crop-off.
    - `Scanner.crop` {"axes", "lo", "hi"} in session.json; `in_crop` is part of `output_mask` (before the
      connectivity step), so cloud, live view, mesh vertices, solid and STL follow. Continued and merged
      scans drop it.
    - Axes (`cloud.crop_axes`): y along the base's normal, else the platform's turn axis (`cloud.turn_axis`,
      now a function), x the first camera's x laid square to it; else the first camera's axes. session.json
      "crop_axes"; for older sessions the file server works them out from scene.json + frame poses
      (`saved_crop_axes`, equal to the restored Scanner's on three replays).
    - The solid is cut flat at every box face the mesh comes within CROP_CUT_MM 1 of
      (`fusion.printing_planes`, solid.close takes a list, each cut capped before the next); unless the
      bottom face is one of them, the base / turn-axis cut too. A cut side left to Poisson: 26% added, up
      to 30 mm out; cut flat: 21-24% (the caps), 14 mm.
    - A face left at 0 stands 5 mm clear of the cloud: the fused mesh reaches up to 1.3 mm past the cloud
      (fingertips, both platform replays), and a face 1 mm out trimmed it. With a saved crop the sliders
      start at its faces.
    - After Apply the mesh is made again when there is one (09:22 replay copy: 31 s; 184,949 points kept);
      Clear crop gave back the same 334,801 points and 491,968 triangles. Checked in headless Chrome:
      the box, the clipping, and the page's own Apply.
  - FIXED with it: `Scanner.restore` now takes the saved scan's depth mode (`use_mode`; continue_from keeps
    the scanner's). The stand-alone 3D view's Process actions (Optimise again, Merge again, base, crop)
    re-saved a long-range scan under the close law's mode and camera fingerprint, and its mesh was then
    refused ("tracked with the board camera (...), this is the board one (...)").
    - The 17:07 replay (no view from above, no base): closed, flat-bottomed, 14% added.
- Optimise's message (2026-11-03) gives the largest correction AT THE OBJECT: how far a keyframe's view of the
  object moves in the world, plus the turn in degrees. The camera's move ("largest_correction_mm") is still in
  optimise.json and evaluate's "corr mm"; "corr obj" is the new one. First platform recording replayed with
  all of 2026-11-03's code: 505 of 505 tracked (live 465), Optimise 117 loops, 0.32 mm at the object (0.51
  degrees), 2.99 mm at the camera.
- PHASE 5 DONE (2026-11-03, plan approved: "Finish Phase 5"): finishing. Each part has its own entry.
  - Mesh after Stop, the closed STL (PRINTABLE MESHES), the late base and the pass from above (COMPLETE, CLEAN
    MODELS), the correction at the object, the mesh at Optimise's poses (fusion.scan_frames uses the corrected
    poses).
  - Hand-held live scans at 20 fps (LIVE SPEED, step 5).
  - solid.ply with colour for Blender, and the crop box in the 3D view (both under PRINTABLE MESHES).
  - Per-mode voxel size, MEASURED AND NOT TAKEN: both platform replays (long range, ~30-40 cm) fused at
    0.5 / 1.5 mm truncation, 0.75 / 2.25 and 0.75 / 1.5 (scratch only). The 0.75 meshes lie 0.014-0.027 mm
    (median) from the 0.5 ones. The two scans agree with each other no better: 0.160 / 0.498 mm (median /
    p90, 12 seeds) at 0.5, 0.166 / 0.526 at 0.75 / 2.25, 0.160 / 0.502 at 0.75 / 1.5. Half the triangles
    (481-492k -> 211-217k) and 20-30% faster (9.4-14.1 -> 6.4-11.1 s). Open boundary edges 5220-5329 at 0.5,
    2290-3030 at 0.75 / 2.25; 0.75 / 1.5 has more small holes (131-282 loops against 100-125). Not better,
    so 0.5 mm stays for both modes; the closed solid fills the holes either way.
  - End to end with the scanner (no window, the real worker, 12 s still scene): 163 frames, 0 lost; Stop gave
    saved, optimised (1 s), meshed (6 s): mesh.ply, mesh.stl (closed) and solid.ply.
  - Full suite 137 tests pass; the turn-axis test no longer needs Open3D (it failed under /usr/bin/python3
    since 920d771).
  - Next: the user's live check of it all (platform: one level pass and one ~45 degrees down at 30-35 cm;
    Blender with solid.ply; the STL in their slicer); then the accuracy session (test block); later texture
    tricks (the LATER note under Roadmap); 30 fps needs the 6x6 solve on the GPU.
- LIVE SPEED, step 5 (2026-11-03, Phase 5, plan approved): hand-held scans at 20 fps, tracking unchanged.
  - The room test settles: after ROOM_SETTLED 5 "moved" verdicts and no "still", check_room stops (each check
    ~80 ms of nearest-point trees, ~4.5 ms a frame over a hand-held scan; a platform scan stops at object_turns).
  - One axial geometry per frame (`scene.axial`, `Scene._axial`, cached by the points' id, the transform and the
    region): height above the kept base and squared radius from the region's axis, computed once at the pose
    guess and shared by table_in_camera, core and the source cut; masks computes it once at the solved pose.
  - Markers when none are seen: after MARKER_IDLE_FRAMES 30, every 8th frame (was every 4th after 15).
  - evaluate (captures/eval/20261103-122738-speed5 against 20261103-094020-topclean): every dataset identical
    except synth (158 vs 159 tracked) in that run; run alone, new code and HEAD both gave 159 / ATE 0.333 twice
    each (the full run shared the GPU with a fusion job). evaluate's ms a frame hand 47.9 -> 40.9, markers 77 -> 74.
  - bench (hand2, captures/bench/20261103-141209-stages-new vs -141252-stages-head): feed 50.0 -> 37.5 ms a frame
    (base masks 6.2 -> 2.6, markers 2.9 -> 1.8). bench --live hand2, alternating with HEAD at load 6-7: 20.2 /
    20.9 fps against 17.8 / 17.7, 440 tracked each.
  - Next lever (not done): the 6x6 solve on the GPU (whole ICP levels in one submission; the wgpu layer is ~11 ms
    a frame, ~7 of it GPU work).
- LIVE SPEED, step 4 (2026-11-02, plan approved): a live frame's depth decoded where it is used.
  - The decode paths:
    - `depth.decode_strided`: the first pixel of each packed group, every 4th row, plus the
      garbled-row check; it equals `decode_depth(...)[::4, ::4]`;
    - `depth.LazyFrame`: a dict whose "depth" is decoded on first access; it pickles;
    - livescan.Reader builds LazyFrames (raw bytes, depth4, cleared rows, shape);
    - `frame_points(frame, 4)` takes depth4; `frame_shape` / `luma_of` take the shape without
      decoding;
    - GPU: `Engine.unpack` (kernel PREP_UNPACK, Y14 / Y10 onto the Y10 scale, cleared rows zeroed)
      fills the buffer `track_source(None)` and `fuse_frame(depth_ready=True)` read;
    - `Tracker.depth(frame)` reads it back;
    - `Scanner.depth_of` serves markers / orb_guess / refind from it;
    - `frame_markers` skips `locate` when nothing was found;
    - the merge process gets the frame without its floats and decodes its own copy.
  - Garbled-row check: now every 4th column with GARBLE_MIN 10 (was every 2nd with 20). On the marker
    recording the same 44 garbled frames are flagged; after the 8-row margin, 5 frames differ by one
    row (2 more, 3 fewer of ~1300 cleared); nothing on either hand recording.
  - Tests:
    - the strided decode against the full one, Y10 / Y14, garbled;
    - the GPU unpack = decode_depth on the GPU and on llvmpipe; fusing from it = fusing the floats;
    - a 14-bit synthetic scan fed as LazyFrames tracks bit-identically to decoded frames (poses, cloud,
      keyframes), and with the merge process the tracking process never decodes the full depth.
  - Measured: "decode beside tracking" 13-15 -> 3 ms. But fps barely moved: bench --live on hand2
    alternating with 1e5ab5d gave 14.3 / 15.3 against 13.9 / 14.5. Numpy's bulk work releases the
    GIL, so that decode was not costing tracking much: an estimate in the plan that did not hold.
  - The desktop was loaded (8-15: CrealityPrint 55% CPU, WebKit). At that load, in-process tracking
    is thread 69 / merge process 51 / no merging 43 ms a frame (at load 3.4 it was 58 / 42 / 36). Live
    is ~62-67 ms, so ~10-15 ms of live-only overhead remains (replies with the model codes, view
    masks, unpickling 4 MB frames).
  - No end-to-end scanner run for this step: the user was at the PC (CrealityPrint open), and bench
    --live drives the same worker with the scanner's own Y14 frames.
- LIVE SPEED, step 3 (2026-11-02, plan approved): the point-cloud merge and the keyframe ORB in a process of
  their own (`merger.py`; `Scanner(merge_process=True)`, the live worker's scans; replays, evaluate and bench
  keep the thread).
  - The pieces:
    - `cloud.merge_points` / `cleaned_frame` / `marker_list` are the merge's pure part, shared by both
      paths;
    - the process holds the VoxelCloud and merged_data and replies per merge (cubes kept, raw points,
      markers, the keyframe slot's ORB);
    - `Scanner.merged_reply` applies the replies;
    - `sync_merges` waits mid-scan (relock, model_ready, orb_guess);
    - `flush` pulls the full state at the end (save, Optimise, the mesh work unchanged);
    - `cloud_points` gives relock a snapshot;
    - `live_points` gives the 3D page the latest snapshot, never waited on (it runs on the page's
      thread).
  - FOUND 1: replays switch to the recording's depth mode only as its frames are read, after the
    Scanner started. The process had the close law, so long-range hand frames were merged at the
    wrong scale and tracking fell apart after its first loss (481 -> 48 tracked). Every job now
    carries calib.CAMERA.mode.
  - FOUND 2: the live worker is a daemonic process, and multiprocessing refuses those children: live
    runs silently merged on the thread (the fallback logged it). Its flag is lifted around the start
    only (`_config["daemon"]`, private, hence the fallback); the merge process ends by itself when its
    parent goes (end of queue / parent check every 2 s).
  - A "ready" handshake at start (60 s, else the thread). A process lost mid-scan is caught by any
    wait or the per-frame poll: merging goes on in the thread, the lost merges are logged, and the
    mesh is unaffected (fused from the recording with the poses kept in the tracking process).
  - Pitfall hit: a script piped on stdin cannot spawn (the child re-runs the main script by path). The
    test runner needs a `__main__` guard, or spawned children re-run all the tests.
  - Equal: evaluate `gpu-proc` against `gpu` (captures/eval/20261102-155411-step3-proc2 against
    20261102-152200-step3-thread): every metric identical on all 9 datasets. Tests compare cloud
    arrays, merged_data, keyframe ORB, markers and poses bit for bit.
  - Speed:
    - in-process feed of the same 270 frames: thread 58-60 ms, process 42-43 ms, no merging 36.5 ms;
    - bench --live on the second hand recording, alternating with edf02ae: 12.4 / 12.5 -> 15.0 /
      15.5 fps (tracking 78 -> 63 ms a frame).
  - End to end with the scanner (read-only, no window, through the real worker): 192 frames back in
    16 s, the model view going grey, Stop saved (state pulled), Optimise ran, nothing left running.
  - STILL BETWEEN LIVE AND IN-PROCESS (~20 ms a frame): the reader thread's decode (14 ms of numpy
    beside tracking: GIL) and the frames' pickling. Next lever: send raw depth to the GPU (unpack
    there) and to the merge process (decode there), with only a stride-4 decode for the object search
    on the CPU.
- LIVE SPEED, step 2 (2026-11-02, plan approved: ~/.claude/plans/happy-wobbling-journal.md).
  - Measured first, per frame, no profiler, hand frames: decode 8.1 ms; depth -> points 7.3 (6.2 of it
    a colour lookup feed never used); RANSAC table 6.9; kept-base re-fit in the masks 6.3; masks
    10.3; core 2.6; table normal 1.0.
  - Measured and dropped: persistent staging buffers for GPU readbacks (0.25 vs 0.26 ms a read; the
    ~1.2 ms per ICP pass is wgpu-py's Python side).
  - The changes:
    - feed's frame_points(colour=False);
    - Scene.table_in_camera: the kept base re-fitted ONCE at the pose guess (eigh of the covariance
      of at most FIT_POINTS 5000 band points), returning (n, c, narrower extent, fitted). It serves
      the source cut (isolate_mask(plane=): no RANSAC), the floor check (frame_table) and
      scene.masks(plane=). RANSAC + table_normal remain before the base is found, in the other
      modes, and when the re-fit is not taken (the guess was off);
    - livescan.Reader: the worker decodes the next frame on its own thread while it tracks the one
      before.
  - FOUND on the way: synthhard's jerk frame 49 was tracked 10 mm off (rmse 1.36 mm), fused into the
    GPU volume, and tracking stayed ~4 mm off (ATE 0.28 -> 2.78). The previous code got that fragile
    frame right by chance. Now GPU_MAX_RMSE_MM 1.0, growing with (z / 450 mm)^2, is part of
    solve_ok for the GPU tracker (accepted, relock, refind); good frames are at most 0.46 mm.
  - Measured and dropped: gating fusion on fitness < 0.7. It starved the synthetic boxes' models from
    frame 1, since a view the model does not cover yet matches little.
  - Evaluate (captures/eval/20261102-141633-oneplane-rmse against 20261102-115226-speed-final): every
    dataset equal or better; tracked hand 481, synthhard ATE 0.285 (0.28), synth 0.372 -> 0.296;
    evaluate's ms a frame hand 70 -> 53.
  - Speed:
    - bench stage table (hand2): feed 56 ms (depth -> points 1.9, object piece 4.5, masks 9.0, GPU
      step 16.2);
    - bench --live on the second hand recording, alternating with the previous code at load 7-11:
      9.2 / 9.3 -> 11.8 / 12.0 fps.
  - THE NEXT LEVER: in-process feed of the same 270 frames is 35 ms a frame with no point-cloud merges
    against 58-71 ms with them. The merge thread (cloud add ~64 ms and keyframe ORB ~51 ms a merge,
    depth -> points, clean) takes 25-35 ms a frame from tracking through the GIL. The half-resolution
    ray cast (plan item 4) would save 3-4 ms, so it was not done. Next step: the point-cloud merge
    out of the tracking process (plan first).
- LIVE SPEED, step 1 (2026-11-02, item 2). The desktop was busy all along (load 8-20: Firefox at 90%),
  so timings swing +-15% between runs. Compare only runs made back to back.
  - The ray cast covers only the piece's pixels plus a margin (the coarse level's 30 mm reach at the
    piece's nearest point, + 16 px), and the rest of the maps is cleared: Params.rect (PARAMS_BYTES
    352), Engine.raycast(rect=), Tracker.source(near_mm=). Hand: 11.9 -> ~6-7 ms; tracking identical.
  - MERGE_MIN_S 0.25: the point cloud (and Optimise's keyframes with their ORB) takes a keyframe at
    most every 0.25 s of frame time. The GPU tracking volume still fuses every keyframe (3 mm / 2
    degrees): capping that too made synth ATE 0.35 -> 2.6 mm. Result: tracking identical; clouds
    cleaner (doubled hand 1.85 -> 0.5%, synth 1.3 -> 0.4%; truth mm 0.33 -> 0.27, synthfp 0.13 ->
    0.10); handgap 275 -> 314 tracked. Synthetic meshes from 28 frames instead of 67: synth 0.056 ->
    0.066 mm, synthfp 0.084 -> 0.049 (p90 0.93 -> 0.19), plate 0.118 -> 0.113. The marker recording
    (3.3 fps) is barely touched. Tracker "gpu-allmerge" keeps the old rule.
  - Live raw copy on its own thread (livescan.RawCopy, a 30-set queue: no frame dropped, a full disk
    stops it): it took 34 ms of each frame in the live session.
  - Marker search on every 4th frame after 15 frames without markers (live detection only; evaluate's
    cached detections are unaffected): hand 7-12 -> 2.4 ms.
  - scene.masks takes the tracking piece (`piece=`) instead of its own centre piece: one or two flood
    fills less (markers: scatter 0.931 -> 0.934, region 180 -> 182 mm).
  - gputrack.ITERATIONS (3, 3, 8) instead of (5, 5, 8): 10 -> 8.3 passes per frame, ATE synth 0.35
    -> 0.37, synthhard 0.33 -> 0.28 mm. (2, 2, 6) lost synth (2.4 mm).
  - The live worker read the cloud's point count under the cloud lock every frame, and a merge holds
    it ~90 ms: now Scanner.points_kept, updated by each merge.
  - Measured and dropped: sys.setswitchinterval(0.5 ms) changed nothing; the wall-clock gaps were not
    a GIL convoy.
  - bench --live on the hand recording, back to back at load 10-13: 6.3 -> 9.5 fps, 480 tracked
    (tracking 135 -> 84 ms a frame). Marker recording: 8.5 fps at that load (9.3 this morning, idle).
  - Stage table on the hand (bench, 66 ms a frame): object piece 12.7, base masks 14, GPU solve 17.6
    (ray cast 6-7, ICP ~1.4 ms a pass under load), depth -> points 5.7. Merge thread: 191 ms a merge
    (cloud add 91, ORB 45, depth -> points 26, clean 22). Next for 20 fps: the object / base search on
    the GPU, fewer readbacks per frame.
- PHASE 6 DONE (2026-11-04, plan approved; the user moved to uv and keeps performance for their own plans):
  - uv builds .venv (b880376; setup-venv.sh removed in 8951e5c).
  - The tracker variants above and unused code removed (5421e39). evaluate on all 13 datasets against HEAD
    (captures/eval/20261104-100143-p6-before-gpu, -101233-p6-before-cpu against -101245-p6-after-gpu,
    -101700-p6-after-cpu): tracking identical with both trackers. The only differences (points +-5, marker
    scatter in the 12th decimal, one handgap jump 71.0 / 70.9 mm) are HEAD's own run-to-run noise: HEAD run
    again (-102929-p6-head-again) gave the after-run's values.
  - `--force-cpu` (8951e5c): viewer, cloud.py, fusion.py. End to end with the scanner (no window, the real
    worker, keep raw on): forced, 35 frames all tracked on the CPU solve (3.4 fps beside two evaluations),
    mesh by open3d-cpu, closed; plain, 163 frames all tracked at 13.5 fps, mesh on the GPU, closed.
  - FOUND in viewer.log: the user's 11-03 14:48 platform scan (1613 frames, 341 merged) got no mesh: GPU
    marching cubes overflowed its 128 MB output (2.2 million triangles; the base and surroundings are cut
    after extraction). Fixed (3103d53): the bricks go in halves until each part fits, and the parts join
    into the one-pass mesh (test). A copy of that scan re-meshed: 498,562 triangles after the cut, closed
    solid (9.7% added), median 0.33 mm / p95 0.71 mm from its own cloud.
  - Tests: 139 pass in .venv and in a fresh `uv sync --all-extras` environment; 138 under /usr/bin/python3
    (no Open3D: the GPU tests pass by).
  - CLAUDE.md shortened to the working notes; this file got the log.
