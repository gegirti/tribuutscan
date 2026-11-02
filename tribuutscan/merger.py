"""Merging into the point cloud in a process of its own (the live worker's scans).

On a thread beside tracking the merges (VoxelCloud.add, a frame's points with normals, the marker
search) and the keyframes' ORB took ~26 ms of every tracked frame through the GIL: 35 ms a frame
without them against 60 with them (the second hand scan, 270 frames). Here they run in another
interpreter, with the same code (cloud.merge_points, features.extract), so the cloud comes out the
same; cloud.Scanner(merge_process=True) sends the jobs and reads the replies.

Messages in:  ("merge", frame, index, pose, pix, step, orb, depth mode) with orb None or (keyframe slot,
              mask pixels); the depth mode is calib.CAMERA's when it was sent (a replay switches to the
              recording's mode only as its frames are read: with the mode of the process's start, long-range
              frames were merged on the close-range law and tracking fell apart after its first loss)
              | ("points", min_seen) | ("state",) | ("quit",)
Replies:      ("ready",) once started | ("merged", index, cubes seen MIN_SEEN times, raw points, markers or [],
              orb slot or None, ORB features or None) | ("points", (xyz, rgb, seen)) | ("state", cloud arrays,
              merged_data)
"""
import multiprocessing
import queue
import time

import numpy as np

CLOUD_ARRAYS = ("keys", "xyz", "rgb", "n", "nrm", "seen")
READY_S = 60.0  # the process must report itself started within this


def run(jobs, replies, point_mm, stride, clean, spec, mode):
    """The process: one VoxelCloud and the merged frames' points, built from the jobs in order."""
    from . import calib
    from . import cloud as cl
    calib.use(spec)
    calib.use_mode(mode)
    features = None
    cloud, merged, found = cl.VoxelCloud(point_mm), {}, []
    parent = multiprocessing.parent_process()
    replies.put(("ready",))
    while True:
        try:
            msg = jobs.get(timeout=2.0)
        except queue.Empty:
            if parent is not None and not parent.is_alive():  # its scanner's process is gone: nothing to wait for
                return
            continue
        except (EOFError, OSError):
            return
        kind = msg[0]
        if kind == "merge":
            _, frame, index, pose, pix, step, orb, frame_mode = msg
            if frame_mode != calib.CAMERA.mode:
                calib.use_mode(frame_mode)
            mxyz, mrgb, mnrm = cl.merge_points(frame, pix, step, stride, clean)
            merged[index] = (mxyz.astype(np.float32), mrgb, mnrm.astype(np.float32))
            cloud.add(mxyz @ pose[:3, :3].T + pose[:3, 3] * 1000, mrgb, mnrm @ pose[:3, :3].T)
            if not found:
                found = cl.marker_list(frame)
            feat = None
            if orb is not None:
                if features is None:
                    from . import features
                feat = features.extract(features.luma_of(frame), frame["depth"], orb[1])
            replies.put(("merged", index, int((cloud.seen >= cl.MIN_SEEN).sum()), cloud.raw_points, found,
                         None if orb is None else orb[0], feat))
        elif kind == "points":
            replies.put(("points", cloud.points(msg[1])))
        elif kind == "state":
            replies.put(("state", {**{k: getattr(cloud, k) for k in CLOUD_ARRAYS}, "frames": cloud.frames,
                                   "raw_points": cloud.raw_points}, merged))
        elif kind == "quit":
            return


class MergeProcess:
    """The parent's side: start the process, send jobs, read replies (`on_merged(reply)` gets each
    "merged" one, in order). Waits fail with RuntimeError if the process died."""

    def __init__(self, point_mm, stride, clean, on_merged):
        from . import calib
        ctx = multiprocessing.get_context("spawn")
        self.jobs, self.replies = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=run, daemon=True, args=(self.jobs, self.replies, point_mm, stride, clean,
                                                                calib.CAMERA.spec, calib.CAMERA.mode))
        # The live worker is itself a daemonic process, and multiprocessing refuses those children (the merge
        # process then never started live: the scan merged on the thread). Its flag is lifted for this start
        # only; the merge process ends by itself when its parent goes (run: end of the queue or parent check).
        config = multiprocessing.current_process()._config
        was_daemon = config.get("daemon")
        config["daemon"] = False
        try:
            self.proc.start()
        finally:
            config["daemon"] = was_daemon
        self.on_merged, self.sent, self.done = on_merged, 0, 0
        self.points_cache, self.points_asked = None, False  # latest snapshot (for the 3D page), one asked for
        try:  # started for real (imports, the camera) before any merge is sent: else the thread merges
            self.replies.get(timeout=READY_S)
        except queue.Empty:
            self.close()
            raise RuntimeError(f"no word from it in {READY_S} s" if self.proc.exitcode is None
                               else f"it ended at once (exit code {self.proc.exitcode})")

    def submit(self, frame, index, pose, pix, step, orb=None):
        from . import calib
        self.jobs.put(("merge", frame, index, pose, pix, step, orb, calib.CAMERA.mode))
        self.sent += 1

    def poll(self, wait_for=None, timeout=None):
        """Take the replies that are there; with `wait_for` ("merged" count reached, "points", "state")
        keep waiting until it has come and return it."""
        while True:
            try:
                reply = self.replies.get(timeout=1.0 if wait_for else 0)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f"the merge process ended (exit code {self.proc.exitcode})")
                if wait_for is None:
                    return None
                if timeout is not None and (timeout := timeout - 1.0) <= 0:
                    raise RuntimeError(f"no {wait_for} from the merge process")
                continue
            if reply[0] == "merged":
                self.done += 1
                self.on_merged(reply)
                if wait_for == "merged" and self.done >= self.sent:
                    return reply
            elif reply[0] == "points":
                self.points_cache, self.points_asked = reply[1], False
                if wait_for == "points":
                    return reply[1]
            elif reply[0] == "state" and wait_for == "state":
                return reply[1], reply[2]

    def sync(self):
        """Wait until every job sent is merged."""
        if self.done < self.sent:
            self.poll("merged")

    def points(self, min_seen):
        """(xyz, rgb, seen) of the cloud after every job sent."""
        self.sync()
        if self.points_asked:  # an earlier snapshot is on its way: it is older than this one must be
            self.poll("points")
        self.jobs.put(("points", min_seen))
        return self.poll("points")

    def ask_points(self, min_seen):
        """Ask for a snapshot without waiting (it lands in points_cache at a later poll)."""
        if not self.points_asked:
            self.jobs.put(("points", min_seen))
            self.points_asked = True

    def state(self):
        """(cloud arrays, merged_data) after every job sent; the process then ends."""
        self.sync()
        self.jobs.put(("state",))
        out = self.poll("state")
        self.close()
        return out

    def close(self):
        if self.proc.is_alive():
            self.jobs.put(("quit",))
            self.proc.join(10)
            if self.proc.is_alive():
                self.proc.kill()
                time.sleep(0.1)
