"""Live viewer: colour, IR and depth side by side. Run with `python -m ferret.viewer`.

Press `s` or click Save to write the current frame set to captures/<timestamp>/.
"""
import threading
import time
import tkinter as tk

import numpy as np

from . import frames as fr
from . import v4l2
from .depth import decode_y10
from .image import depth_to_rgb, ppm, yuyv_to_rgb

PANE_W = 480


def fit(rgb):
    step = max(1, rgb.shape[1] // PANE_W)
    return rgb[::step, ::step]


class Stream(threading.Thread):
    """Reads one node in the background; keeps the latest raw frame and its rendered preview."""

    def __init__(self, dev, fourcc, w, h, render):
        super().__init__(daemon=True)
        self.cap = v4l2.Capture(dev, fourcc, w, h)
        self.render = render
        self.latest = None
        self.raw = None
        self.fps = 0.0

    def run(self):
        n, t0 = 0, time.time()
        for buf in self.cap.frames():
            if len(buf) < self.cap.sizeimage:  # truncated USB transfer
                continue
            self.raw = buf
            self.latest = self.render(self.cap, buf)
            n += 1
            if time.time() - t0 >= 1:
                self.fps, n, t0 = n / (time.time() - t0), 0, time.time()

    def stop(self):
        self.cap.close()


def render_color(cap, buf):
    return ppm(fit(yuyv_to_rgb(buf, cap.width, cap.height)))


def render_ir(cap, buf):
    g = np.frombuffer(buf, np.uint8, cap.width * cap.height).reshape(cap.height, cap.width)
    g = fit(g[:, ::-1])  # IR is mirrored relative to colour/depth
    return ppm(np.repeat(g[..., None], 3, axis=2))


def render_depth(cap, buf):
    return ppm(fit(depth_to_rgb(decode_y10(buf, cap.width, cap.height, cap.bytesperline))))


def main():
    nodes = v4l2.discover()
    streams = {
        "Colour": Stream(nodes["color"], "YUYV", fr.W, fr.H, render_color),
        "IR (flipped)": Stream(nodes["ir"], "GREY", fr.W, fr.H, render_ir),
        "Depth (raw, near=red)": Stream(nodes["depth"], "Y10", fr.W, fr.H, render_depth),
    }
    color, ir, depth = streams.values()

    root = tk.Tk()
    root.title("Ferret Pro")
    labels = {}
    for col, name in enumerate(streams):
        tk.Label(root, text=name).grid(row=0, column=col)
        labels[name] = tk.Label(root)
        labels[name].grid(row=1, column=col)
    status = tk.Label(root, anchor="w")
    status.grid(row=2, column=0, columnspan=2, sticky="we")
    message = {"text": ""}

    def save(_event=None):
        if not (color.raw and ir.raw and depth.raw):
            message["text"] = "nothing to save yet"
            return
        frames = fr.decode(color.raw, ir.raw, depth.raw, depth.cap.bytesperline)
        message["text"] = f"saved {fr.save(frames).name}"

    tk.Button(root, text="Save frame (s)", command=save).grid(row=2, column=2, sticky="e")
    root.bind("s", save)

    for s in streams.values():
        s.start()
    images = {}

    def tick():
        for name, s in streams.items():
            if s.latest is not None:
                images[name] = tk.PhotoImage(data=s.latest, format="PPM")
                labels[name].configure(image=images[name])
        fps = "  ".join(f"{n.split()[0]}: {s.fps:4.1f} fps" for n, s in streams.items())
        status.configure(text=f"{fps}   {message['text']}")
        root.after(33, tick)

    def close():
        for s in streams.values():
            s.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
