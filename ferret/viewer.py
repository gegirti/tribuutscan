"""Live viewer: colour, IR and depth side by side. Run with `python -m ferret.viewer`."""
import threading
import tkinter as tk

import numpy as np

from . import v4l2
from .depth import decode_y10

DEPTH_MAX = 200  # raw values are ~1..200; scale is unknown, so this is only for display
PANE_W = 480


def ramp(x):
    """Cheap blue->red colour ramp for x in [0, 1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def ppm(rgb):
    h, w = rgb.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + rgb.tobytes()


def yuyv_to_rgb(buf, w, h):
    y = np.frombuffer(buf, np.uint8, w * h * 2).reshape(h, w // 2, 4)
    luma = y[..., [0, 2]].reshape(h, w).astype(np.float32)
    u = np.repeat(y[..., 1], 2, axis=1).astype(np.float32) - 128
    v = np.repeat(y[..., 3], 2, axis=1).astype(np.float32) - 128
    rgb = np.stack([luma + 1.402 * v, luma - 0.344 * u - 0.714 * v, luma + 1.772 * u], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def fit(rgb):
    step = max(1, rgb.shape[1] // PANE_W)
    return rgb[::step, ::step]


class Stream(threading.Thread):
    """Reads one node in the background and keeps the latest rendered frame."""

    def __init__(self, dev, fourcc, w, h, render):
        super().__init__(daemon=True)
        self.cap = v4l2.Capture(dev, fourcc, w, h)
        self.render = render
        self.latest = None
        self.fps = 0.0

    def run(self):
        import time
        n, t0 = 0, time.time()
        for buf in self.cap.frames():
            if len(buf) < self.cap.sizeimage:  # truncated USB transfer
                continue
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
    d = decode_y10(buf, cap.width, cap.height, cap.bytesperline)
    rgb = ramp(np.clip(d / DEPTH_MAX, 0, 1))
    rgb[d == 0] = 0
    return ppm(fit(rgb))


def main():
    nodes = v4l2.discover()
    streams = {
        "Colour": Stream(nodes["color"], "YUYV", 1280, 720, render_color),
        "IR (flipped)": Stream(nodes["ir"], "GREY", 1280, 800, render_ir),
        "Depth (raw, near=red)": Stream(nodes["depth"], "Y10", 1280, 800, render_depth),
    }
    root = tk.Tk()
    root.title("Ferret Pro")
    labels = {}
    for col, name in enumerate(streams):
        tk.Label(root, text=name).grid(row=0, column=col)
        labels[name] = tk.Label(root)
        labels[name].grid(row=1, column=col)
    status = tk.Label(root, anchor="w")
    status.grid(row=2, column=0, columnspan=3, sticky="we")

    for s in streams.values():
        s.start()
    images = {}

    def tick():
        for name, s in streams.items():
            if s.latest is not None:
                images[name] = tk.PhotoImage(data=s.latest, format="PPM")
                labels[name].configure(image=images[name])
        status.configure(text="  ".join(f"{n}: {s.fps:4.1f} fps" for n, s in streams.items()))
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
