"""Browser 3D view of a merged cloud. Python stdlib server + a three.js page (tribuutscan/web/).

    python3 -m tribuutscan.view3d captures/<stamp>-scan     # or a .ply file
    python3 -m tribuutscan.view3d <path> --port 8765 --no-browser

Serves http://localhost:8765 on this machine only. The page polls /info.json and reloads the cloud
when it changes, so the same page works for a finished scan and (later) a live one.
"""
import json
import struct
import sys
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

WEB = Path(__file__).resolve().parent / "web"
PORT = 8765
FILES = {"/": ("index.html", "text/html"), "/three.module.min.js": ("three.module.min.js", "text/javascript"),
         "/OrbitControls.js": ("OrbitControls.js", "text/javascript")}
PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "ushort": "<u2", "int": "<i4", "uint": "<u4"}


def read_ply(path):
    """Vertices of a binary little-endian PLY as a numpy record array."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: only binary little-endian PLY is supported")
    count, fields, in_vertex = 0, [], False
    for line in header:
        parts = line.split()
        if parts[:1] == ["element"]:
            in_vertex = parts[1] == "vertex"
            count = int(parts[2]) if in_vertex else count
        elif parts[:1] == ["property"] and in_vertex:
            fields.append((parts[2], PLY_TYPES[parts[1]]))
    return np.frombuffer(data, dtype=fields, count=count, offset=end)


def pack_cloud(rec):
    """Binary payload for the page from a PLY record array (see pack_arrays)."""
    names = rec.dtype.names
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1)
    rgb = np.stack([rec["red"], rec["green"], rec["blue"]], 1) if "red" in names else None
    return pack_arrays(xyz, rgb, rec["seen"] if "seen" in names else None)


def pack_arrays(xyz, rgb=None, seen=None):
    """Binary payload for the page: u32 count, u32 has_seen, f32 xyz, u8 rgb (padded to 4), u16 seen."""
    n = len(xyz)
    rgb = np.full((n, 3), 200, np.uint8) if rgb is None else np.asarray(rgb, np.uint8)
    rgb_bytes = rgb.tobytes() + b"\0" * (-3 * n % 4)
    seen_bytes = b"" if seen is None else np.minimum(seen, 65535).astype("<u2").tobytes()
    return (struct.pack("<II", n, int(seen is not None)) + np.asarray(xyz, "<f4").tobytes()
            + rgb_bytes + seen_bytes)


class FileSource:
    """A cloud.ply on disk (and its session.json), reloaded when the file changes."""

    def __init__(self, path):
        path = Path(path)
        self.ply = path / "cloud.ply" if path.is_dir() else path
        self.session = self.ply.parent / "session.json"
        self.version, self.payload, self.info = None, b"", {}
        self.lock = threading.Lock()

    def current(self):
        version = self.ply.stat().st_mtime_ns
        with self.lock:
            if version != self.version:
                rec = read_ply(self.ply)
                self.payload = pack_cloud(rec)
                session = json.loads(self.session.read_text()) if self.session.exists() else {}
                self.info = {"name": self.ply.parent.name, "points": len(rec), "session": session}
                self.version = version
            return self.version, self.payload, self.info


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, source, **kwargs):
        self.source = source
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in FILES:
            name, kind = FILES[path]
            return self.reply((WEB / name).read_bytes(), kind)
        version, payload, info = self.source.current()
        if path == "/info.json":
            return self.reply(json.dumps({"version": str(version), **info}).encode(), "application/json")
        if path == "/cloud.bin":
            return self.reply(payload, "application/octet-stream")
        if path == "/cloud.ply" and isinstance(self.source, FileSource):
            return self.reply(self.source.ply.read_bytes(), "application/octet-stream")
        self.send_error(404)

    def reply(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the console for the viewer's own log


def serve(source, port=PORT):
    """Start the server in a background thread on 127.0.0.1; returns it (call .shutdown() to stop)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, source=source))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else PORT
    if "--port" in sys.argv:
        args.remove(str(port))
    source = FileSource(args[0])
    _, _, info = source.current()
    serve(source, port)
    url = f"http://localhost:{port}/"
    print(f"{info['name']}: {info['points']} points -> {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
