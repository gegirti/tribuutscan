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
PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "ushort": "<u2", "int": "<i4", "uint": "<u4",
             "char": "i1", "short": "<i2", "float32": "<f4", "float64": "<f8", "uint8": "u1", "int8": "i1",
             "uint16": "<u2", "int16": "<i2", "int32": "<i4", "uint32": "<u4"}


def read_ply(path, faces=False):
    """Vertices of a binary little-endian PLY as a numpy record array; with `faces`, also its
    triangles ((M, 3) int, None when it has none)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: only binary little-endian PLY is supported")
    elements = []  # [name, count, [property parts]]
    for line in header:
        parts = line.split()
        if parts[:1] == ["element"]:
            elements.append([parts[1], int(parts[2]), []])
        elif parts[:1] == ["property"] and elements:
            elements[-1][2].append(parts[1:])
    vertex, triangles, offset = None, None, end
    for name, count, props in elements:
        if name == "vertex":
            vertex = np.frombuffer(data, dtype=[(p[1], PLY_TYPES[p[0]]) for p in props], count=count, offset=offset)
            offset += vertex.nbytes
        elif name == "face" and props and props[0][0] == "list":
            rec = np.frombuffer(data, dtype=[("n", PLY_TYPES[props[0][1]]), ("i", PLY_TYPES[props[0][2]], 3)],
                                count=count, offset=offset)
            if len(rec) and (rec["n"] != 3).any():
                raise ValueError(f"{path}: only triangle faces are supported")
            triangles, offset = rec["i"].astype(np.int64), offset + rec.nbytes
        else:
            break  # an element this reader does not know: what follows cannot be located
    return (vertex, triangles) if faces else vertex


def pack_cloud(rec):
    """Binary payload for the page from a PLY record array (see pack_arrays)."""
    names = rec.dtype.names
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1)
    rgb = np.stack([rec["red"], rec["green"], rec["blue"]], 1) if "red" in names else None
    return pack_arrays(xyz, rgb, rec["seen"] if "seen" in names else None)


def pack_mesh(path):
    """Binary mesh payload for the page from a mesh PLY: u32 vertices, u32 triangles, f32 xyz, u8 rgb
    (padded to 4), u32 indices."""
    rec, triangles = read_ply(path, faces=True)
    n, m = len(rec), 0 if triangles is None else len(triangles)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1)
    rgb = (np.stack([rec["red"], rec["green"], rec["blue"]], 1) if "red" in rec.dtype.names
           else np.full((n, 3), 200, np.uint8)).astype(np.uint8)
    return (struct.pack("<II", n, m) + np.asarray(xyz, "<f4").tobytes() + rgb.tobytes() + b"\0" * (-3 * n % 4)
            + (np.asarray(triangles, "<u4").tobytes() if m else b""))


def mesh_version(path):
    """A token that changes when the mesh file does (None: no mesh)."""
    return str(path.stat().st_mtime_ns) if path is not None and path.exists() else None


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

    def process(self, action):
        """A Process-menu action on the saved session (needs keyframes.npz and merged.npz)."""
        from . import cloud as cl
        from . import optimise
        if not (self.ply.parent / "merged.npz").exists():
            return "this session has no saved frames to process (scanned before sessions kept them)"
        scanner = cl.Scanner.restore(self.ply.parent, log=lambda m: None)
        return optimise.process(scanner, self.ply.parent, action, log=lambda m: None)

    def mesh_path(self):
        return self.ply.parent / "mesh.ply"

    def ply_bytes(self):
        return self.ply.read_bytes()

    def download_name(self):
        return self.ply.parent.name if self.ply.name == "cloud.ply" else self.ply.stem

    def current(self):
        version = self.ply.stat().st_mtime_ns
        with self.lock:
            if version != self.version:
                rec = read_ply(self.ply)
                self.payload = pack_cloud(rec)
                session = json.loads(self.session.read_text()) if self.session.exists() else {}
                self.info = {"name": self.ply.parent.name, "points": len(rec), "session": session, "phase": "done"}
                self.version = version
            return self.version, self.payload, dict(self.info, mesh=mesh_version(self.mesh_path()))


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
        if path == "/cloud.ply":  # the live scan's page too (it used to give 404 there)
            body = self.source.ply_bytes() if hasattr(self.source, "ply_bytes") else None
            if body is not None:
                return self.reply(body, "application/octet-stream", f"{self.source.download_name()}.ply")
        mesh = self.source.mesh_path() if hasattr(self.source, "mesh_path") else None
        if path in ("/mesh.ply", "/mesh.stl") and mesh is not None and mesh.with_suffix(path[-4:]).exists():
            return self.reply(mesh.with_suffix(path[-4:]).read_bytes(), "application/octet-stream",
                              f"{self.source.download_name()}-mesh{path[-4:]}")
        if path == "/mesh.bin" and mesh is not None and mesh.exists():
            return self.reply(pack_mesh(mesh), "application/octet-stream")
        self.send_error(404)

    def do_POST(self):
        """/process/<action>: optimise, small/medium/large (merge again) or mesh, on a finished scan."""
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) != 2 or parts[0] != "process" or not hasattr(self.source, "process"):
            return self.send_error(404)
        try:
            message = self.source.process(parts[1])
        except Exception as e:  # report to the page instead of dropping the connection
            message = f"failed: {e}"
        self.reply(json.dumps({"message": message}).encode(), "application/json")

    def reply(self, body, kind, filename=None):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        if filename:  # a download named after the scan, not cloud.ply every time
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
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
