"""What the toolkit uses of the machine, read from /proc (Linux; stdlib only, so the viewer's plain
Python can use it): CPU and memory per process, a process's children, the GPU per process (DRM
fdinfo: memory and busy time, as i915, xe, amdgpu and others report them), free memory.
"""
import os
import time
from collections import defaultdict
from pathlib import Path


def parse_fdinfo(texts):
    """GPU memory (MiB per region) and busy time (s per engine) from DRM fdinfo texts, one entry per
    client (fds sharing a client are counted once). None when no fd is a DRM client."""
    clients = {}
    for text in texts:
        fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if "drm-client-id" in fields:
            clients[(fields.get("drm-pdev"), fields["drm-client-id"])] = fields
    if not clients:
        return None
    units = {"": 1 / 2 ** 20, "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}
    memory, busy = defaultdict(float), defaultdict(float)
    for fields in clients.values():
        for key, value in fields.items():
            number, _, unit = value.partition(" ")
            if key.startswith(("drm-total-", "drm-memory-")) and number.isdigit():
                memory[key.split("-", 2)[2]] += int(number) * units.get(unit, 0)
            elif key.startswith("drm-engine-") and not key.startswith("drm-engine-capacity-") and unit == "ns":
                busy[key[len("drm-engine-"):]] += int(number) / 1e9
    return {"driver": next(iter(clients.values())).get("drm-driver"),
            "memory_mib": {k: round(v, 1) for k, v in memory.items()},
            "busy_s": {k: round(v, 3) for k, v in busy.items()}}


def gpu_usage(pid="self"):
    """parse_fdinfo of a process's open files (None: it uses no GPU, or it is gone)."""
    texts, folder = [], Path(f"/proc/{pid}/fdinfo")
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    for fd in names:
        try:
            texts.append((folder / fd).read_text())
        except OSError:
            pass
    return parse_fdinfo(texts)


def rss_mb(pid="self"):
    """Resident memory (MB) of a process, None if it is gone."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def cpu_seconds(pid="self"):
    """CPU time (user + system, s) a process has used, None if it is gone."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def children(pid):
    """[(pid, command line)] of a process's children."""
    out = []
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            kids = (task / "children").read_text().split()
        except OSError:
            continue
        for kid in kids:
            try:
                cmd = Path(f"/proc/{kid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                continue
            out.append((int(kid), cmd))
    return out


def memory_mb():
    """(available, total) memory of the machine (MB)."""
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            info[key] = int(value.split()[0]) / 1024
    except (OSError, ValueError, IndexError):
        return None, None
    return info.get("MemAvailable"), info.get("MemTotal")


class Rates:
    """CPU use (% of one core) and GPU busy (% of the time) per process, from the change since the
    previous call for the same process."""

    def __init__(self):
        self.last = {}

    def _rate(self, key, value):
        now = time.monotonic()
        before = self.last.get(key)
        self.last[key] = (now, value)
        if before is None or before[1] is None or value is None or now <= before[0]:
            return None
        return 100 * (value - before[1]) / (now - before[0])

    def cpu(self, pid="self"):
        return self._rate(("cpu", pid), cpu_seconds(pid))

    def gpu_busy(self, pid, usage):
        """`usage` from gpu_usage(pid): its busiest engine's share of the time since the last call."""
        busy = max(usage["busy_s"].values()) if usage and usage["busy_s"] else None
        return self._rate(("gpu", pid), busy)
