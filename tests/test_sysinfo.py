import os
import subprocess
import sys
import time

from tribuutscan import sysinfo


def test_cpu_memory_and_children_of_processes():
    rates = sysinfo.Rates()
    assert rates.cpu() is None  # nothing to compare with yet
    end = time.monotonic() + 0.3
    while time.monotonic() < end:
        pass
    assert rates.cpu() > 30  # % of one core (a busy desktop takes some of it)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        kids = dict(sysinfo.children(os.getpid()))
        assert child.pid in kids and "time.sleep" in kids[child.pid]
        assert sysinfo.rss_mb(child.pid) is not None and sysinfo.cpu_seconds(child.pid) is not None  # (just started)
    finally:
        child.kill()
        child.wait()
    assert sysinfo.rss_mb(child.pid) is None and sysinfo.cpu_seconds(child.pid) is None  # gone
    free, total = sysinfo.memory_mb()
    assert 0 < free <= total


def test_gpu_busy_is_the_busiest_engine_since_the_last_call():
    rates = sysinfo.Rates()
    assert rates.gpu_busy(1, None) is None  # no GPU (yet)
    assert rates.gpu_busy(1, {"busy_s": {"render": 1.0}}) is None  # the reading before had nothing
    time.sleep(0.1)
    busy = rates.gpu_busy(1, {"busy_s": {"render": 1.05, "copy": 0.01}})  # 50 ms of ~100
    assert 20 < busy < 60
