"""Closing the start.bat window must not leave ComfyUI or a worker running (an orphaned ComfyUI was
found holding RAM and VRAM).  A parent that bound itself is killed hard; its child must go with it."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PARENT = f"""
import subprocess, sys, time
sys.path.insert(0, {str(ROOT)!r})
from lorastudio.procjob import bind_children_to_this_process
assert bind_children_to_this_process()
c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(c.pid, flush=True)
time.sleep(120)
"""


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object")
def test_children_end_when_the_server_is_killed():
    parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    assert _alive(child)
    parent.kill()                       # like closing the console window: no cleanup code runs
    parent.wait()
    for _ in range(50):
        if not _alive(child):
            break
        time.sleep(0.1)
    assert not _alive(child)
