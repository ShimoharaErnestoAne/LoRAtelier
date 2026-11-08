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


RELAUNCHER = """
import subprocess, sys, time
sys.path.insert(0, {root!r})
from lorastudio.procjob import bind_children_to_this_process, start_in_new_window
assert bind_children_to_this_process()
code = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(120)"
p = start_in_new_window([sys.executable, "-c", code, {out!r}], dict(__import__("os").environ))
print(p.pid, flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object")
def test_restart_window_outlives_the_old_server(tmp_path):
    """設定 → 再起動: the new server's window is started outside the job, so it keeps running when the old server
    ends (which ends everything in its job).  The path has a space and Japanese to check the cmd quoting."""
    out = tmp_path / "再起動 テスト" / "pid.txt"
    out.parent.mkdir()
    parent = subprocess.Popen([sys.executable, "-c", RELAUNCHER.format(root=str(ROOT), out=str(out))],
                              stdout=subprocess.PIPE, text=True)
    cmd_pid = int(parent.stdout.readline())
    try:
        for _ in range(100):
            if out.exists() and out.read_text():
                break
            time.sleep(0.1)
        child = int(out.read_text())
        parent.kill()
        parent.wait()
        time.sleep(1.0)
        assert _alive(child)
    finally:
        subprocess.run(["taskkill", "/PID", str(cmd_pid), "/T", "/F"], capture_output=True)
        if parent.poll() is None:
            parent.kill()
