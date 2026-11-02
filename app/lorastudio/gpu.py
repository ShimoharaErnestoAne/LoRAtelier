"""GPU arbitration on a single GPU.

* Normal state: the resident analysis worker may use the GPU.
* Training / evaluation take the GPU exclusively: the analysis dispatcher is paused, the
  worker is told to release its models (VRAM freed), ComfyUI is told to unload.
* Only one exclusive owner at a time; heavy jobs are serialised by the job runner anyway.
"""
from __future__ import annotations

import shutil
import subprocess
import threading


class GPU:
    def __init__(self):
        self._cond = threading.Condition()
        self.owner: str | None = None       # exclusive owner label, e.g. "学習 #3"
        # analysis requests in flight: material analysis, the identity check and i2t can overlap, and the
        # first one to finish must not let training start while another still uses the GPU
        self._analyses = 0
        self._release_hooks = []             # callables run when an exclusive owner takes over
        self._analysis_hooks = []            # callables run when analysis is about to use the GPU

    @property
    def analysis_active(self) -> bool:
        return self._analyses > 0

    def on_exclusive(self, fn):
        self._release_hooks.append(fn)

    def on_analysis(self, fn):
        self._analysis_hooks.append(fn)

    # analysis side ---------------------------------------------------------
    def begin_analysis(self, timeout=None) -> bool:
        for fn in self._analysis_hooks:   # e.g. the ComfyUI kept loaded after image generation
            try:
                fn()
            except Exception:
                pass
        with self._cond:
            if not self._cond.wait_for(lambda: self.owner is None, timeout=timeout):
                return False
            self._analyses += 1
            return True

    def end_analysis(self):
        with self._cond:
            self._analyses = max(0, self._analyses - 1)
            self._cond.notify_all()

    # exclusive side --------------------------------------------------------
    def acquire(self, owner: str, skip_hooks: tuple = (), on_wait=None):
        """on_wait(): called once (outside the lock) when the GPU is ours but analysis requests still in flight
        (material analysis, i2t) must end first, so the caller can say why it has not started yet."""
        with self._cond:
            self._cond.wait_for(lambda: self.owner is None)
            self.owner = owner
            waiting = self.analysis_active
        if waiting and on_wait:
            try:
                on_wait()
            except Exception:
                pass
        with self._cond:   # owner is set: no new analysis starts meanwhile
            self._cond.wait_for(lambda: not self.analysis_active)
        for fn in self._release_hooks:
            if fn.__name__ in skip_hooks:
                continue
            try:
                fn()
            except Exception:
                pass

    def release(self, owner: str):
        with self._cond:
            if self.owner == owner:
                self.owner = None
                self._cond.notify_all()


def query_gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append({"name": parts[0], "vram_total_mb": int(parts[1]), "vram_used_mb": int(parts[2]), "driver": parts[3]})
            except ValueError:
                continue   # e.g. memory shown as "[N/A]": that GPU cannot be planned for
    return gpus


gpu = GPU()


def system_ram_gb() -> float | None:
    """Installed system memory (block swap keeps part of a large DiT there while training)."""
    st = _memory_status()
    return round(st.ullTotalPhys / 2 ** 30, 1) if st else None


def free_ram_gb() -> float | None:
    """System memory available now (None when it cannot be read)."""
    st = _memory_status()
    return round(st.ullAvailPhys / 2 ** 30, 1) if st else None


def _memory_status():
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        st = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st
    except (AttributeError, OSError):   # not Windows
        return None


def shared_gpu_memory_gb() -> float | None:
    """Total 'shared GPU memory' in use (system RAM the Windows driver lends to the GPU).
    On Windows, running out of VRAM usually does NOT raise an out-of-memory error: the driver spills
    into system RAM and training becomes many times slower.  Measured: SDXL at 2048px/batch 16 on a
    16GB card -> 8.1GB shared, 16 s/step, no error."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              r"((Get-Counter '\GPU Process Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Sum).Sum"],
                             capture_output=True, text=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        return round(float(out) / 2**30, 2)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
