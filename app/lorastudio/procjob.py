"""Child processes end with the server, however the server ends.

The server starts ComfyUI, the analysis / i2t workers and the training engine.  Closing the start.bat
window (or killing the server) left them running: an orphaned ComfyUI kept about 1GB of RAM and its
VRAM (found 2026-10-12).  On Windows the server puts itself in a Job Object with KILL_ON_JOB_CLOSE;
every process it starts inherits the job, and when the server's handle closes - which the OS does on
any exit, including a crash or a closed console - Windows terminates them all.  A restart already
treats runs in progress as interrupted, so nothing is meant to outlive the server.  The one exception
is the browser: it is opened outside the job (open_browser), or closing the window would close it too.
"""
from __future__ import annotations

import sys

_job = None   # kept for the life of the process: closing it would kill the children


def bind_children_to_this_process() -> bool:
    global _job
    if sys.platform != "win32" or _job is not None:
        return _job is not None
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                      "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x800   # only processes started with CREATE_BREAKAWAY_FROM_JOB leave the job
    JobObjectExtendedLimitInformation = 9
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(job)
        return False
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        k32.CloseHandle(job)   # e.g. already in a job that forbids nesting: children behave as before
        return False
    _job = job
    return True


def open_browser(url: str) -> None:
    """Open the URL in the default browser outside the job.  A browser that was not running yet is
    started by this call; inside the job it would be killed when the start.bat window closes."""
    import subprocess
    import webbrowser
    if sys.platform == "win32":
        try:
            # cmd leaves the job; the browser it starts is its child, so it is outside the job too
            subprocess.Popen(["cmd", "/c", "start", "", url], creationflags=0x01000000 | 0x08000000,   # BREAKAWAY | NO_WINDOW
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    webbrowser.open(url)


_instance = None   # the open lock file, held for the life of the process


def hold_instance_lock(path) -> bool:
    """True if this process is the only LoRAtelier using this data folder.  A second server would mark
    the first one's training as interrupted and start its own ComfyUI on the same port."""
    global _instance
    if _instance is not None:
        return True
    if sys.platform != "win32":
        return True
    import msvcrt
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    try:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        return False
    _instance = f
    return True


def instance_lock_held(path) -> bool:
    """For setup: is a LoRAtelier server running on this data folder right now?"""
    if sys.platform != "win32":
        return False
    import msvcrt
    import os
    if not os.path.exists(path):
        return False
    try:
        f = open(path, "a+b")
    except OSError:
        return True
    with f:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    return False


_ctrl_handler = None   # kept alive: ctypes callbacks must not be garbage-collected


def exit_with_console() -> None:
    """End the server as soon as the start.bat window goes away.

    Closing the window used to leave the server running (found 2026-10-13): uvicorn turns the close
    event into a graceful shutdown that then waits for worker threads, and the old server kept
    answering with the old code.  Two guards, both ending the process at once (os._exit; the job
    object above then ends ComfyUI, the workers and training, and SQLite's WAL keeps edits safe):
    1. a console close / logoff / shutdown event handler, called before Python's own handler;
    2. a watcher on the nearest ancestor that is not Python (the cmd.exe of start.bat), in case the
       event never arrives (e.g. a terminal tab closed without delivering it).
    """
    global _ctrl_handler
    if sys.platform != "win32" or _ctrl_handler is not None:
        return
    import ctypes
    import os
    import threading
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    def on_ctrl(event):
        if event in (2, 5, 6):   # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
            os._exit(0)
        return False             # Ctrl+C etc.: leave to Python / uvicorn

    _ctrl_handler = HANDLER(on_ctrl)
    k32.SetConsoleCtrlHandler(_ctrl_handler, True)

    class ENTRY(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)   # TH32CS_SNAPPROCESS
    procs = {}
    e = ENTRY()
    e.dwSize = ctypes.sizeof(ENTRY)
    ok = k32.Process32FirstW(snap, ctypes.byref(e))
    while ok:
        procs[e.th32ProcessID] = (e.th32ParentProcessID, e.szExeFile.lower())
        ok = k32.Process32NextW(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    pid = os.getpid()
    for _ in range(8):   # skip the venv launcher (python.exe -> python.exe)
        parent = procs.get(pid, (0, ""))[0]
        if parent not in procs:
            return
        if not procs[parent][1].startswith("python"):
            break
        pid = parent
    else:
        return
    h = k32.OpenProcess(0x00100000, False, parent)   # SYNCHRONIZE
    if not h:
        return

    def watch():
        k32.WaitForSingleObject(wintypes.HANDLE(h), 0xFFFFFFFF)
        os._exit(0)

    threading.Thread(target=watch, daemon=True, name="console-watch").start()
