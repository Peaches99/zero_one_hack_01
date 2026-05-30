"""Single-instance guard for training runs.

On a low-RAM Mac, running two MPS training jobs at once exhausts unified memory
and can force a reboot. This guard makes a second concurrent run fail fast with a
clear message instead of silently piling on. Uses a PID lockfile under the OS
temp dir; a stale lock (process gone) is reclaimed automatically.
"""
from __future__ import annotations

import atexit
import os
import sys
import tempfile
from pathlib import Path

LOCK = Path(tempfile.gettempdir()) / "process_lm_train.lock"


def _alive(pid: int) -> bool:
    """True if a process with this PID is currently running.

    POSIX uses the signal-0 existence check. On Windows, ``os.kill(pid, 0)``
    would *terminate* the target via TerminateProcess (Python maps any sig
    other than CTRL_*_EVENT to a kill), so we query liveness through the Win32
    API instead and never signal anything.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False  # no such process (or access denied) -> treat as gone
        try:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # signal 0 = existence check (POSIX); does not signal
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else
    return True


def acquire(force: bool = False) -> None:
    """Take the training lock or exit(1) if another live run holds it."""
    if LOCK.exists():
        try:
            holder = int(LOCK.read_text().strip())
        except (ValueError, OSError):
            holder = None
        if holder and holder != os.getpid() and _alive(holder) and not force:
            raise SystemExit(
                f"[runguard] another process_lm training run is active (PID {holder}).\n"
                f"  Running two MPS jobs on this Mac can exhaust RAM and force a reboot.\n"
                f"  Wait for it to finish, or pass --force / delete {LOCK} if it is stale."
            )
    LOCK.write_text(str(os.getpid()))
    atexit.register(release)


def release() -> None:
    try:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()
    except OSError:
        pass
