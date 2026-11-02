"""Run an external tool in an owned process group and reap leftover workers."""
import math
import os
import signal
import subprocess
import time


def _signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def _cleanup(process, grace):
    # The parent may already have exited while its browser/model workers live.
    if os.name == 'posix':
        if _signal_group(process.pid, signal.SIGTERM):
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                process.poll()  # reap the direct child before testing the group
                if not _signal_group(process.pid, 0):
                    break
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
            _signal_group(process.pid, signal.SIGKILL)
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait()


def run_isolated(argv, *, timeout, cleanup_grace=5, **kwargs):
    """Like a checked subprocess.run, with POSIX process-group cleanup.

    Descendants must stay in the tool's session (do not daemonize). On Windows,
    only the directly launched process can be terminated by this helper.
    """
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('External timeout must be a finite positive number.')
    if not math.isfinite(cleanup_grace) or cleanup_grace < 0:
        raise ValueError('Cleanup grace must be finite and nonnegative.')
    process = subprocess.Popen(argv, start_new_session=(os.name == 'posix'), **kwargs)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, argv)
        return subprocess.CompletedProcess(argv, code)
    finally:
        _cleanup(process, cleanup_grace)
