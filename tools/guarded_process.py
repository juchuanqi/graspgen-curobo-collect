"""Terminate a stage/worker if its supervisor dies, including SIGKILL."""
import ctypes
import os
import signal
import subprocess
import sys


def main():
    expected_parent = int(sys.argv[1])
    def parent_died(*unused):
        # This wrapper owns a fresh process group. Include descendants that may
        # have inherited the run lock (e.g. a compiler or multiprocessing helper).
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGUSR1, parent_died)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGUSR1, 0, 0, 0) != 0:  # Linux PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'Cannot bind child lifetime to supervisor')
    if os.getppid() != expected_parent:
        raise SystemExit('Supervisor exited before child startup')
    child = subprocess.Popen(sys.argv[2:], close_fds=False)
    result = child.wait()
    raise SystemExit(result if result >= 0 else 128 - result)


if __name__ == '__main__':
    main()
