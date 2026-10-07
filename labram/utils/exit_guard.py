# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Guarantee that a finished run's process exits.
# ---------------------------------------------------------
import os
import sys
import threading

DEFAULT_GRACE_SEC = 180


def run_and_exit(fn, *args, grace_sec: float = DEFAULT_GRACE_SEC, **kwargs):
    """Run an entry point's ``main`` and make sure the process ends afterwards.

    Third-party exit handlers (ClearML waiting for its reporter subprocess and
    pending uploads) have kept finished runs alive for hours, holding the GPU
    and keeping SageMaker jobs billing. Once ``fn`` returns (or raises), a
    daemon timer gives normal interpreter shutdown ``grace_sec`` seconds and
    then forces the exit with the run's status: 0 on success, 1 on an error.
    A clean shutdown that finishes sooner is unaffected.
    """
    status = 1
    try:
        result = fn(*args, **kwargs)
        status = 0
        return result
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        raise
    finally:
        _arm_exit_timer(status, grace_sec)


def _arm_exit_timer(status: int, grace_sec: float) -> None:
    def force_exit():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            print(f"[exit_guard] process still alive {grace_sec:.0f}s after the run finished; "
                  f"forcing exit({status})", file=sys.stderr, flush=True)
        finally:
            os._exit(status)

    timer = threading.Timer(grace_sec, force_exit)
    timer.daemon = True
    timer.start()
