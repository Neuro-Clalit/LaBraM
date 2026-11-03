# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Multi-GPU launch: re-run an entry point under torchrun, one process per GPU.
# ---------------------------------------------------------
"""Start a run on several GPUs from a single switch.

Every trainer already runs under PyTorch DDP when launched by ``torchrun``
(one process per GPU, gradients all-reduced every optimizer step). Setting
``distributed.nproc_per_node`` (``-1`` = every visible GPU) makes the entry
point do that itself: when it is not already running under a launcher, it
replaces its own process with

    python -m torch.distributed.run --standalone --nproc_per_node=N <same entry> <same args>

so ``python -m labram.runs.run_finetune --config ... --set distributed.nproc_per_node=-1``
and an explicit ``torchrun --nproc_per_node=N -m labram.runs.run_finetune ...``
are the same run. Inside a multi-instance SageMaker job the node count, rank
and master address come from ``SM_HOSTS`` / ``SM_CURRENT_HOST``.

torchrun parses option-like arguments after the script name as its own when
they are a prefix of one of its options (``--run`` -> ``--run-path``), so the
workers start this module as a trampoline and the real entry point and its
arguments travel in ``$LABRAM_LAUNCH_TARGET`` (JSON), never through torchrun's
parser.
"""

import json
import logging
import os
import sys
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

MASTER_PORT = 29500
_LAUNCHED_ENV = "LABRAM_TORCHRUN_LAUNCHED"
TARGET_ENV = "LABRAM_LAUNCH_TARGET"


def visible_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # pragma: no cover - torch import problems
        return 0


def resolve_nproc(nproc_per_node: int, gpus: Optional[int] = None) -> int:
    """``-1`` -> every visible GPU (1 without CUDA); ``0`` / ``1`` -> 1 process.
    More processes than GPUs is an error on a CUDA machine (CPU runs with
    several gloo processes are allowed, e.g. for tests)."""
    gpus = visible_gpus() if gpus is None else gpus
    if nproc_per_node is None or nproc_per_node == 0:
        return 1
    if nproc_per_node < 0:
        return max(gpus, 1)
    if gpus and nproc_per_node > gpus:
        raise ValueError(f"distributed.nproc_per_node={nproc_per_node} but only {gpus} GPU(s) "
                         f"are visible")
    return nproc_per_node


def under_launcher() -> bool:
    """True inside a torchrun / SLURM / MPI worker (rank variables are set)."""
    return ("LOCAL_RANK" in os.environ or ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
            or os.environ.get(_LAUNCHED_ENV) == "1")


def node_layout(env=None):
    """``(nnodes, node_rank, master_addr)`` of a multi-instance SageMaker job,
    else ``(1, 0, None)``."""
    env = os.environ if env is None else env
    hosts = env.get("SM_HOSTS")
    if not hosts:
        return 1, 0, None
    hosts = sorted(json.loads(hosts))
    if len(hosts) <= 1:
        return 1, 0, None
    return len(hosts), hosts.index(env["SM_CURRENT_HOST"]), hosts[0]


def torchrun_command(nproc: int, env=None) -> List[str]:
    """``python -m torch.distributed.run ... -m labram.utils.launch``: the
    workers run the trampoline below, which starts the target from
    ``$LABRAM_LAUNCH_TARGET``."""
    nnodes, node_rank, master = node_layout(env)
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}"]
    if nnodes > 1:
        cmd += [f"--nnodes={nnodes}", f"--node_rank={node_rank}",
                f"--master_addr={master}", f"--master_port={MASTER_PORT}"]
    else:
        cmd += ["--standalone"]
    return cmd + ["-m", "labram.utils.launch"]


def launch_target(module: Optional[str], script: Optional[str], argv: Sequence[str]) -> str:
    if (module is None) == (script is None):
        raise ValueError("pass exactly one of module / script")
    return json.dumps({"module": module, "script": script, "argv": list(argv)})


def run_target(spec: str) -> None:
    """Run the entry point described by ``spec`` as ``__main__`` with its argv."""
    import runpy
    target = json.loads(spec)
    if target["module"]:
        sys.argv = [target["module"], *target["argv"]]
        runpy.run_module(target["module"], run_name="__main__", alter_sys=True)
    else:
        script = target["script"]
        sys.argv = [script, *target["argv"]]
        sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
        runpy.run_path(script, run_name="__main__")


def relaunch_if_needed(nproc_per_node: int, *, module: Optional[str] = None,
                       script: Optional[str] = None, argv: Optional[Sequence[str]] = None) -> None:
    """Replace this process with a torchrun launch of the same entry point when
    ``nproc_per_node`` asks for more than one process (per node) and we are not
    a launched worker yet. Returns (and the caller runs normally) otherwise."""
    nnodes, _, _ = node_layout()
    if under_launcher():
        return
    nproc = resolve_nproc(nproc_per_node)
    if nproc <= 1 and nnodes <= 1:
        return
    target = launch_target(module, script, sys.argv[1:] if argv is None else argv)
    cmd = torchrun_command(nproc)
    env = dict(os.environ)
    env[_LAUNCHED_ENV] = "1"
    env[TARGET_ENV] = target
    # One intra-op thread per worker unless the user chose otherwise: N
    # processes each spawning all-core OpenMP pools oversubscribe the CPU.
    env.setdefault("OMP_NUM_THREADS", "1")
    print(f"[launch] {nproc} process(es) per node x {nnodes} node(s): {' '.join(cmd)} "
          f"-> {module or script} {' '.join(json.loads(target)['argv'])}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    run_target(os.environ[TARGET_ENV])
