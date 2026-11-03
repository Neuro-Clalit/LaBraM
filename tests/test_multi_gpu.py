"""Multi-GPU (DDP) support: the torchrun relaunch switch, the global batch
split, the DDP wrapper, no_sync on gradient-accumulation micro-steps, and an
end-to-end 2-process fine-tune on CPU (gloo) whose sharded evaluation matches
a single-process evaluation of the same checkpoint."""
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import DistributedConfig, TrainerConfig
from labram.runs.common import resolve_global_batch, wrap_distributed
from labram.utils import launch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ launch

def test_resolve_nproc():
    assert launch.resolve_nproc(1, gpus=4) == 1 and launch.resolve_nproc(0, gpus=4) == 1
    assert launch.resolve_nproc(-1, gpus=4) == 4 and launch.resolve_nproc(-1, gpus=0) == 1
    assert launch.resolve_nproc(2, gpus=0) == 2                     # CPU / gloo workers
    with pytest.raises(ValueError, match="only 2 GPU"):
        launch.resolve_nproc(4, gpus=2)


def test_torchrun_command_single_and_multi_node():
    cmd = launch.torchrun_command(4, env={})
    assert cmd[1:] == ["-m", "torch.distributed.run", "--nproc_per_node=4", "--standalone",
                       "-m", "labram.utils.launch"]
    env = {"SM_HOSTS": json.dumps(["algo-2", "algo-1"]), "SM_CURRENT_HOST": "algo-2"}
    cmd = launch.torchrun_command(8, env=env)
    assert "--nnodes=2" in cmd and "--node_rank=1" in cmd and "--master_addr=algo-1" in cmd
    assert "--standalone" not in cmd


def test_relaunch_execs_torchrun_with_target_in_env(monkeypatch):
    for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE", launch._LAUNCHED_ENV, "SM_HOSTS"):
        monkeypatch.delenv(k, raising=False)
    calls = []
    monkeypatch.setattr(launch.os, "execvpe", lambda f, cmd, env: calls.append((cmd, env)))
    launch.relaunch_if_needed(1, module="labram.runs.run_finetune", argv=["--x"])
    assert calls == []                                         # one process: run in place
    monkeypatch.setattr(launch, "visible_gpus", lambda: 2)
    launch.relaunch_if_needed(-1, module="labram.runs.run_finetune",
                              argv=["--config", "c.json", "--run", "a=b"])
    (cmd, env), = calls
    assert "--nproc_per_node=2" in cmd and cmd[-2:] == ["-m", "labram.utils.launch"]
    target = json.loads(env[launch.TARGET_ENV])
    # torchrun never sees the entry's own options (--run would match --run-path)
    assert target == {"module": "labram.runs.run_finetune", "script": None,
                      "argv": ["--config", "c.json", "--run", "a=b"]}
    assert env[launch._LAUNCHED_ENV] == "1" and env["OMP_NUM_THREADS"]
    monkeypatch.setenv("LOCAL_RANK", "0")                       # already a worker
    launch.relaunch_if_needed(-1, module="labram.runs.run_finetune", argv=[])
    assert len(calls) == 1


def test_run_target_runs_script_as_main(tmp_path, monkeypatch):
    script = tmp_path / "entry.py"
    out = tmp_path / "argv.json"
    script.write_text(f"import json, sys\njson.dump(sys.argv[1:], open({str(out)!r}, 'w'))\n")
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    launch.run_target(launch.launch_target(None, str(script), ["--run", "x=y"]))
    assert json.loads(out.read_text()) == ["--run", "x=y"]


# ------------------------------------------------------------------ batch / DDP

@pytest.mark.parametrize("glob,world,cap,expected", [
    (128, 1, 128, (128, 1)), (128, 4, 128, (32, 1)), (128, 1, 64, (64, 2)),
    (512, 4, 64, (64, 2)), (96, 8, 128, (12, 1)),
])
def test_resolve_global_batch(glob, world, cap, expected):
    per_gpu, freq = resolve_global_batch(glob, world, cap)
    assert (per_gpu, freq) == expected and per_gpu * freq * world == glob and per_gpu <= cap


def test_resolve_global_batch_rejects_uneven_split():
    with pytest.raises(ValueError, match="multiple of"):
        resolve_global_batch(100, 3, 128)


@pytest.fixture
def gloo_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_wrap_distributed_cpu(gloo_group):
    model = torch.nn.Linear(4, 1)
    cfg = DistributedConfig(distributed=True, find_unused_parameters=False)
    wrapped, inner = wrap_distributed(cfg, model)
    assert isinstance(wrapped, torch.nn.parallel.DistributedDataParallel) and inner is model
    assert wrapped.device_ids is None and wrapped.find_unused_parameters is False
    assert wrapped.gradient_as_bucket_view
    assert wrap_distributed(DistributedConfig(distributed=False), model) == (model, model)


class _TinyAge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, x, channel_indices=None, **_):
        return self.lin(x.mean(dim=(1, 2, 3)).unsqueeze(-1))


def test_accumulation_micro_steps_skip_gradient_sync(gloo_group, monkeypatch):
    from labram.train import train_finetune as tf
    from labram.utils.training import NativeScalerWithGradNormCount
    model = torch.nn.parallel.DistributedDataParallel(_TinyAge())
    entered = []
    real = model.no_sync

    def spy():
        entered.append(len(entered))
        return real()
    monkeypatch.setattr(model, "no_sync", spy)
    data = [(torch.randn(2, 2, 400), torch.randn(2)) for _ in range(6)]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    tf.train_one_epoch(model, torch.nn.MSELoss(), data, opt, torch.device("cpu"), 0,
                       NativeScalerWithGradNormCount(), trainer_cfg=TrainerConfig(update_freq=3),
                       optim_cfg=OptimizerConfig(), num_training_steps_per_epoch=2,
                       start_steps=0, task=tf.TASK_REGRESSION, is_binary=False)
    # micro-steps 0,1 and 3,4 accumulate locally; 2 and 5 all-reduce and step
    assert len(entered) == 4


# ------------------------------------------------------------------ end to end

def _run(args, cwd=REPO, timeout=600):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
    for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE", launch._LAUNCHED_ENV):
        env.pop(k, None)
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=timeout)


def test_two_process_finetune_and_sharded_eval_match_single_process(tmp_path):
    from test_dataset_groups import _mixed_corpus
    root = _mixed_corpus(tmp_path, windows=8)

    def finetune(nproc, out, *extra):
        return _run(["-m", "labram.runs.run_finetune",
                     "--config", "labram/configs/defaults/finetune_tuab_age.json", "--set",
                     f"data.data_path={root}", "data.data_format=npy", "data.trim_start_sec=0.0",
                     "data.trim_end_sec=0.0", "data.eval_minutes=0.5", "data.num_workers=0",
                     "data.stage_dir=", "trainer.epochs=1", "optimizer.warmup_epochs=0",
                     "evaluation.train_eval_recordings=2", "distributed.device=cpu",
                     "clearml.enabled=false", "output.append_timestamp=false",
                     f"output.output_dir={out}", f"output.log_dir={out}",
                     "logging.log_model_graph=false", f"distributed.nproc_per_node={nproc}",
                     *extra])

    train = finetune(2, tmp_path / "ddp", "finetune_checkpoint.finetune=",
                     "trainer.global_batch_size=8", "trainer.batch_size=2")
    assert train.returncode == 0, train.stdout[-3000:] + train.stderr[-3000:]
    log = train.stdout + train.stderr
    assert "backend gloo" in log and "per-GPU batch 2 x update_freq 2" in log
    assert "TrainEval EEG mae by dataset/cohort" in log
    ckpt = tmp_path / "ddp" / "checkpoint-best.pth"
    assert ckpt.exists()

    scores = {}
    for n in (1, 2):
        run = finetune(n, tmp_path / f"eval{n}", "trainer.eval=true",
                       f"finetune_checkpoint.finetune={ckpt}")
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        scores[n] = [ln.split("INFO] ", 1)[1] for ln in (run.stdout + run.stderr).splitlines()
                     if "mae:" in ln and "±" in ln]
    assert scores[1] and scores[1] == scores[2]
