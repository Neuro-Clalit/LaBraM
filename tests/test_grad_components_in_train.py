"""Integration smoke test: train_one_epoch logs per-loss-component gradient
norms (Task 4) when evaluation.log_grad_components is enabled with the
codebook-regularized criterion."""

import torch

from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import EvaluationConfig, TrainerConfig
from labram.losses import CodebookRegularizedCriterion, LossConfig
from labram.models.outputs import PredictorOutput
from labram.train.train_finetune import train_one_epoch
from labram.utils import NativeScalerWithGradNormCount


class _TinyRegModel(torch.nn.Module):
    """Emits a PredictorOutput whose logits and reconstruction terms are all
    connected to trainable parameters (so component grads are non-trivial)."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.head = torch.nn.Linear(200, n_classes)
        self.dec = torch.nn.Linear(200, 200)

    def forward(self, samples, channel_indices=None, classify_only=False, **kw):
        b = samples.shape[0]
        flat = samples.reshape(b, -1)[:, :200]
        logits = self.head(flat)
        if classify_only:
            return PredictorOutput(logits=logits)
        x_patched = samples  # [B, N, A, T]
        recon = self.dec(flat).reshape(b, 1, 1, 200).expand_as(x_patched)
        return PredictorOutput(
            logits=logits,
            recon_magnitude=recon,
            recon_phase=recon,
            quantize_loss=(flat ** 2).mean(),
            x_patched=x_patched,
        )


def _loader(n=4, n_classes=3):
    # samples: [B, N=1, A*T=200]; targets: class ids
    xs = torch.randn(n, 1, 200)
    ys = torch.arange(n) % n_classes
    return [(xs[i:i + 2], ys[i:i + 2]) for i in range(0, n, 2)]


def test_train_one_epoch_logs_component_grad_norms():
    torch.manual_seed(0)
    model = _TinyRegModel()
    criterion = CodebookRegularizedCriterion(torch.nn.CrossEntropyLoss(), LossConfig())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    eval_cfg = EvaluationConfig(log_grad_components=True, log_grad_freq=1,
                                detailed_metrics=False)

    stats = train_one_epoch(
        model, criterion, _loader(), optimizer, torch.device("cpu"),
        epoch=0, loss_scaler=NativeScalerWithGradNormCount(),
        trainer_cfg=TrainerConfig(update_freq=1), optim_cfg=OptimizerConfig(),
        num_training_steps_per_epoch=2, log_writer=None, start_steps=0,
        is_binary=False, nb_classes=3, eval_cfg=eval_cfg,
    )

    # Component grad-norm meters were recorded for each loss term.
    for term in ("classifier", "magnitude", "phase", "quantize"):
        assert f"grad_norm_{term}" in stats
        assert stats[f"grad_norm_{term}"] >= 0.0
