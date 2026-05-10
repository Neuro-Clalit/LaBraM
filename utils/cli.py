# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# CLI helpers.
# ---------------------------------------------------------

import argparse

import torch


def bool_flag(s):
    """Parse boolean arguments from the command line."""
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    if s.lower() in TRUTHY_STRINGS:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def get_model(model):
    """Return the inner module if `model` is DataParallel/DDP-wrapped, else `model`."""
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model
