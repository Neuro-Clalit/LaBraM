# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Dropping channels from a dataset bundle (data.drop_channels), e.g. T1/T2 so
# recordings that lack them can join a 21-channel cohort.
# ---------------------------------------------------------
"""``data.drop_channels``: serve every sample without the named channels.

The npy loaders read only the kept columns of each recording, and the bundle's
``ch_names`` shrink to match, so the model gets the right channel positions
(``get_channel_indices``) and the LaBraM++ common average is taken over the
kept channels only. The files on disk are unchanged.
"""
import logging
from typing import Sequence

import numpy as np

from labram.data.window_selection import _leaves

logger = logging.getLogger(__name__)


def drop_channels(bundle, names: Sequence[str]):
    """Drop ``names`` (10-20 names, any case) from every split of ``bundle``."""
    if not names:
        return bundle
    names = {n.upper() for n in names}
    unknown = sorted(names - {c.upper() for c in bundle.ch_names})
    if unknown:
        raise ValueError(f"data.drop_channels: {unknown} not among the dataset's channels "
                         f"{bundle.ch_names}")
    keep = [i for i, c in enumerate(bundle.ch_names) if c.upper() not in names]
    for split in (bundle.train, bundle.val, bundle.test):
        for leaf in _leaves(split):
            if not hasattr(leaf, "keep_channels"):
                raise ValueError("data.drop_channels needs the npy data format "
                                 "(data.data_format=npy)")
            leaf.keep_channels = np.asarray(keep)
    logger.info("Dropping channels %s: %d -> %d channels", sorted(names),
                len(bundle.ch_names), len(keep))
    bundle.ch_names = [bundle.ch_names[i] for i in keep]
    return bundle
