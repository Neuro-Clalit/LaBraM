"""Tests for utils.channels.{standard_1020, get_channel_indices}."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from utils.channels import get_channel_indices, standard_1020


class TestGetChannelIndices:
    def test_prepends_cls_token_index(self):
        # First entry must always be 0 (the CLS slot).
        assert get_channel_indices(['FP1'])[0] == 0

    def test_known_channel_resolves(self):
        # FP1 is at standard_1020 index 0, so the returned index is 0+1=1.
        idx = get_channel_indices(['FP1'])
        assert idx == [0, 1]

    def test_multiple_channels_preserve_input_order(self):
        idx = get_channel_indices(['FP2', 'FP1', 'F3'])
        # standard_1020 = ['FP1','FPZ','FP2', 'AF9',..., 'F3', ...]
        # FP2 -> idx 2 -> +1 = 3
        # FP1 -> idx 0 -> +1 = 1
        # F3  -> idx 17 -> +1 = 18
        assert idx == [0, 3, 1, 18]

    def test_unknown_channel_raises_valueerror(self):
        with pytest.raises(ValueError):
            get_channel_indices(['DOES_NOT_EXIST'])


class TestStandard1020:
    def test_starts_with_frontal(self):
        # First three channels should be the frontal-pole electrodes.
        assert standard_1020[:3] == ['FP1', 'FPZ', 'FP2']

    def test_no_duplicates_in_unipolar_section(self):
        # The first 86 entries are unipolar electrodes; no name should repeat.
        unipolar = standard_1020[:86]
        assert len(unipolar) == len(set(unipolar))

    def test_includes_bipolar_montage(self):
        # The list ends with the standard TUH bipolar montage entries.
        assert "FP1-F7" in standard_1020
        assert "P4-O2" in standard_1020
