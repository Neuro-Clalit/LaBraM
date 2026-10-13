"""labram.features.export / dataset: flat training-ready export on synthetic caches."""

import json

import numpy as np
import pandas as pd
import pytest

from labram.features import dataset as fds
from labram.features import export as fex
from labram.features import tuab


def _fake_caches(n=40, seed=0):
    rng = np.random.default_rng(seed)
    names = {"time": ["a", "b"], "psd_bins": ["f1", "f2", "f3"]}
    c = len(tuab.CHANNELS_1020)
    D = {"meta/stem": np.array([f"r{i:03d}" for i in range(n)]),
         "rec/time": rng.standard_normal((n, c, 2)).astype(np.float32),
         "rec/psd_bins": rng.standard_normal((n, c, 3)).astype(np.float32),
         "fbcov": np.tile(np.eye(c, dtype=np.float32), (n, 6, 1, 1))}
    meta = pd.DataFrame({"stem": D["meta/stem"], "subject": [f"s{i // 2}" for i in range(n)],
                         "age": rng.uniform(20, 80, n).round(), "sex": "F",
                         "label": np.where(np.arange(n) % 3 == 0, "abnormal", "normal"),
                         "source_split": ["train"] * (n - 10) + ["test"] * 10})
    meta["abnormal"] = meta.label.eq("abnormal").astype(int)
    return D, names, meta


def test_flat_matrix_order_matches_channel_major_names():
    D, names, _ = _fake_caches()
    X, cols = fex.flat_matrix(D, names, ["time", "psd_bins"])
    c = len(tuab.CHANNELS_1020)
    assert X.shape == (40, c * 5) and X.dtype == np.float32
    assert cols.name.iloc[0] == f"time:a@{tuab.CHANNELS_1020[0]}"
    assert cols.name.iloc[1] == f"time:b@{tuab.CHANNELS_1020[0]}"
    # column of psd_bins:f2 on the 3rd channel
    j = cols.index[cols.name == f"psd_bins:f2@{tuab.CHANNELS_1020[2]}"][0]
    np.testing.assert_array_equal(X[:, j], D["rec/psd_bins"][:, 2, 1])
    assert set(cols.group) == {"time", "psd_bins"}


def test_split_columns_partition_dev_into_folds():
    _, _, meta = _fake_caches()
    sp = fex.tuab_split_columns(meta)
    dev = sp.split.eq("dev")
    assert (sp.fold_abn[dev] >= 0).all() and (sp.fold_abn[~dev] == -1).all()
    age_pool = dev & meta.label.eq("normal")
    assert (sp.fold_age[age_pool] >= 0).all() and (sp.fold_age[~age_pool] == -1).all()
    for f in ("fold_abn", "fold_age"):  # folds are subject-disjoint
        g = meta[sp[f] >= 0].assign(fold=sp[f]).groupby("subject").fold.nunique()
        assert (g == 1).all()


def test_tueg_cohort_split_rules():
    _, _, meta_a = _fake_caches()
    meta_t = pd.DataFrame({"stem": ["r000", "x1", "x2", "x3", "x4", "x5"],
                           "subject": ["s0", "t1", "t2", "t3", "t4", meta_a.subject.iloc[-1]]})
    split = fex.tueg_cohort_split(meta_t, meta_a, frac=0.5)
    assert split[5] in ("excluded", "train")  # a TUAB-test subject is never "test"
    assert (split[1:5] == "test").sum() == 2 and split[0] == "train"


def test_write_load_and_torch_dataset(tmp_path):
    torch = pytest.importorskip("torch")
    D, names, meta = _fake_caches()
    X, cols = fex.flat_matrix(D, names, ["time", "psd_bins"])
    m = pd.concat([meta, fex.tuab_split_columns(meta)], axis=1)
    info = fex.write_flat(str(tmp_path), X, D["fbcov"], m, cols, {"dataset": "fake"})
    assert json.load(open(tmp_path / "info.json"))["n_features"] == X.shape[1] == info["n_features"]

    ff = fds.load_flat(str(tmp_path), groups=["riemann", "psd_bins"])
    assert ff.X.shape == (40, 19 * 3) and ff.fbcov.shape == (40, 6, 19, 19)
    Xd, n_flat = ff.design()
    assert n_flat == 57 and Xd.shape == (40, 57 + 6 * 19 * 19)
    with pytest.raises(KeyError):
        fds.load_flat(str(tmp_path), groups=["nope"])

    tr = fds.HandcraftedFeatureDataset(str(tmp_path), target="age", rows=m.fold_age.between(1, 4))
    mean, std = tr.fit_standardizer()
    item = tr[0]
    assert item["x"].shape == (X.shape[1],) and item["x"].dtype == torch.float32
    assert abs(float(torch.stack([tr[i]["x"] for i in range(len(tr))]).mean())) < 1e-5
    with pytest.raises(ValueError):  # statistics of all columns do not fit a 2-group selection
        fds.HandcraftedFeatureDataset(str(tmp_path), groups=["time"], mean=mean, std=std)
    va = fds.HandcraftedFeatureDataset(str(tmp_path), target="abnormal", rows=np.where(m.split.eq("test"))[0],
                                       groups=["time", "riemann"])
    assert len(va) == 10 and va[0]["x"].shape == (19 * 2,) and va[0]["fbcov"].shape == (6, 19, 19)
    loader = torch.utils.data.DataLoader(tr, batch_size=8)
    b = next(iter(loader))
    assert b["x"].shape == (8, X.shape[1]) and b["y"].shape == (8,)
