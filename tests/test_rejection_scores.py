"""The rejection scores (spread, total, learned) of scripts/rejection_scores.py never read the scored cohort's ages."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _windows(path, recordings, rng, age=None):
    rows = []
    for r in recordings:
        n = int(rng.integers(8, 14))
        a = rng.random(n) + 0.1
        rows.append(pd.DataFrame({"recording": r, "slot": range(n), "minute": 0.0, "alpha": a / a.sum(),
                                  "window_pred": 40 + 10 * rng.standard_normal() + 3 * rng.standard_normal(n),
                                  "age": np.nan if age is None else age[r]}))
    pd.concat(rows).to_parquet(path)


def test_scores_and_kept_set_do_not_depend_on_the_scored_ages(tmp_path):
    import rejection_scores as rs
    rng = np.random.default_rng(0)
    members = ["M_f0", "M_f1", "M_f2"]
    test_recs = [f"t{i}" for i in range(60)]
    for m in members:                                   # test windows carry no usable age at all
        _windows(tmp_path / f"{m}_test.parquet", test_recs, rng)
    val_recs = [f"v{i}" for i in range(200)]            # validation: other recordings, with ages
    val_age = {r: float(rng.uniform(20, 80)) for r in val_recs}
    _windows(tmp_path / "val_f0.parquet", val_recs, rng, val_age)
    val_model, n = rs.fit_val_model([str(tmp_path / "val_f0.parquet")])
    assert n == 200

    def scores(ages):
        meta = pd.DataFrame({"subject": [f"s{i}" for i in range(60)], "age": ages, "healthy": True,
                             "pathological": False}, index=pd.Index(test_recs, name="stem"))
        return rs.add_cohort_scores(rs.cohort_units(meta, str(tmp_path), members, val_model))

    a = scores(rng.uniform(20, 80, 60))
    b = scores(rng.uniform(20, 80, 60))                 # different "true" ages
    assert not np.allclose(a["age"], b["age"]) and not np.allclose(a["abs_err"], b["abs_err"])
    for col in ("spread", "total", "fold_sd", "learned_val", "learned_val+fold"):
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy())
        assert ((a[col] <= a[col].quantile(0.8)) == (b[col] <= b[col].quantile(0.8))).all()
