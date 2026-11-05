"""Harmonised clinical statistics (labram/eval/clinical_analytics.py)."""
import numpy as np
import pandas as pd

from labram.eval import clinical_analytics as ca


def test_any_of_keeps_unknown_distinct_from_false():
    df = pd.DataFrame({"a": pd.array([True, False, None, False], dtype="boolean"),
                       "b": pd.array([None, None, None, True], dtype="boolean")})
    assert ca._any_of(df, ["a", "b"]).tolist() == [True, False, pd.NA, True]
    assert ca._any_of(df, ["missing"]).isna().all()


def _frame(corpus, n, pos, known):
    x = [True] * pos + [False] * (known - pos) + [None] * (n - known)
    return pd.DataFrame({"corpus": corpus, "stem": [f"{corpus}{i}" for i in range(n)],
                         "subject": range(n), "age": np.linspace(20, 80, n), "sex": "F",
                         "status": ["abnormal" if i % 2 else "normal" for i in range(n)],
                         "px_a": pd.array(x, dtype="boolean"),
                         "ab_t": pd.array([True] * n, dtype="boolean"),
                         "has_health_record": [i < known for i in range(n)],
                         "has_abnormal_detail": [bool(i % 2) for i in range(n)]})


def test_prevalence_counts_known_only_and_suppresses_small_heedb_cells():
    df = pd.concat([_frame("TUH cohort", 100, 5, 50), _frame("HEEDB", 100, 5, 50)])
    p = ca.prevalence(df)
    assert p.loc["TUH cohort", "a"] == 10.0                 # 5 of 50 known
    assert np.isnan(p.loc["HEEDB", "a"])                    # 5 positives < 10: suppressed
    k = ca.known_counts(df).set_index(["corpus", "pathology"])
    assert k.loc[("TUH cohort", "a"), "known"] == 50


def test_abnormal_rate_and_types():
    df = _frame("TUH cohort", 100, 40, 100)
    ar = ca.abnormal_rate_by_pathology(df).iloc[0]
    assert ar["n with"] == 40 and ar["abnormal % with"] == 50.0
    t = ca.abnormal_types(df)
    assert t.loc["TUH cohort", "n"] == 50 and t.loc["TUH cohort", "t"] == 100.0
