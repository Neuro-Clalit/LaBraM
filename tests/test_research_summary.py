"""Tests for labram/eval/research_summary.py (the research-summary notebook's helpers)."""
import os

import numpy as np
import pandas as pd
import pytest

from labram.eval import research_summary as rs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGE_JSON = os.path.join(ROOT, "labram", "configs", "defaults", "finetune_tuab_age.json")
PAPER_JSON = os.path.join(ROOT, "labram", "configs", "defaults", "finetune_tuab_paper.json")
SCRIPT = os.path.join(ROOT, "scripts", "submit_tuh_cohorts.sh")
REGISTRY = os.path.join(ROOT, "docs", "age_overfitting_experiments.md")
AB = os.path.join(ROOT, "docs", "data", "line_removal_ab.json")
COHORT_CSV = os.path.join(ROOT, "docs", "data", "tuh_cohort_eval_v1.csv")
CLEARML_CSV = os.path.join(ROOT, "docs", "data", "brain_age_clearml_runs.csv")


def _key(override: str):
    return override.split("=", 1)[0]


@pytest.mark.parametrize("exp", [e for e in rs.PHASE2 if e.script])
def test_catalog_matches_the_submit_script(exp):
    """A run submitted by submit_tuh_cohorts.sh has the cohort, schedule and
    overrides of the script (the CV runs add the CV block, set per job there)."""
    s = rs.parse_submit_script(SCRIPT)[exp.script]
    assert (exp.cohort, exp.epochs, exp.warmup) == (s["cohort"], s["epochs"], s["warmup"])
    ours = {o for o in exp.extra if o not in rs.CV5}
    theirs = set(s["extra"])
    assert {_key(o) for o in ours} == {_key(o) for o in theirs}
    # compare the resolved configs, not the strings (the script prints 2e-4 as 0.0002)
    a = rs.flatten(rs.resolve_config(AGE_JSON, sorted(ours)).as_dict())
    b = rs.flatten(rs.resolve_config(AGE_JSON, sorted(theirs)).as_dict())
    assert a == b


def test_every_csv_model_is_in_the_catalog():
    names = set(pd.read_csv(COHORT_CSV).model)
    assert names <= set(rs.EXPERIMENTS)
    for e in rs.PHASE2:
        assert e.parent == "" or e.parent in rs.EXPERIMENTS, e.name


def test_resolve_config_applies_the_m1_recipe():
    c = rs.resolve_config(AGE_JSON, rs.experiment_overrides(rs.EXPERIMENTS["cvctrl"], "/x"))
    assert c.mixup.enabled and c.mixup.alpha == 0.4
    assert (c.trainer.epochs, c.optimizer.warmup_epochs, c.trainer.batch_size) == (10, 3, 128)
    assert c.optimizer.lr == pytest.approx(2e-4)
    assert c.model.session_features == "head" and c.model.use_mean_pooling is False
    assert c.cross_validation.enabled and c.cross_validation.n_folds == 5
    assert c.data.data_path == "/x/all_both_v3_cvcompact"


def test_config_diff_skips_bookkeeping_and_switched_off_sections():
    a = rs.resolve_config(AGE_JSON)
    b = rs.resolve_config(AGE_JSON, ["output.output_dir=/elsewhere", "mixup.alpha=0.2",
                                     "trainer.epochs=4"])
    d = rs.config_diff(a, b)
    assert list(d.field) == ["trainer.epochs"]          # mixup is off in both; output ignored
    d = rs.config_diff(rs.resolve_config(PAPER_JSON), a, ("paper", "age"))
    assert {"model.task", "optimizer.lr", "labram_plus.enabled"} <= set(d.field)


def test_registry_master_table_parses_numbers():
    t = rs.registry_master_table(REGISTRY)
    d = t.set_index("run")
    assert d.loc["D", "test"] == pytest.approx(8.30)
    assert d.loc["M1", "val"] == pytest.approx(7.46)
    assert d.loc["M1", "test_normal"] == pytest.approx(6.60)
    assert d.loc["E2", "epochs"] == 4
    assert len(t) >= 40


def test_seed_noise_pools_within_recipe_spread():
    runs, sd = rs.seed_noise(AB)
    assert len(runs) == 9 and runs.groupby("variant").size().eq(3).all()
    assert 0.0 < sd["val_mae"] < sd["test_mae"] < 0.5


def test_verdict_rule():
    t = rs.pair_threshold(0.1)
    assert t == pytest.approx(1.96 * 2 ** 0.5 * 0.1)
    assert rs.verdict(-0.5, t) == "improves"
    assert rs.verdict(0.5, t) == "hurts"
    assert rs.verdict(0.1, t) == "n.s."
    assert rs.verdict(-0.05, t, lo=-0.1, hi=-0.01) == "improves"
    assert rs.verdict(-0.3, t, lo=-0.6, hi=0.02) == "n.s."


def test_parse_markdown_table_strips_links():
    text = "x\n### Table 1.1. T\n\n| a | b |\n|---|---|\n| [`ab`](http://x) | $−0.4$ |\n\nafter"
    t = rs.parse_markdown_table(text, "### Table 1.1.")
    assert t.to_dict("records") == [{"a": "ab", "b": "−0.4"}]


def test_parse_markdown_table_keeps_link_targets_on_request():
    text = "### Table 1.1. T\n\n| a | b |\n|---|---|\n| [`ab`](http://x/1) | 2 |\n| c | 3 |\n"
    t = rs.parse_markdown_table(text, "### Table 1.1.", link_columns=("a",))
    assert list(t["a"]) == ["ab", "c"] and list(t["a url"]) == ["http://x/1", ""]


def test_clearml_link_shows_the_name_not_the_url():
    url = rs.clearml_url("task", "f" * 32)
    assert url == f"{rs.CLEARML_WEB}/projects/{rs.CLEARML_PROJECT}/experiments/{'f' * 32}/output/log"
    assert rs.clearml_url("project", "e" * 32) == f"{rs.CLEARML_WEB}/projects/{'e' * 32}/experiments"
    assert rs.clearml_link("age_<D>", url) == f'<a href="{url}" target="_blank">age_<wbr>&lt;D&gt;</a>'
    assert rs.clearml_link("x", "") == "–"


def test_clearml_csv_covers_every_run_and_matches_the_registry():
    """docs/data/brain_age_clearml_runs.csv names every phase-1 run of the
    registry (with the task id of its registry link) and every phase-2 / CV run."""
    reg = rs.registry_master_table(REGISTRY)
    want = rs.clearml_targets(reg).set_index("run")
    have = rs.load_clearml_runs(CLEARML_CSV)
    assert set(have.index) == set(want.index) == set(reg.run) | set(rs.EXPERIMENTS)
    assert (have.loc[want.index, ["kind", "id"]] == want[["kind", "id"]]).all().all()
    assert have.name.str.len().gt(0).all()
    assert have.loc["D", "url"] == reg.set_index("run").loc["D", "clearml_url"]


def test_every_run_has_a_training_loss():
    reg = rs.registry_master_table(REGISTRY)
    assert set(reg.run) == set(rs.PHASE1_LOSS)
    assert set(rs.PHASE1_LOSS.values()) <= set(rs.LOSSES)
    assert rs.run_loss("D2") == rs.LOSSES["l1"] and rs.run_loss("M1c") == rs.LOSSES["huber_cmixup"]
    for e in rs.PHASE2:
        aug = "channel_drop.consistency_weight=0.1" in e.extra
        assert rs.run_loss(e.name) == rs.LOSSES["consistency" if aug else "huber_mixup"]


def _tueg_frame():
    return pd.DataFrame({
        "subject": ["a", "a", "b", "c"], "session": ["s1", "s2", "s1", "s1"],
        "duration_sec": [600.0, 1200.0, 1800.0, 3600.0], "year": [2010, 2012, 1899, 2015],
        "missing_channels": [[], ["T1", "T2"], ["A1"], []], "sfreq": [256.0, 256.0, 250.0, 256.0],
        "montage": ["01_tcp_ar"] * 3 + ["02_tcp_le"], "sex": ["F", "F", "M", None],
        "age": [30.0, 32.0, 70.0, float("nan")], "raw_age": [30.0, 32.0, 70.0, 999.0],
        "status": ["normal", "abnormal", "abnormal", "no label"],
        "status_source": ["NEDC report", "TUAB folder", "NEDC report", "–"]})


def test_tueg_overview_and_status_counts():
    t = _tueg_frame()
    o = rs.tueg_overview(t).set_index("quantity").value
    assert o["recordings (EDF files)"] == "4" and o["patients"] == "3" and o["sessions"] == "4"
    assert o["hours of EEG"] == "2"
    assert o["missing only T1/T2"].startswith("1 ") and o["missing other channels"].startswith("1 ")
    assert o["age 90+ (redacted as 999)"].startswith("1 ") and o["age missing"].startswith("0 ")
    s = rs.tueg_status(t).set_index(["status", "source"])
    assert s.loc[("abnormal", "all"), "recordings"] == 2 and s.loc[("abnormal", "all"), "age mean"] == 51.0
    assert s.loc[("total", "all"), "recordings"] == 4 and s.loc[("total", "all"), "age n"] == 3
    assert s.loc[("total", "all"), "% of TUEG"] == 100.0


def test_split_fingerprint_compares_recordings_not_windows():
    a = {"train": {"recordings": ["r2", "r1"], "n_subjects": 2, "n_windows": 10},
         "val": {"recordings": ["r3"], "n_subjects": 1, "n_windows": 3}}
    b = {"train": {"recordings": ["r1", "r2"], "n_subjects": 2, "n_windows": 99},
         "val": {"recordings": ["r4"], "n_subjects": 1, "n_windows": 3}}
    rows = [{"run": run, **r} for run, d in (("a", a), ("b", b)) for r in rs.split_fingerprint(d)]
    sig = rs.split_signature(pd.DataFrame(rows))
    ta, va, xa = sig["a"].split("/")
    tb, vb, xb = sig["b"].split("/")
    assert ta == tb and va != vb and xa == xb == "-"


def test_phase1_runs_share_one_split():
    splits = pd.read_csv(os.path.join(ROOT, "docs", "data", "brain_age_run_splits.csv"))
    sig = rs.split_signature(splits)
    reg = rs.registry_master_table(REGISTRY)
    assert sig.loc[list(reg.run)].nunique() == 1
    test = {s.split("/")[2] for s in sig if s.split("/")[2] != "-"}
    assert len(test) == 1                      # every run tests on the same TUAB recordings


def test_age_fit_metrics_bias_and_slope():
    true = pd.Series([20.0, 40.0, 60.0, 80.0])
    m = rs.age_fit_metrics(true + 2.0, true)              # constant offset: bias 2, no slope
    assert m["MAE"] == pytest.approx(2.0) and m["bias"] == pytest.approx(2.0) and m["β"] == pytest.approx(0.0)
    m = rs.age_fit_metrics(pd.Series([50.0] * 4), true)  # constant prediction: β = -1
    assert m["β"] == pytest.approx(-1.0)


def test_prediction_sets_split_tuab_tueg_and_status():
    preds = pd.DataFrame({
        "exp": ["x"] * 6, "fold": [0, 1, 0, 1, 0, 0],
        "split": ["heldout", "heldout", "heldout", "heldout", "test", "test"],
        "stem": ["t1", "t1", "t2", "t2", "a1", "g1"],
        "pred": [30.0, 34.0, 60.0, 62.0, 40.0, 50.0], "true": [32.0, 32.0, 61.0, 61.0, 41.0, 55.0]})
    preds["err"] = preds.pred - preds.true
    recs = pd.DataFrame({"stem": ["t1", "t2", "a1", "g1"], "status": ["normal", "abnormal", "normal", "no label"],
                         "corpora": ["tuab,tueg", "tuab,tueg", "tuab,tueg", "tueg"]})
    s = rs.prediction_sets(preds, recs).set_index(["set", "stem"])
    assert s.loc[("TUAB test, ensemble", "t1"), "pred"] == 32.0          # median of the fold models
    assert ("TUAB out-of-fold", "a1") in s.index and ("TUEG out-of-fold", "g1") in s.index
    assert s.loc[("TUEG out-of-fold", "g1"), "status"] == "no label"
    m = rs.prediction_metrics(s.reset_index())
    assert set(m.status) >= {"all", "normal", "abnormal", "no label"}


def _toy_sets():
    true_oof = np.arange(20.0, 81.0, 5.0)
    true_test = np.array([25.0, 50.0, 75.0])
    shrink = lambda y: 50.0 + 0.6 * (y - 50.0)              # regression to the mean, beta = -0.4
    rows = [{"exp": "x", "set": "TUEG out-of-fold", "stem": f"o{i}", "true": y, "pred": shrink(y),
             "status": "normal"} for i, y in enumerate(true_oof)]
    rows += [{"exp": "x", "set": "TUAB test, ensemble", "stem": f"t{i}", "true": y, "pred": shrink(y),
              "status": "abnormal"} for i, y in enumerate(true_test)]
    d = pd.DataFrame(rows)
    d["err"] = d.pred - d.true
    return d


def test_bias_correction_is_fitted_out_of_sample():
    c = rs.bias_corrected_test(_toy_sets(), degree=1)
    assert set(c.set) == {"TUAB test, ensemble"}
    assert np.allclose(c.err, 0.0)                         # the linear trend fitted on OOF removes it exactly
    d = _toy_sets()
    d.loc[d.set == "TUEG out-of-fold", "pred"] += 3.0     # shift only the OOF rows ...
    d["err"] = d.pred - d.true
    c = rs.bias_corrected_test(d, degree=1)
    assert np.allclose(c.err, -3.0)                        # ... so the fit comes from them, not from the test rows


def test_agreement_summary_pairs():
    idx = [f"s{i}" for i in range(6)]
    true = pd.Series([20.0, 30.0, 40.0, 50.0, 60.0, 70.0], index=idx)
    status = pd.Series(["normal", "abnormal"] * 3, index=idx)
    rng = np.random.default_rng(0)
    wide = pd.DataFrame({f"{e} f{k}": true + rng.normal(0, 2, 6) for e in ("a", "b") for k in range(3)}, index=idx)
    for e in ("a", "b"):
        wide[f"{e} ens"] = wide[[f"{e} f{k}" for k in range(3)]].median(axis=1)
    s = rs.agreement_summary(wide, true, status, ("a", "b"))
    assert set(s.status) == {"all", "normal", "abnormal"}
    assert (s.pair == "a vs b (ensembles)").sum() == 3
    assert s.agree_5y.between(0, 1).all() and s.r_pred.le(1).all()


def test_bias_fit_check_and_deployable_correction():
    d = _toy_sets()
    chk = rs.bias_fit_check(d).iloc[0]
    assert chk["fit ∩ test recordings"] == 0 and chk["fit ∩ test patients"] == 0
    leaky = pd.concat([d, d[d.set == "TUAB test, ensemble"].assign(set="TUEG out-of-fold")], ignore_index=True)
    assert rs.bias_fit_check(leaky).iloc[0]["fit ∩ test recordings"] == 3   # a leak is detected
    c = rs.cole_corrected_test(d)                       # inverts pred = 0.6 y + 20 using the prediction only
    assert np.allclose(c.pred, c.true)


def test_display_name_capitalizes_corpora_only():
    assert rs.display_name("normal_tueg_v1") == "normal_TUEG_v1"
    assert rs.display_name("tuab_v3.0.1_interp50-60") == "TUAB_v3.0.1_interp50-60"
    assert rs.display_name("all_both_v3") == "all_both_v3"


def test_raw_corpora_counts_status_and_channels():
    t = _tueg_frame().assign(corpora=["tuab,tueg", "tuab,tueg", "tueg", "tueg"],
                             source_split=["train", "eval", None, None])
    r = rs.raw_corpora(t).set_index(["corpus", "part"])
    assert r.loc[("TUAB", "all"), "recordings"] == 2 and r.loc[("TUEG", "all"), "recordings"] == 4
    assert r.loc[("TUEG", "not in TUAB"), "patients"] == 2
    assert r.loc[("TUAB", "eval split (test)"), "abnormal"] == 1
    tueg = r.loc[("TUEG", "all")]
    assert (tueg["all 23 channels"], tueg["only T1/T2 missing"], tueg["other channels missing"]) == (2, 1, 1)
    assert tueg["normal"] + tueg["abnormal"] + tueg["no label"] == tueg["recordings"]


def test_raw_summary_rows_and_selection():
    t = _tueg_frame().assign(stem=["r1", "r2", "r3", "r4"], corpora=["tuab,tueg", "tuab,tueg", "tueg", "tueg"],
                             source_split=["train", "eval", None, None])
    r = rs.raw_summary(t, converted=["r1", "r2", "r3"], cohort=pd.DataFrame({"stem": ["r1", "r2"]}))
    assert list(r.part.unique()) == ["TUAB train", "TUAB eval (test)", "TUEG (not in TUAB)", "TOTAL"]
    total = r[r.part == "TOTAL"].iloc[0]
    assert (total.recordings, total["readable EDF, ≥ 60 s"], total["age known"]) == (4, 3, 3)
    assert (total["selected recordings"], total.removed, total["T1/T2 missing (selected)"]) == (2, 2, 1)
    tueg_all = r[(r.part == "TUEG (not in TUAB)") & (r.status == "all")].iloc[0]
    assert tueg_all.recordings == 2


def test_removal_reasons_first_failing_rule(tmp_path):
    t = _tueg_frame().assign(stem=["r1", "r2", "r3", "r4"], corpora=["tuab,tueg", "tuab,tueg", "tueg", "tueg"],
                             source_split=["train", "eval", None, None])
    split = tmp_path / "age_split.json"
    split.write_text('{"files": {"train": ["train/a_s1_t0_0.pkl"], "test": ["test/b_s1_t0_0.pkl"]}}')
    manifest = {"cohort": {"min_age": 18, "max_age": 89, "min_minutes": 5.0, "allow_missing": ["T1", "T2"]}}
    r = rs.removal_reasons(t, converted=["r1", "r2", "r3"], cohort=pd.DataFrame({"stem": ["r1"]}),
                           cohort_manifest=manifest, tuab_split_json=str(split))
    tot = r[("TOTAL", "recordings")]
    assert tot["selected"] == 1 and tot["raw"] == 4
    assert tot["over the per-patient cap"] == 1                      # r2: passes every rule, not selected
    assert tot["channels other than the allowed ones missing"] == 1  # r3 lacks A1
    assert tot["EDF unreadable (not converted)"] == 1                # r4 (>= 60 s, not converted)
