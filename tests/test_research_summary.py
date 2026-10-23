"""Tests for labram/eval/research_summary.py (the research-summary notebook's helpers)."""
import os

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
