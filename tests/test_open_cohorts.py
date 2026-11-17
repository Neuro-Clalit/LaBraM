"""Readers of the open external cohorts (AHEPA, Iowa and Tel Aviv Parkinson's, SRM) in
dataset_maker/make_external_npy.py, and scripts/clinical_cohort_gap.py."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from dataset_maker import make_external_npy as mx  # noqa: E402

TEN_TWENTY = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
              "O1", "O2"]


def _raw(names, data_uv, sfreq=250.0, annotations=()):
    import mne
    raw = mne.io.RawArray(np.asarray(data_uv) * 1e-6, mne.create_info(list(names), sfreq, "eeg"), verbose="error")
    if annotations:
        raw.set_annotations(mne.Annotations([a[0] for a in annotations], [0.0] * len(annotations),
                                            [a[1] for a in annotations]))
    return raw


def _true_signals(seconds, sfreq, seed=0):
    rng = np.random.default_rng(seed)
    return 10 * rng.standard_normal((len(TEN_TWENTY), int(seconds * sfreq)))


def _car19(raw, present):
    """What the pipeline's ``reference='average'`` leaves: the 19 picked channels minus their mean."""
    x = raw.get_data(picks=[present[c] for c in sorted(present)]) * 1e6
    return x - x.mean(axis=0, keepdims=True), sorted(present)


def test_keep_longest_segment():
    raw = _raw(["Cz"], np.arange(1000.0)[None, :], sfreq=100.0)
    mx.keep_longest_segment(raw, [0.0, 2.0, 9.5])            # pieces of 2 s, 7.5 s and 0.5 s
    assert raw.n_times == 750 and round(raw.get_data()[0, 0] * 1e6) == 200
    raw = _raw(["Cz"], np.arange(1000.0)[None, :], sfreq=100.0)
    mx.keep_longest_segment(raw, [0.0])                      # only the start marker: nothing is cut
    assert raw.n_times == 1000


def test_iowa_gets_pz_back_from_the_pz_reference(monkeypatch):
    import mne
    s = _true_signals(20, 500.0)
    pz = TEN_TWENTY.index("Pz")
    recorded = np.delete(s - s[pz], pz, axis=0) + 4000.0     # referenced to Pz, Pz not stored, DC offset
    names = [c for c in TEN_TWENTY if c != "Pz"] + ["FC1"]
    recorded = np.vstack([recorded, s[0] - s[pz]])
    monkeypatch.setattr(mne.io, "read_raw_eeglab", lambda *a, **k: _raw(names, recorded, 500.0, [(0.0, "boundary")]))
    raw, present, reference = mx.open_iowapd("x.set")
    assert reference == "average" and len(present) == 19 and present["PZ"] == "Pz"
    car, order = _car19(raw, present)
    want = s - s.mean(axis=0, keepdims=True)                 # the true common average reference
    want = want - want.mean(axis=1, keepdims=True)
    for k, tuh in enumerate(order):
        src = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(tuh, tuh.capitalize())
        np.testing.assert_allclose(car[k], want[TEN_TWENTY.index(src)], atol=1e-6)


def test_telaviv_renames_the_vertex_reference_and_cuts_at_a_break(monkeypatch):
    import mne
    s = _true_signals(100, 250.0)
    cz = TEN_TWENTY.index("Cz")
    recorded = s - s[cz]
    names = [c if c != "Cz" else "VREF" for c in TEN_TWENTY] + ["EOG1"]
    recorded = np.vstack([recorded, 50 * np.ones((1, recorded.shape[1]))]) - 70000.0
    recorded[cz] = 0.0                                       # the reference channel is flat
    marks = [(0.0, "boundary"), (3.0, "bgin"), (80.0, "break cnt"), (80.1, "good")]
    monkeypatch.setattr(mne.io, "read_raw_eeglab", lambda *a, **k: _raw(names, recorded, 250.0, marks))
    raw, present, reference = mx.open_telavivpd("x.set")
    assert reference == "average" and len(present) == 19 and present["CZ"] == "Cz" and "VREF" not in raw.ch_names
    assert raw.n_times == 80 * 250                           # the 20 s after the break are dropped
    assert np.abs(raw.get_data().mean(axis=1)).max() < 1e-9  # DC offset removed
    car, order = _car19(raw, present)
    assert np.std(car[order.index("CZ")]) > 5                # Cz carries a signal again


def test_ahepa_keeps_the_19_channels(monkeypatch):
    import mne
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz",
             "Cz", "Pz"]
    monkeypatch.setattr(mne.io, "read_raw_eeglab", lambda *a, **k: _raw(names, _true_signals(20, 500.0) + 100.0, 500.0))
    raw, present, reference = mx.open_ahepa("x.set")
    assert reference is None and len(present) == 19 and present["T5"] == "T5"
    assert np.abs(raw.get_data().mean(axis=1)).max() < 1e-9


def _tree(tmp_path, name, header, rows, files):
    root = tmp_path / name
    root.mkdir()
    (root / "participants.tsv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8-sig")
    for rel in files:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    return root


def test_listers_read_groups_ages_and_scores(tmp_path):
    root = _tree(tmp_path, "ahepa", "participant_id\tGender\tAge\tGroup\tMMSE",
                 ["sub-001\tF\t57\tA\t16", "sub-002\tM\t70\tC\t30", "sub-003\tM\t66\tF\t22"],
                 [f"sub-00{i}/eeg/sub-00{i}_task-eyesclosed_eeg.set" for i in (1, 2, 3)])
    r = {x["stem"]: x for x in mx.ahepa_recordings(str(root), str(root / "participants.tsv"))}
    assert r["sub-001_ahepa"]["dx_ad"] and not r["sub-001_ahepa"]["healthy"] and r["sub-001_ahepa"]["label"] is None
    assert r["sub-002_ahepa"]["healthy"] and r["sub-002_ahepa"]["label"] == "normal" and r["sub-002_ahepa"]["mmse"] == 30
    assert r["sub-003_ahepa"]["clinical_group"] == "ftd" and r["sub-003_ahepa"]["age"] == 66.0

    root = _tree(tmp_path, "iowa", "participant_id\tGROUP\tID\tEEG\tAGE\tGENDER\tMOCA\tUPDRS\tTYPE",
                 ["sub-001\tPD\t1\tPD1\t80\tM\t19\t28\t1", "sub-002\tControl\t2\tC2\t71\tF\t27\tn/a\t0"],
                 [f"sub-00{i}/eeg/sub-00{i}_task-Rest_eeg.set" for i in (1, 2)])
    r = {x["stem"]: x for x in mx.iowapd_recordings(str(root), str(root / "participants.tsv"))}
    assert r["sub-001_iowapd"]["dx_pd"] and r["sub-001_iowapd"]["updrs"] == 28.0
    assert r["sub-002_iowapd"]["healthy"] and r["sub-002_iowapd"]["updrs"] is None and r["sub-002_iowapd"]["sex"] == "F"

    root = _tree(tmp_path, "telaviv", "participant_id\tsubject_id\tgroup\tupdrs_total\tmoca\tage\tsex\tdisease_duration\tledd",
                 ["sub-001\tHC1\tHC\t0.0\t30.0\t42.0\tM\tn/a\tn/a", "sub-002\tPD1\tPD\t40.0\t25.0\t66.0\tF\t5.0\t600.0"],
                 ["sub-001/eeg/sub-001_task-rest_eeg.set", "sub-001/eeg/sub-001_task-walk_eeg.set",
                  "sub-002/eeg/sub-002_task-rest_eeg.set"])
    r = {x["stem"]: x for x in mx.telavivpd_recordings(str(root), str(root / "participants.tsv"))}
    assert set(r) == {"sub-001_rest_telavivpd", "sub-002_rest_telavivpd"}        # the walking run is left out
    assert r["sub-002_rest_telavivpd"]["ledd"] == 600.0 and r["sub-001_rest_telavivpd"]["healthy"]


def test_srm_second_session_age_and_unit_scaling(tmp_path):
    from test_dortmund import LABELS, _signal, _write_edf
    root = tmp_path / "srm"
    for ses, when in (("t1", "2017-05-02T17:42:06"), ("t2", "2018-05-02T09:00:00")):
        d = root / "sub-001" / f"ses-{ses}" / "eeg"
        d.mkdir(parents=True)
        (d.parent / f"sub-001_ses-{ses}_scans.tsv").write_text(f"filename\tacq_time\neeg/x.edf\t{when}\n")
        edf = d / f"sub-001_ses-{ses}_task-resteyesc_eeg.edf"
        _write_edf(str(edf), _signal(12, 3), phys_max=3147.5)
        with open(edf, "r+b") as fh:                         # SRM's headers carry no physical unit
            fh.seek(256 + len(LABELS) * (16 + 80))
            fh.write(b" " * 8 * len(LABELS))
    (root / "participants.tsv").write_text("participant_id\tage\tsex\nsub-001\t29\tf\n")
    recs = {r["stem"]: r for r in mx.srm_recordings(str(root), str(root / "participants.tsv"))}
    assert recs["sub-001_ses-t1_srm"]["age"] == 29.0 and recs["sub-001_ses-t1_srm"]["sex"] == "F"
    assert abs(recs["sub-001_ses-t2_srm"]["age"] - 30.0) < 0.01 and recs["sub-001_ses-t2_srm"]["age_session_matched"]
    raw, present, reference = mx.open_srm(recs["sub-001_ses-t1_srm"]["_file"])
    x = raw.get_data(picks=list(present.values())) * 1e6
    assert reference is None and len(present) == 19
    np.testing.assert_allclose(np.median(x.std(axis=1)), np.sqrt(50 + 9), rtol=0.05)   # µV, not volts


def test_clinical_gap_is_adjusted_for_age():
    import clinical_cohort_gap as cg
    rng = np.random.default_rng(0)
    n = 400
    patient = rng.random(n) < 0.5
    age = np.where(patient, rng.uniform(60, 85, n), rng.uniform(45, 70, n))     # the patients are older
    gap = -0.3 * (age - 60) + 3.0 * patient + 3 * rng.standard_normal(n)        # compression + a true gap of 3
    d = pd.DataFrame({"age": age, "gap": gap, "sex": rng.choice(["M", "F"], n)})
    mask = pd.Series(patient)
    raw_diff = d.gap[patient].mean() - d.gap[~patient].mean()
    out = cg.adjusted_gap(d, mask, ~mask)
    assert raw_diff < 0.5 and out["gap_adj_lo"] < 3.0 < out["gap_adj_hi"] and out["gap_adj_p"] < 1e-6
    score = 30 - 2 * gap + rng.standard_normal(n)                               # a score that falls with the gap
    r, p, m = cg.partial_r(gap, score, age)
    assert r < -0.9 and p < 1e-6 and m == n
    assert np.isnan(cg.partial_r(gap[:5], score[:5], age[:5])[0])
