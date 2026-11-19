"""Dortmund Vital Study reader (dataset_maker/make_external_npy.py --dataset dortmund), the EDF
header audit and the results script, on a synthetic BIDS tree."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from dataset_maker import make_external_npy as mx  # noqa: E402

LABELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
          "O1", "O2", "FC5", "TP9", "Status"]
FS = 1000


def _write_edf(path, x_uv, phys_max):
    """Minimal EDF (16-bit, 1 s records) of ``x_uv`` [signals, samples] in µV with the given physical range."""
    ns, n = x_uv.shape
    n_rec = n // FS
    dig = np.clip(np.round((x_uv[:, :n_rec * FS] + phys_max) * 65535 / (2 * phys_max) - 32768), -32768, 32767).astype("<i2")

    def f(text, width):
        return str(text).ljust(width)[:width].encode("latin1")

    head = (f("0", 8) + f("X X 13-May-1951 Anonymous", 80) + f("Startdate 22-May-2024", 80) + f("22.05.24", 8)
            + f("10.23.48", 8) + f(256 + ns * 256, 8) + f("", 44) + f(n_rec, 8) + f(1, 8) + f(ns, 4))
    sig = b"".join(f(v, 16) for v in LABELS[:ns]) + f("", 80) * ns + f("uV", 8) * ns
    sig += f(-phys_max, 8) * ns + f(phys_max, 8) * ns + f(-32768, 8) * ns + f(32767, 8) * ns
    sig += f("LP:   NaN Hz; HP:   NaN Hz; Notch: NaN", 80) * ns + f(FS, 8) * ns + f("", 32) * ns
    with open(path, "wb") as fh:
        fh.write(head + sig)
        fh.write(dig.reshape(ns, n_rec, FS).transpose(1, 0, 2).tobytes())


def _signal(seconds, seed, offset_uv=1500.0):
    """10 µV alpha + noise on a DC offset of millivolts (the amplifier is DC-coupled)."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    x = 10 * np.sin(2 * np.pi * 10 * t)[None, :] + 3 * rng.standard_normal((len(LABELS), len(t)))
    x += offset_uv * rng.uniform(-1, 1, (len(LABELS), 1))
    x[-1] = 0                                                           # Status
    return x


def _tree(tmp_path):
    root = tmp_path / "ds005385"
    root.mkdir()
    (root / "participants.tsv").write_text(
        "participant_id\tsex\tage\thandedness\tsession1\tlate_ses1\tsession2\tlate_ses2\n"
        "sub-001\tF\t60\tright\tyes\t0\tyes\t0\nsub-002\tM\t31\tn/a\tyes\t12\tno\tn/a\n")
    k = 0
    for sub, sessions in (("sub-001", {"1": 2017, "2": 2022}), ("sub-002", {"1": 2019})):
        (root / sub).mkdir()
        (root / sub / f"{sub}_sessions.tsv").write_text(
            "session_id\trecording_year\n" + "".join(f"ses-{s}\t{y}\n" for s, y in sessions.items()))
        for ses in sessions:
            d = root / sub / f"ses-{ses}" / "eeg"
            d.mkdir(parents=True)
            for task in ("EyesClosed", "EyesOpen"):
                for acq in ("pre", "post"):
                    if ses == "2" and (task, acq) != ("EyesClosed", "pre"):
                        continue
                    base = d / f"{sub}_ses-{ses}_task-{task}_acq-{acq}"
                    x = _signal(70, k)
                    events = "onset\tduration\ttype\n1\tn/a\tboundary\n"
                    if (sub, task, acq) == ("sub-002", "EyesOpen", "post"):      # a 4 s piece, a jump, then 66 s
                        x[:, :4000] += 5000.0
                        events += "4001\tn/a\tboundary\n" + "".join(f"{4100 + 200 * i}\t1\tS  {i + 1}\n" for i in range(3))
                    _write_edf(str(base) + "_eeg.edf", x, phys_max=[3147.5, 10658.5, 16383.5][k % 3])
                    (d / (base.name + "_events.tsv")).write_text(events)
                    k += 1
    return root


def test_recordings_ages_and_flags(tmp_path):
    root = _tree(tmp_path)
    recs = {r["stem"]: r for r in mx.dortmund_recordings(str(root), str(root / "participants.tsv"))}
    assert len(recs) == 9 and "sub-001_ses-1_EC-pre_dortmund" in recs
    assert {r["label"] for r in recs.values()} == {"normal"}
    follow = recs["sub-001_ses-2_EC-pre_dortmund"]
    assert follow["age"] == 65.0 and follow["raw_age"] == 60.0 and "recording years" in follow["age_basis"]
    assert follow["age_session_matched"] and follow["recording_year"] == 2022.0
    flagged = recs["sub-002_ses-1_EO-post_dortmund"]
    assert (flagged["n_boundaries"], flagged["n_triggers"], flagged["late_triggers"]) == (2, 3, 12)
    assert flagged["handedness"] is None and flagged["task"] == "EO" and flagged["acq"] == "post"
    assert recs["sub-001_ses-1_EO-pre_dortmund"]["n_boundaries"] == 1


def test_open_applies_each_files_own_calibration_and_keeps_the_longest_segment(tmp_path):
    root = _tree(tmp_path)
    recs = {r["stem"]: r for r in mx.dortmund_recordings(str(root), str(root / "participants.tsv"))}
    stds = []
    for stem in ("sub-001_ses-1_EC-pre_dortmund", "sub-001_ses-1_EC-post_dortmund", "sub-001_ses-1_EO-pre_dortmund"):
        raw, present, reference = mx.open_dortmund(recs[stem]["_file"])           # three physical ranges
        assert "Status" not in raw.ch_names and len(present) == 19 and reference is None
        assert present["T3"] == "T7" and present["T6"] == "P8"
        x = raw.get_data(picks=list(present.values())) * 1e6
        assert np.abs(x.mean(axis=1)).max() < 1e-6                                # DC offset removed
        stds.append(np.median(x.std(axis=1)))
    np.testing.assert_allclose(stds, np.sqrt(50 + 9), rtol=0.03)                 # 10 µV sine + 3 µV noise, any range
    raw, present, _ = mx.open_dortmund(recs["sub-002_ses-1_EO-post_dortmund"]["_file"])
    assert raw.n_times == 66 * FS                                                 # the piece before the jump is cut
    assert np.abs(raw.get_data(picks=list(present.values())) * 1e6).max() < 60


def test_convert_writes_a_test_only_cohort(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    out = tmp_path / "cohort"
    monkeypatch.setattr(sys, "argv", ["make_external_npy", "--dataset", "dortmund", "--raw", str(root),
                                      "--meta", str(root / "participants.tsv"), "--out", str(out), "--jobs", "1"])
    mx.main()
    m = json.load(open(out / "manifest.json"))
    assert m["n_recordings"] == 9 and not m["cohort"]["errors"]
    rows = {r["stem"]: r for r in m["recordings"]}
    r = rows["sub-001_ses-1_EC-pre_dortmund"]
    assert r["n_windows"] == 7 and sum(r["channel_mask"]) == 19 and r["sfreq_native"] == 1000.0
    assert rows["sub-002_ses-1_EO-post_dortmund"]["n_windows"] == 6             # 66 s kept of 70
    x = np.load(out / "recordings" / f"{r['stem']}.npy").astype(np.float32)
    assert x.shape == (7 * 2000, 23) and np.abs(x[:, np.array(r["channel_mask"])]).max() < 100
    split = json.load(open(out / "age_split.json"))
    assert split["counts"] == {"train": 0, "val": 0, "test": 9}
    ages = json.load(open(out / "age_metadata.json"))["recordings"]
    assert ages["sub-001_ses-2_EC-pre_dortmund"]["age"] == 65.0 and ages[r["stem"]]["label"] == "normal"


def test_edf_header_audit_reads_ranges_and_flags(tmp_path):
    import edf_header_audit as ea
    x = _signal(5, 0)
    _write_edf(str(tmp_path / "a_eeg.edf"), x, phys_max=3147.5)
    a = ea.audit(str(tmp_path / "a_eeg.edf"))
    assert (a["n_signals"], a["n_eeg"], a["sfreq"], a["duration_sec"]) == (len(LABELS), len(LABELS) - 1, 1000.0, 5.0)
    np.testing.assert_allclose(a["step_max"], 2 * 3147.5 / 65535)
    assert not (a["flag_invalid_range"] or a["flag_ranges_differ"] or a["flag_coarse_step"])
    _write_edf(str(tmp_path / "b_eeg.edf"), x, phys_max=40000.0)                  # 1.2 µV per bit
    assert ea.audit(str(tmp_path / "b_eeg.edf"))["flag_coarse_step"]


def _runs(rng, n_subjects=120):
    rows = []
    for i in range(n_subjects):
        age, trait = float(rng.uniform(20, 70)), rng.normal(0, 3)
        for ses, add in (("1", 0.0), ("2", 5.0)) if i < 60 else (("1", 0.0),):
            for task, eo in (("EC", 0.0), ("EO", 1.5)):
                for acq in ("pre", "post"):
                    rows.append({"subject": f"s{i}", "session": ses, "task": task, "acq": acq, "age": age + add,
                                 "pred": age + add + trait + eo + rng.normal(0, 1.5), "flagged": False})
    return pd.DataFrame(rows)


def test_state_effects_and_longitudinal_change():
    import dortmund_brain_age as db
    runs = _runs(np.random.default_rng(0))
    state = pd.DataFrame(db.state_rows("M", runs)).set_index("contrast")
    eo = state.loc["eyes open - eyes closed, pre-task"]
    assert eo["lo"] < 1.5 < eo["hi"] and eo["n"] == 180                           # 120 + 60 sessions
    post = state.loc["post - pre, eyes closed"]
    assert post["lo"] < 0.0 < post["hi"]
    rows, wide = db.longitudinal_rows("M", runs)
    lon = pd.DataFrame(rows).set_index("comparison")
    r = lon.loc["session 2 vs session 1"]
    assert r["n"] == 60 == len(wide) and r["elapsed_years_mean"] == 5.0
    assert r["pred_change_lo"] < 5.0 < r["pred_change_hi"] and r["r_gap"] > 0.85 and r["icc_gap"] > 0.85
    assert lon.loc["session 1: post-task vs pre-task runs (2 h apart)", "n"] == 120
    ses = db.sessions(runs)
    assert len(ses) == 180 and (ses["n_runs"] == 4).all()


def test_icc_is_low_when_the_two_measurements_are_offset():
    import dortmund_brain_age as db
    a = np.random.default_rng(0).normal(0, 1, 200)
    assert db.icc_agreement(a, a) > 0.999
    assert db.icc_agreement(a, a + 3.0) < 0.3                                     # perfect correlation, no agreement


def test_rejection_scatter_sets_and_overlap():
    import rejection_scatter as rsc
    rng = np.random.default_rng(0)
    n = 200
    u = pd.DataFrame({"age": rng.uniform(20, 70, n), "spread": rng.random(n)})
    u["pred"] = u["age"] + 20 * u["spread"] * rng.choice([-1, 1], n)            # the uncertain ones are wrong
    u["total"] = u["spread"] + 0.05 * rng.random(n)
    u["learned_val+fold"] = rng.random(n)                                        # unrelated to the other two
    rej = rsc.rejected_sets(u, ["spread", "total", "learned"], 0.2)
    assert {k: int(v.sum()) for k, v in rej.items()} == {"spread": 40, "total": 40, "learned": 40}
    t = rsc.summary(u, rej).set_index("score")
    assert t.loc["spread", "mae_rejected"] > 2 * t.loc["spread", "mae_kept"]
    assert t.loc["spread & total", "jaccard"] > 0.7 > t.loc["spread & learned", "jaccard"]
    counts = t.loc[[f"rejected by {k} of 3 scores" for k in range(4)], "rejected"]
    assert counts.sum() == n
    assert "learned" not in rsc.rejected_sets(u.drop(columns="learned_val+fold"), ["spread", "learned"], 0.2)
