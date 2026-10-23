"""NEDC clinical sheet -> per-session flags, joined onto the recording index."""
import pandas as pd

from dataset_maker import tuh_clinical as tc

COLS = (["Subject ID"] + list(tc.DIAGNOSES) + ["Other diagnosis",
        "Major Medications (Comma Separated List)", "abnormal due to"] + list(tc.FINDINGS))


def _sheet(path, rows):
    df = pd.DataFrame([{c: "N" for c in COLS} | r for r in rows], columns=COLS)
    df.loc[len(df)] = {"Subject ID": 3}                         # the trailing totals row
    df.to_excel(path, sheet_name="TUEG", index=False)
    return path


def test_load_normalises_ids_flags_and_drug_classes(tmp_path):
    p = _sheet(tmp_path / "m.xlsx", [
        {"Subject ID": "aaaaaAal_s001", "Stroke": "Y", "abnormal due to": "slowing",
         "Major Medications (Comma Separated List)": "Keppraa, Ativan", "Other diagnosis": None},
        {"Subject ID": "bbbbbbbb_s002", "abnormal due to": None,
         "Major Medications (Comma Separated List)": "aspirin", "Other diagnosis": None},
    ])
    s = tc.load_sheet(p).set_index(["subject", "session"])
    assert list(s.index) == [("aaaaaaal", "s001"), ("bbbbbbbb", "s002")]   # lower-cased, total dropped
    a = s.loc[("aaaaaaal", "s001")]
    assert a.dx_stroke and a.any_neuro_dx and a.abnormal_report
    assert a.med_antiseizure and a.med_benzodiazepine and a.sedated
    b = s.loc[("bbbbbbbb", "s002")]
    assert not (b.any_neuro_dx or b.abnormal_report or b.sedated)


def test_join_and_normative_mask(tmp_path):
    sheet = tc.load_sheet(_sheet(tmp_path / "m.xlsx", [
        {"Subject ID": "aaaaaaaa_s001", "abnormal due to": None,
         "Major Medications (Comma Separated List)": None, "Other diagnosis": None},
        {"Subject ID": "bbbbbbbb_s001", "Epilepsy": "Y", "abnormal due to": None,
         "Major Medications (Comma Separated List)": None, "Other diagnosis": None},
    ]))
    index = pd.DataFrame({"stem": ["aaaaaaaa_s001_t000", "bbbbbbbb_s001_t000", "cccccccc_s001_t000",
                                   "aaaaaaaa_s001_t001"],
                          "subject": ["aaaaaaaa", "bbbbbbbb", "cccccccc", "aaaaaaaa"],
                          "session": ["s001"] * 4, "age": [40.0, 40.0, 40.0, 12.0]})
    j = tc.join_index(index, sheet)
    assert list(j.in_sheet) == [True, True, False, True]
    assert pd.isna(j.loc[2, "dx_epilepsy"])                     # not in sheet -> unknown, not False
    assert list(tc.normative_mask(j)) == [True, False, False, False]   # child excluded
    rep = tc.report(j)
    assert rep["normative_candidate"]["recordings"] == 1
