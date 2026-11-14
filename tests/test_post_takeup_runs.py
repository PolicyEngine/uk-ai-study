"""Take-up post-processing (analysis/post_takeup_runs.py) and its emitter
wiring (emit_tex_values.emit_takeup) on fake data. No simulation is run."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import post_takeup_runs as pt  # noqa: E402


def run15(n=20):
    s = np.arange(n)
    return pd.DataFrame({
        "seed": s, "displaced_weighted_m": 0.5 + 0 * s, "wage_bill_lost_bn": 24.0 + 0 * s,
        "uc_change_bn": 0.7 + 0 * s, "benefits_change_bn": 0.4 + 0 * s, "childcare_change_bn": -0.25 + 0 * s,
        "statutory_pay_change_bn": -0.05 + 0 * s, "exchequer_cost_bn": 10.0 + 0.1 * s,
        "poverty_change_bhc_pp": 0.6 + 0 * s})


def test_obr_per_head():
    out = pt.obr_per_head(run15(), labour_force_m=35.0, rise_pp=1.4, welfare_bn=3.0)
    assert out["obr_extra_unemployed_m"] == pytest.approx(0.49)
    assert out["obr_welfare_per_extra_unemployed_gbp"] == pytest.approx(3e9 / 0.49e6)
    assert out["uc_per_displaced_gbp"] == pytest.approx(1400.0)
    assert out["mean_lost_earnings_gbp"] == pytest.approx(48_000.0)
    assert out["benefits_pence"] == pytest.approx(100 * 0.4 / 24)
    # cash perimeter adds back the lost childcare and statutory pay
    assert out["cash_benefits_bn"] == pytest.approx(0.7)
    assert out["cash_benefits_pence"] == pytest.approx(100 * 0.7 / 24)


def test_dolls_working_paper_uk_row_adds_up():
    d = pt.DOLLS_WP_UK_UNEMPLOYMENT
    assert d["benefits"] == 0.163
    assert d["income_tax"] + d["state_tax"] + d["social_insurance"] + d["benefits"] == pytest.approx(d["total"])


def takeup_draws(skip=None):
    rows = []
    for scen, seeds in pt.TAKEUP_SEEDS.items():
        for v, extra in (("takeup1", 0.0), ("jsa", 0.5)):
            for s in seeds:
                if skip == (scen, v, s):
                    continue
                rows.append({"scenario": scen, "variant": v, "seed": s, "displaced_weighted_m": 0.5,
                             "wage_bill_lost_bn": 97.0 if scen == "disp_only" else 24.0,
                             "exchequer_cost_bn": (42.0 if scen == "disp_only" else 10.5 + 0.1 * s) + extra,
                             "narrow_cash_change_bn": -30.0 - extra, "uc_change_bn": 1.4,
                             "cash_benefits_change_bn": 1.2, "poverty_change_bhc_pp": 0.55, "gini_change_pp": 0.2, "jsa_change_bn": extra,
                             "jsa_eligible_m": 0.48 if v == "jsa" else 0.0})
    return pd.DataFrame(rows)


def test_variant_means_and_completeness():
    m = pt.variant_means(takeup_draws(), "obr", "takeup1")
    assert m["uc_per_displaced_gbp"] == pytest.approx(2800.0)
    assert m["cash_benefits_pence"] == pytest.approx(5.0)
    with pytest.raises(pt.Incomplete):
        pt.variant_means(takeup_draws(skip=("obr", "jsa", 7)), "obr", "jsa")


def test_paired_change_requires_identical_displaced_sets():
    d = takeup_draws()
    ex = run15()
    assert pt.paired_change(d, "obr", "takeup1", ex, "exchequer_cost_bn") == pytest.approx(0.5)
    ex.loc[3, "displaced_weighted_m"] = 0.51
    with pytest.raises(ValueError):
        pt.paired_change(d, "obr", "takeup1", ex, "exchequer_cost_bn")
    with pytest.raises(ValueError):
        pt.paired_change(d, "obr", "takeup1", run15(10), "exchequer_cost_bn")


def test_absorption_variants():
    a = pt.absorption_variants(takeup_draws(), 97.0)
    assert a["takeup1"]["gov_balance"] == pytest.approx(42 / 97)
    assert a["jsa"]["household"] == pytest.approx(30.5 / 97)
    with pytest.raises(ValueError):
        pt.absorption_variants(takeup_draws(), 90.0)


def test_emitter_marks_partial_takeup_draws_pending(tmp_path, monkeypatch):
    import emit_tex_values as etv
    (tmp_path / "takeup").mkdir()
    takeup_draws(skip=("central", "jsa", 19)).to_csv(tmp_path / "takeup/takeup_draws.csv", index=False)
    monkeypatch.setattr(etv, "R", tmp_path)
    monkeypatch.setattr(etv, "_checked", {})
    with pytest.raises(etv.Pending, match="81 of 82"):
        etv.csv("takeup/takeup_draws.csv")
    # the summary is only valid with its complete draws
    (tmp_path / "takeup/takeup_runs.json").write_text("{}")
    with pytest.raises(etv.Pending):
        etv.jload("takeup/takeup_runs.json")


def test_emit_takeup_macros(monkeypatch):
    import emit_tex_values as etv
    monkeypatch.setattr(etv, "lines", [])
    monkeypatch.setattr(etv, "missing", [])
    monkeypatch.setattr(etv, "emitted", set())
    monkeypatch.setattr(etv, "values", {})
    central = run15().assign(run="central", exchequer_cost_bn=7.0)
    files = {"next_runs/obr_benchmark_draws.csv": run15(),
             "robustness/dividend_coverage_draws.csv": central,
             "takeup/takeup_draws.csv": takeup_draws()}
    jsons = {"next_runs/obr_benchmark.json": {"labour_force_m": 35.0},
             "wage_margin_central.json": {"gross_cut_realised": 97e9},
             "takeup/takeup_runs.json": {
                 "diagnostics": {"obr": {"flag_true_share": 0.48, "flag_true_share_entitled_after": 0.53,
                                         "flag_true_share_newly_entitled": 0.47, "over_capital_limit_share": 0.41,
                                         "other_earner_share": 0.58, "owner_share": 0.70}},
                 "existing_unemployed": {"flag_true_share_entitled": 0.88, "uc_per_person_gbp": 6045.0},
                 "jsa": {"taxed_by_model": True}}}
    monkeypatch.setattr(etv, "csv", lambda rel: files[rel].copy())
    monkeypatch.setattr(etv, "jload", lambda rel: jsons[rel])
    etv.emit_takeup()
    vals = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(etv.lines)))
    assert not etv.missing, etv.missing
    assert vals["genObrWelfarePerUnempGbp"] == "6,100"  # 3bn / (1.4% of 35m)
    assert vals["genObrCashBenefitsPence"] == "2.9"
    assert vals["genObrBenefitsPence"] == "1.7"
    assert vals["genDollsUkBenefitsPct"] == "16.3"
    assert vals["genDollsUkTotalPct"] == "41.5"
    assert vals["genTkObrTakeupCostDiffBn"] == "+0.5"  # paired with run 15 seed by seed
    assert vals["genTkCentralJsaCostDiffBn"] == "+5.0"  # 11.95 - 7.0, paired with the central draws
    assert vals["genTkObrTakeupUcPerDisplacedGbp"] == "2,800"
    assert vals["genTkObrJsaEligiblePct"] == "96"
    assert vals["genAbsJobGovTakeupPct"] == "43"
    assert vals["genAbsJobHhJsaPct"] == "31"
    assert vals["genTkFlagTrueEntitledPct"] == "53"
    assert vals["genTkUnempFlagEntitledPct"] == "88"
    assert vals["genTkUnempUcPerPersonGbp"] == "6,000"
    assert vals["genTkJsaTaxWords"] == "counts it as taxable income"


def test_ordering_guard_flags_a_reversal(monkeypatch):
    import emit_tex_values as etv
    monkeypatch.setattr(etv, "missing", [])
    monkeypatch.setattr(etv, "emitted", set())
    monkeypatch.setattr(etv, "values", {"genAbsCutGovPct": 54.0, "genAbsJobGovPct": 42.0,
                                        "genAbsJobGovJsaPct": 55.0})
    monkeypatch.setattr(etv, "DIRECTIONS", {})
    etv.check_directions()
    assert len(etv.missing) == 1 and "genAbsJobGovJsaPct" in etv.missing[0]
