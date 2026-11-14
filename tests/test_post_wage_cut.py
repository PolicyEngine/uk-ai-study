"""Wage-cut post-processing (analysis/post_wage_cut.py) on fake data. No
simulation is run."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import post_wage_cut as pw  # noqa: E402


def wage_draws(skip=None):
    rows = []
    for v in ("table3", "cut_only"):
        for s in pw.SEEDS:
            if skip == (v, s):
                continue
            rows.append({"variant": v, "seed": s, "gross_cut_bn": 100.0,
                         "gross_cut_share_pct": 7.7,
                         "exchequer_cost_bn": (19.0 + 0.1 * s) if v == "table3" else 55.0,
                         "household_basis_cost_bn": 9.0 if v == "table3" else 40.0,
                         "employer_ni_change_bn": -9.5 if v == "table3" else -15.0,
                         "uc_change_bn": 0.6 if v == "table3" else 1.0,
                         "poverty_change_bhc_pp": 0.2, "poverty_change_ahc_pp": 0.4,
                         "relative_poverty_change_bhc_pp": -0.7, "gini_change_pp": 0.3})
    return pd.DataFrame(rows)


def existing_draws():
    rows = []
    for s in pw.SEEDS:
        rows.append({"run": "central", "seed": s, "exchequer_cost_bn": 7.0 + 0.1 * s,
                     "poverty_change_bhc_pp": 2.3, "poverty_change_ahc_pp": 2.5, "gini_change_pp": 1.7})
        rows.append({"run": "channel_displacement", "seed": s, "exchequer_cost_bn": 42.0,
                     "poverty_change_bhc_pp": 2.4, "poverty_change_ahc_pp": 2.6, "gini_change_pp": 1.0})
    return pd.DataFrame(rows)


def test_margin_comparison_pairs_by_seed():
    out = pw.margin_comparison(wage_draws(), existing_draws())
    assert out["draws"] == 20
    assert out["cost"]["job"] == pytest.approx(7.95)
    assert out["cost"]["wage"] == pytest.approx(19.95)
    # the seed trend is common to both rows, so the paired difference is exact
    assert out["cost"]["diff"] == pytest.approx(12.0)
    assert out["cost"]["diff_se"] == pytest.approx(0.0, abs=1e-12)
    assert out["cost"]["diff_min"] == pytest.approx(12.0)
    assert out["pov_bhc"]["diff"] == pytest.approx(-2.1)
    assert out["relative_pov_bhc_wage"] == pytest.approx(-0.7)


def test_cut_absorption_bases_add_up():
    out = pw.cut_absorption(wage_draws())
    assert out["gov_balance"] == pytest.approx(0.55)
    assert out["household"] == pytest.approx(0.40)
    assert out["employer_ni"] == pytest.approx(0.15)
    assert out["household"] + out["employer_ni"] == pytest.approx(out["gov_balance"])


def test_job_absorption_uses_the_paired_wage_bill():
    out = pw.job_absorption(wage_draws(), existing_draws())
    assert out["gov_balance"] == pytest.approx(0.42)
    assert out["draws"] == 20


def test_incomplete_variant_is_rejected():
    with pytest.raises(pw.Incomplete):
        pw.cut_absorption(wage_draws(skip=("cut_only", 7)))
    with pytest.raises(pw.Incomplete):
        pw.margin_comparison(wage_draws(skip=("table3", 0)), existing_draws())
    e = existing_draws()
    with pytest.raises(pw.Incomplete):
        pw.job_absorption(wage_draws(), e[~((e.run == "channel_displacement") & (e.seed == 3))])


def test_committed_run_is_paired_with_the_central_draws():
    """The committed wage-cut draws remove the central draw's wage bill."""
    import json
    root = Path(__file__).resolve().parents[1] / "results"
    f = root / "wage_cut" / "wage_cut_draws.csv"
    if not f.exists():
        pytest.skip("wage-cut run not present")
    d = pd.read_csv(f)
    wm = json.loads((root / "wage_margin_central.json").read_text())
    s0 = d[(d.variant == "table3") & (d.seed == 0)].iloc[0]
    assert s0["gross_cut_bn"] == pytest.approx(wm["gross_cut_realised"] / 1e9, rel=1e-6)
    assert s0["exchequer_cost_bn"] == pytest.approx(wm["exchequer_cost"] / 1e9, rel=1e-4)
    k = pw.cut_absorption(d)
    assert np.isclose(k["household"] + k["employer_ni"], k["gov_balance"], atol=2e-3)


def test_emit_wage_cut_macros_on_the_committed_run(monkeypatch):
    """Emitter wiring on the committed draws (read only; no simulation)."""
    import re
    import emit_tex_values as etv
    root = Path(__file__).resolve().parents[1] / "results"
    if not (root / "wage_cut" / "wage_cut_draws.csv").exists():
        pytest.skip("wage-cut run not present")
    monkeypatch.setattr(etv, "lines", [])
    monkeypatch.setattr(etv, "missing", [])
    monkeypatch.setattr(etv, "emitted", set())
    monkeypatch.setattr(etv, "values", {})
    etv.emit_wage_cut()
    vals = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(etv.lines)))
    assert not etv.missing, etv.missing
    assert vals["genWcDraws"] == "20"
    # the bases add up to the government balance within rounding
    assert abs(int(vals["genAbsCutHhPct"]) + int(vals["genAbsCutErNiPct"]) - int(vals["genAbsCutGovPct"])) <= 1
    assert float(vals["genWcWageCostMeanBn"]) > float(vals["genWcJobCostMeanBn"])
    assert float(vals["genWcWagePovBhcMeanPp"]) < float(vals["genWcJobPovBhcMeanPp"])
    assert "genWcMarginRows" in etv.emitted


def test_review_macros_on_committed_results(monkeypatch):
    """Household shares and closed-account crossovers (read only)."""
    import re
    import emit_tex_values as etv
    root = Path(__file__).resolve().parents[1] / "results"
    if not (root / "next_runs" / "obr_household_deciles.csv").exists():
        pytest.skip("run 15 not present")
    monkeypatch.setattr(etv, "lines", [])
    monkeypatch.setattr(etv, "missing", [])
    monkeypatch.setattr(etv, "emitted", set())
    monkeypatch.setattr(etv, "values", {})
    etv.emit_review()
    etv.emit_wage_cut()
    vals = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(etv.lines)))
    assert not etv.missing, etv.missing
    # households bear what the stabilisation coefficient leaves
    assert 50 < int(vals["genObrHhBearPct"]) < 80
    assert 50 < int(vals["genWcCutOnlyHhBearPct"]) < int(vals["genObrHhBearPct"])
    assert 0 < int(vals["genObrBenefitShareOfAbsorbedPct"]) < 20
    assert float(vals["genWcJobCostMeanTwoDpBn"]) + float(vals["genWcPairCostDiffTwoDpBn"]) == pytest.approx(
        float(vals["genWcWageCostMeanTwoDpBn"]), abs=0.011)


def test_closed_account_crossovers_on_committed_results(monkeypatch):
    """The closed cost is linear in phi; the crossovers reproduce Table 8."""
    import re
    import emit_tex_values as etv
    root = Path(__file__).resolve().parents[1] / "results"
    if not (root / "robustness" / "dividend_coverage.json").exists():
        pytest.skip("results not present")
    monkeypatch.setattr(etv, "lines", [])
    monkeypatch.setattr(etv, "missing", [])
    monkeypatch.setattr(etv, "emitted", set())
    monkeypatch.setattr(etv, "values", {})
    etv.emit_revision()
    vals = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", "\n".join(etv.lines)))
    new = ["genFaPhiCentralShareOne", "genFaPhiCentralShareHalf", "genFaPhiZeroShareOne", "genFaDivUnfinancedBn",
           "genIndexSlopeResidSdBn", "genAbsJobRestPct"]
    assert not [m for m in etv.missing if any(n in m for n in new)], etv.missing
    assert 0 < float(vals["genFaPhiCentralShareOne"]) < float(vals["genFaPhiCentralShareHalf"]) < 1
    assert float(vals["genFaPhiCentralShareOne"]) < float(vals["genFaPhiZeroShareOne"]) < 1
    c0 = etv._factor(0.0, 1.0)["closed_cost_bn"]
    c1 = etv._factor(1.0, 1.0)["closed_cost_bn"]
    mid = etv._factor(0.5, 1.0)["closed_cost_bn"]
    assert mid == pytest.approx((c0 + c1) / 2)
    assert float(vals["genFaDivUnfinancedBn"]) == pytest.approx(
        float(vals["genFaDivUpliftBn"]) - float(vals["genFaHhDivPhiOneShareOneBn"]), abs=0.11)
    assert int(vals["genAbsJobRestPct"]) == int(vals["genAbsJobGovPct"]) - int(vals["genAbsJobHhPct"])
