"""Runs 15 and 16 (analysis/next_runs.py): table builders on fake persons.
No simulation is run."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import next_runs as nr  # noqa: E402


def persons():
    return pd.DataFrame({
        "weight": [2.0, 1.0, 1.0, 3.0],
        "employment_income": [30_000.0, 50_000.0, 0.0, 20_000.0],
        "savings_interest_income": [100.0, 0.0, 400.0, 0.0],
        "dividend_income": [0.0, 1_000.0, 3_000.0, 0.0],
    })


def central_draw(p):
    t = p.copy()
    t["displaced"] = [False, True, False, False]
    t["employment_income"] = [30_000.0 * 1.03, 0.0, 0.0, 20_000.0 * 1.02]
    t["savings_interest_income"] = p["savings_interest_income"] * nr.JR16_FACTOR
    t["dividend_income"] = p["dividend_income"] * nr.JR16_FACTOR
    return t


def test_obr_rate_converts_labour_force_points_to_employees():
    # 1.4 points of a 35m labour force = 0.49m = 1.75% of 28m employees
    assert nr.obr_displacement_rate(35.0, 28.0) == pytest.approx(0.0175)
    assert nr.OBR_UNEMPLOYMENT_RISE_PP == pytest.approx(1.4)
    with pytest.raises(ValueError):
        nr.obr_displacement_rate(20.0, 28.0)


def test_wage_flows():
    p = persons()
    f = nr.wage_flows(p, central_draw(p))
    assert f["wage_bill_lost"] == pytest.approx(50_000.0)
    assert f["survivor_gain"] == pytest.approx(2 * 900.0 + 3 * 400.0)


def test_obr_table_keeps_survivors_and_capital_at_baseline():
    p = persons()
    t = nr.obr_table(p, central_draw(p))
    assert list(t["employment_income"]) == [30_000.0, 0.0, 0.0, 20_000.0]
    assert t["dividend_income"].equals(p["dividend_income"])
    assert t["displaced"].tolist() == [False, True, False, False]


def test_capital_variants_scale_from_baseline_not_from_the_jr16_draw():
    p = persons()
    full = central_draw(p)
    v = {x.name: x for x in nr.CAPITAL_VARIANTS}
    t = nr.capital_variant_table(p, full, v["cap10"])
    assert np.allclose(t["dividend_income"], p["dividend_income"] * 1.1)
    assert np.allclose(t["employment_income"], full["employment_income"])  # labour kept
    t = nr.capital_variant_table(p, full, v["div_only"])
    assert np.allclose(t["savings_interest_income"], p["savings_interest_income"])
    assert np.allclose(t["dividend_income"], p["dividend_income"] * nr.JR16_FACTOR)


def test_closed_variant_adds_post_ct_profit_to_dividend_holders():
    p = persons()
    full = central_draw(p)
    v = {x.name: x for x in nr.CAPITAL_VARIANTS}
    t = nr.capital_variant_table(p, full, v["closed_k0375"])
    w = p["weight"].to_numpy()
    added = ((t["dividend_income"] - p["dividend_income"]) * w).sum()
    net = 50_000.0 - 3_000.0
    assert added == pytest.approx(0.375 * net)
    assert np.allclose(t["savings_interest_income"], p["savings_interest_income"])
    # pro rata: the holder with three times the dividends gets three times the rise
    rise = t["dividend_income"] - p["dividend_income"]
    assert rise[2] == pytest.approx(3 * rise[1])
    assert t.attrs["closed_wedge_bn"] == pytest.approx(net / 1e9)


def test_decile_flows_stabilisation():
    d = nr.decile_flows([1, 1, 2], [1.0, 1.0, 2.0], [-100.0, -100.0, 50.0], [30.0, 10.0, 0.0], [-20.0, 0.0, 5.0])
    d = d.set_index("decile")
    # decile 1: market -200, benefits +40, tax -20 -> (40 + 20)/200 = 0.3
    assert d.loc[1, "stabilisation"] == pytest.approx(0.3)
    assert np.isnan(d.loc[2, "stabilisation"])  # market gain: undefined


def test_summarise_reports_the_standard_error():
    s = nr.summarise(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), ["x"])["x"]
    assert s["se"] == pytest.approx(s["sd"] / 2)


def test_emitter_next_run_macros_on_fake_outputs(tmp_path, monkeypatch):
    import json

    import emit_tex_values as etv

    (tmp_path / "next_runs").mkdir()
    stat = lambda v: {"mean": v, "sd": 1.0, "se": 0.2, "n": 20}  # noqa: E731
    summary = {k: stat(v) for k, v in {
        "exchequer_cost_bn": 9.5, "household_basis_cost_bn": 7.0, "uc_change_bn": 1.0, "benefits_change_bn": 1.5,
        "employer_ni_change_bn": -2.0, "displaced_weighted_m": 0.5, "poverty_change_bhc_pp": 0.4,
        "gini_change_pp": 0.1}.items()}
    (tmp_path / "next_runs/obr_benchmark.json").write_text(json.dumps(
        {"displacement_rate": 0.0171, "summary": summary}))
    cap = {k: {"exchequer_cost_bn": stat(c), "poverty_change_bhc_pp": stat(2.0), "gini_change_pp": stat(1.0),
               "capital_income_added_bn": stat(12.0)}
           for k, c in [("cap10", 15.0), ("cap20", 11.0), ("cap60", 2.0), ("div_only", 9.0),
                        ("closed_k0375", -1.0), ("closed_k01875", 14.0), ("closed_k009375", 15.0),
                        ("closed_k075", -9.0)]}
    (tmp_path / "next_runs/capital_variants.json").write_text(json.dumps(cap))
    # the summaries are valid only with their draws (REQUIRES); the OBR lines
    # come from the draws: welfare = UC + other cash benefits, without
    # childcare support and statutory pay (split columns)
    draws = pd.DataFrame({
        "seed": [0, 1], "exchequer_cost_bn": [9.5, 9.5], "household_basis_cost_bn": [7.5, 7.5],
        "income_tax_change_bn": [-5.0, -5.0], "ni_employee_change_bn": [-1.0, -1.0],
        "employer_ni_change_bn": [-2.0, -2.0], "uc_change_bn": [1.0, 1.0], "benefits_change_bn": [1.5, 1.5],
        "childcare_change_bn": [-0.3, -0.3], "statutory_pay_change_bn": [-0.2, -0.2],
        "displaced_weighted_m": [0.5, 0.5], "public_paybill_displaced_bn": [6.0, 6.0],
        "wage_bill_lost_bn": [24.0, 24.0], "certified_data_build_id": "fake"})
    draws.to_csv(tmp_path / "next_runs/obr_benchmark_draws.csv", index=False)
    # run-16 draws: closed variants deduct corporation tax on phi x (W - S);
    # closed_k0375 (phi = 1): 3.0 - 0.25 x 16 = -1.0
    capd = pd.DataFrame([{"variant": k, "seed": s, "exchequer_cost_bn": (3.0 if k == "closed_k0375" else c),
                          "closed_wedge_bn": 16.0 if k.startswith("closed") else float("nan"),
                          "poverty_change_bhc_pp": 2.0, "gini_change_pp": 1.0, "capital_income_added_bn": 12.0,
                          "public_paybill_displaced_bn": 25.0, "certified_data_build_id": "fake"}
                         for k, c in [("cap10", 15.0), ("cap20", 11.0), ("cap60", 2.0), ("div_only", 9.0),
                                      ("closed_k0375", 3.0), ("closed_k01875", 14.0), ("closed_k009375", 15.0),
                                      ("closed_k075", -9.0)] for s in (0, 1)])
    capd.to_csv(tmp_path / "next_runs/capital_variants_draws.csv", index=False)
    monkeypatch.setattr(etv, "EXPECTED_CSV", {})
    for name, val in [("R", tmp_path), ("_checked", {}), ("lines", []), ("missing", []), ("emitted", set()),
                      ("values", {})]:
        monkeypatch.setattr(etv, name, val)
    monkeypatch.setattr(etv, "_build_id", lambda: "fake")
    etv.emit_next_runs()
    text = "\n".join(etv.lines)
    assert etv.missing == []
    assert "\\newcommand{\\genObrRatePct}{1.71}" in text
    assert "\\newcommand{\\genObrCostMeanBn}{9.5}" in text
    # receipts = income tax + employee NI + employer NI lost = 5 + 1 + 2
    assert "\\newcommand{\\genObrReceiptsBn}{8.0}" in text
    # welfare = benefits - childcare - statutory pay = 1.5 + 0.3 + 0.2
    assert "\\newcommand{\\genObrWelfareBn}{2.0}" in text
    assert "\\newcommand{\\genCapClosedOneCostMeanSignedBn}{-1.0}" in text or \
        "\\newcommand{\\genCapClosedOneCostMeanBn}{-1.0}" in text
    assert "Closed, $\\phi=1$, payout 50\\% & -1.0 $\\pm$ 0.0 & 4.0" in text
    assert "Interest and dividends $+10\\%$ & 15.0 $\\pm$ 0.0 & --" in text


def test_labour_force_counts_active_adults_only():
    status = ["FT_EMPLOYED", "PT_SELF_EMPLOYED", "UNEMPLOYED", "RETIRED", "STUDENT", "UNEMPLOYED"]
    age = [30, 40, 20, 70, 19, 15]
    w = [1e6, 2e6, 1e6, 5e6, 1e6, 1e6]
    assert nr.labour_force_m(status, age, w) == pytest.approx(4.0)


def test_public_paybill_counts_displaced_public_employees_only():
    p = pd.DataFrame({"weight": [2.0, 1.0, 1.0], "employment_income": [30_000.0, 20_000.0, 10_000.0]})
    t = pd.DataFrame({"displaced": [True, True, False]})
    got = nr.public_paybill(p, t, public=[True, False, True], employer_costs=[5_000.0, 1.0, 1.0])
    assert got == 2.0 * 35_000.0


def test_run16_order_puts_cap60_third():
    assert [v.name for v in nr.CAPITAL_VARIANTS] == [
        "cap10", "cap20", "cap60", "div_only", "closed_k0375", "closed_k01875", "closed_k009375", "closed_k075"]


def test_decile_flows_net_income_and_family_split():
    dec = np.array([1, 1, 2]); w = np.array([1.0, 2.0, 1.0])
    g = nr.decile_flows(dec, w, [-10.0, -10.0, -5.0], [3.0, 3.0, 1.0], [-1.0, -1.0, 0.0], [-6.0, -6.0, -4.0])
    assert g.net_income_change.tolist() == [-18.0, -4.0]
    f = nr.decile_family_flows(dec, ["A", "B", "A"], w, [-10.0, -10.0, -5.0], [3.0, 3.0, 1.0],
                               [-1.0, -1.0, 0.0], [-6.0, -6.0, -4.0])
    assert len(f) == 3 and f.weight.sum() == 4.0
    assert f.loc[(f.decile == 1) & (f.family_type == "B"), "stabilisation"].item() == pytest.approx(0.4)


def test_lf_components_rate():
    c = nr.lf_components(["FT_EMPLOYED", "UNEMPLOYED", "FT_SELF_EMPLOYED", "RETIRED"], [30, 30, 30, 70],
                         [8e6, 1e6, 1e6, 5e6])
    assert c["labour_force_m"] == 10.0 and c["unemployment_rate_pct"] == 10.0
