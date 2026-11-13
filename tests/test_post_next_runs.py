"""Post-processing of runs 15-16 (analysis/post_next_runs.py) on fake data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import post_next_runs as pn  # noqa: E402


def variant_rows(name, seeds=range(20), build="B", fp="F", paybill=None, cost=1.0):
    return pd.DataFrame({
        "variant": name, "seed": list(seeds), "certified_data_build_id": build, "code_fingerprint": fp,
        "public_paybill_displaced_bn": [paybill if paybill is not None else 20.0 + s for s in seeds],
        "exchequer_cost_bn": [cost + 0.1 * s for s in seeds]})


def test_complete_variant_requires_all_seeds_build_and_fingerprint():
    d = pd.concat([variant_rows("cap60"), variant_rows("closed", seeds=range(6))])
    assert len(pn.complete_variant(d, "cap60", "B", "F")) == 20
    with pytest.raises(pn.Incomplete):
        pn.complete_variant(d, "closed", "B", "F")
    with pytest.raises(pn.Incomplete):
        pn.complete_variant(d, "cap60", "other", "F")
    with pytest.raises(pn.Incomplete):
        pn.complete_variant(d, "cap60", "B", "other")


def test_public_paybill_agrees_across_variants():
    a, b = variant_rows("cap10"), variant_rows("cap20")
    out = pn.public_paybill({"cap10": a, "cap20": b})
    assert out["mean"] == pytest.approx(20.0 + 9.5) and out["draws"] == 20
    b.loc[3, "public_paybill_displaced_bn"] += 1
    with pytest.raises(ValueError):
        pn.public_paybill({"cap10": a, "cap20": b})
    with pytest.raises(pn.Incomplete):
        pn.public_paybill({})


def test_capital_breakeven_interpolates_and_requires_a_sign_change():
    central = pd.DataFrame({"seed": range(4), "exchequer_cost_bn": [8.0, 6.0, 7.0, 7.0]})  # mean 7
    cap60 = pd.DataFrame({"seed": range(4), "exchequer_cost_bn": [-1.0, -2.0, -1.0, -2.0]})  # mean -1.5
    be = pn.capital_breakeven(central, cap60)
    x0 = 100 * (1.405 / 1.005 - 1)
    assert be["uplift_pct"] == pytest.approx(x0 + (60 - x0) * 7 / 8.5)
    assert be["pp"] == pytest.approx(1.005 * be["uplift_pct"] / 100)
    assert pn.CENTRAL_UPLIFT_PCT == pytest.approx(39.80, abs=0.01)
    with pytest.raises(ValueError):
        pn.capital_breakeven(central, central)
    with pytest.raises(ValueError):
        pn.capital_breakeven(central, cap60.assign(seed=[0, 1, 2, 9]))


def obr_draws():
    return pd.DataFrame({
        "income_tax_change_bn": [-5.0, -5.4], "ni_employee_change_bn": [-1.0, -1.0],
        "employer_ni_change_bn": [-3.0, -3.2], "uc_change_bn": [0.6, 0.8],
        "childcare_change_bn": [-0.2, -0.2], "statutory_pay_change_bn": [-0.1, 0.0],
        "benefits_change_bn": [0.35, 0.65], "household_basis_cost_bn": [6.35, 7.05],
        "displaced_weighted_m": [0.5, 0.5], "public_paybill_displaced_bn": [5.0, 6.0],
        "wage_bill_lost_bn": [23.0, 24.0]}).assign(
        exchequer_cost_bn=lambda d: -d.income_tax_change_bn - d.ni_employee_change_bn
        - d.employer_ni_change_bn + d.benefits_change_bn)


def test_obr_lines_split_welfare_from_childcare_and_statutory_pay():
    a = pn.obr_lines(obr_draws())
    assert a["receipts"] == pytest.approx(5.2 + 1.0 + 3.1)
    assert a["uc"] == pytest.approx(0.7)
    assert a["other_benefits"] == pytest.approx(0.5 - 0.7 + 0.2 + 0.05)
    assert a["welfare"] == pytest.approx(0.5 + 0.2 + 0.05)  # benefits less childcare and statutory pay
    assert a["total"] == pytest.approx(a["cost"]) and abs(a["residual"]) < 1e-9
    assert a["cost_per_displaced_gbp"] == pytest.approx(1e3 * np.mean([9.35 / 0.5, 10.25 / 0.5]))
    bad = obr_draws().assign(exchequer_cost_bn=lambda d: d.exchequer_cost_bn + 1)
    with pytest.raises(ValueError):
        pn.obr_lines(bad)


def test_decile_stabilisation_from_arrays():
    base = {"weight": np.array([1.0, 2.0, 1.0]), "decile": np.array([1, 1, 2]),
            "hh_market": np.array([100.0, 50.0, 80.0]), "hh_benefits": np.zeros(3), "hh_uc": np.zeros(3),
            "hh_it": np.array([10.0, 5.0, 8.0]), "hh_ni": np.array([2.0, 1.0, 1.0]),
            "hh_net": np.array([88.0, 44.0, 71.0])}
    shocked = dict(base, hh_market=np.array([0.0, 50.0, 80.0]), hh_benefits=np.array([20.0, 0, 0]),
                   hh_uc=np.array([20.0, 0, 0]), hh_it=np.array([0.0, 5.0, 8.0]), hh_ni=np.array([0.0, 1.0, 1.0]),
                   hh_net=np.array([20.0, 44.0, 71.0]))
    f = pn.flows_from_arrays(base, shocked)
    assert f.loc[1, "market_change"] == pytest.approx(-100.0)
    assert f.loc[1, "tax_change"] == pytest.approx(-12.0)
    out = pn.decile_stabilisation([f * 1e9, f * 1e9]).set_index("decile")
    assert out.loc[1, "tax_benefit_share"] == pytest.approx(32 / 100)
    assert out.loc[1, "stabilisation_net"] == pytest.approx(1 - 68 / 100)
    assert out.loc["all", "market_change"] == pytest.approx(-100.0)
    assert set(out["draws"]) == {2}


def test_pe_parameter_reads_the_rate_in_force():
    assert pn.pe_parameter("universal_credit/means_test/reduction_rate.yaml") == pytest.approx(0.55)
    assert pn.pe_parameter("universal_credit/means_test/capital/limit.yaml") == 16_000
    single = pn.pe_parameter("universal_credit/standard_allowance/amount.yaml", key="SINGLE_OLD")
    assert 350 < single < 500
