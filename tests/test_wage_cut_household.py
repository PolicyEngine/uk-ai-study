"""Wage-cut household run (analysis/wage_cut_household.py) on fake data.
No simulation is run."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import wage_cut_household as wc  # noqa: E402
from uk_ai_study.shocks import PRESETS, WAGE_MARGIN_PRESETS, apply_wage_margin_shock, draw_displaced  # noqa: E402


def fake_persons(n=400, seed=1):
    rng = np.random.default_rng(seed)
    grp = rng.integers(1, 10, n).astype(float)
    exposure = {g: e for g, e in zip(range(1, 10), np.linspace(-1.0, 1.5, 9))}
    theta = {g: t for g, t in zip(range(1, 10), np.linspace(0.3, 0.8, 9))}
    emp = np.where(rng.random(n) < 0.8, rng.lognormal(10, 0.5, n), 0.0)
    return pd.DataFrame({
        "weight": rng.uniform(50, 150, n),
        "employment_income": emp,
        "soc_major_group": grp,
        "exposure": [exposure[int(g)] for g in grp],
        "complementarity": [theta[int(g)] for g in grp],
        "age": rng.integers(16, 66, n),
        "savings_interest_income": rng.uniform(0, 500, n),
        "dividend_income": np.where(rng.random(n) < 0.1, rng.uniform(0, 5000, n), 0.0),
    })


def test_table3_variant_is_the_paper_definition():
    assert wc.VARIANTS["table3"] == WAGE_MARGIN_PRESETS["wage_margin_central"]
    assert wc.VARIANTS["table3"].gradient == "caioe"
    assert wc.VARIANTS["table3"].aggregate_earnings_loss_share is None  # paired to the central draw
    cut = wc.VARIANTS["cut_only"]
    assert (cut.wage_uplift, cut.capital_return_increase, cut.gradient) == (0.0, 0.0, "caioe")
    assert cut.aggregate_earnings_loss_share is None


def test_gross_cut_equals_paired_central_draw_for_each_seed():
    p = fake_persons()
    e, w = p.employment_income.to_numpy(), p.weight.to_numpy()
    for seed in (0, 3):
        disp = draw_displaced(p, PRESETS["central"], seed=seed)
        lost = float((e * w)[disp].sum())
        for v in wc.VARIANTS.values():
            t = apply_wage_margin_shock(p, v, seed=seed)
            assert t.attrs["gross_cut_realised"] == pytest.approx(lost, rel=1e-9)
            assert not t["displaced"].any()


def test_cut_only_has_no_uplift_or_capital_shock():
    p = fake_persons()
    t = apply_wage_margin_shock(p, wc.VARIANTS["cut_only"], seed=0)
    assert (t.employment_income <= p.employment_income + 1e-9).all()
    assert t.dividend_income.equals(p.dividend_income)
    t3 = apply_wage_margin_shock(p, wc.VARIANTS["table3"], seed=0)
    assert (t3.dividend_income >= p.dividend_income).all() and t3.dividend_income.sum() > p.dividend_income.sum()


def test_stabilisation_coefficients():
    # market income falls 100, disposable 70: 30 per cent absorbed
    assert wc.stabilisation(-100.0, -70.0) == pytest.approx(0.30)
    # benefits +10, taxes -20 -> 30 of the 100 absorbed by tax and benefits
    assert wc.tax_benefit_stabilisation(-100.0, 10.0, -20.0) == pytest.approx(0.30)
    assert np.isnan(wc.stabilisation(0.0, 0.0))


def test_household_flows_by_decile_and_family():
    dec = np.array([1, 1, 2, 2])
    fam = np.array(["A", "B", "A", "A"])
    w = np.array([1.0, 2.0, 1.0, 3.0])
    d = {"market": np.array([-100.0, -50.0, -10.0, 0.0]), "net": np.array([-60.0, -40.0, -8.0, 0.0]),
         "income_tax": np.array([-20.0, -5.0, -2.0, 0.0]), "ni_employee": np.array([-8.0, -2.0, 0.0, 0.0]),
         "uc": np.array([10.0, 3.0, 0.0, 0.0]), "benefits": np.array([12.0, 3.0, 0.0, 0.0])}
    g = wc.household_flows(dec, w, d).set_index("decile")
    assert g.loc[1, "market_change"] == pytest.approx(-200.0)
    assert g.loc[1, "net_income_change"] == pytest.approx(-140.0)
    assert g.loc[1, "other_benefit_change"] == pytest.approx(2.0)
    assert g.loc[1, "tax_change"] == pytest.approx(-30.0 - 12.0)
    assert g.loc[1, "stabilisation_net"] == pytest.approx(1 - 140 / 200)
    assert g.loc[1, "stabilisation_tax_benefit"] == pytest.approx((18.0 + 42.0) / 200)
    assert g.loc[2, "weight"] == pytest.approx(4.0)
    gf = wc.household_flows([dec, fam], w, d)
    assert set(map(tuple, gf[["decile", "family_type"]].to_numpy())) == {(1, "A"), (1, "B"), (2, "A")}
    assert gf.market_change.sum() == pytest.approx(g.market_change.sum())


def test_summary_pools_flows_per_variant():
    rows = []
    for v, k in (("table3", 1.0), ("cut_only", 2.0)):
        for s in range(3):
            rows.append({"variant": v, "seed": s, "market_income_change_bn": -10.0 * k,
                         "net_income_change_bn": -6.0 * k, "benefits_change_bn": 1.0 * k,
                         "income_tax_change_bn": -2.0 * k, "ni_employee_change_bn": -1.0 * k,
                         "exchequer_cost_bn": 5.0 + s})
    out = wc.summarise(pd.DataFrame(rows))
    assert list(out) == ["table3", "cut_only"]
    assert out["table3"]["pooled"]["stabilisation_net"] == pytest.approx(0.4)
    assert out["cut_only"]["pooled"]["stabilisation_tax_benefit"] == pytest.approx(0.4)
    assert out["table3"]["exchequer_cost_bn"]["mean"] == pytest.approx(6.0)
    assert out["table3"]["pooled"]["draws"] == 3


def test_variant_parsing_keeps_canonical_order():
    assert wc.parse_variants(["x"]) == ["table3", "cut_only"]
    assert wc.parse_variants(["x", "--variants", "cut_only,table3"]) == ["table3", "cut_only"]
    assert wc.parse_variants(["x", "--variants", "table3"]) == ["table3"]
    with pytest.raises(SystemExit):
        wc.parse_variants(["x", "--variants", "bogus"])


def test_fingerprint_covers_next_runs(monkeypatch, tmp_path):
    a = wc.fingerprint()
    assert a.startswith("c") and len(a) == 17
    fake_root = tmp_path
    (fake_root / "analysis").mkdir()
    (fake_root / "analysis" / "next_runs.py").write_text("# changed\n")
    monkeypatch.setattr(wc, "ROOT", fake_root)
    assert wc.fingerprint() != a


def test_outputs_stay_outside_runs_15_16():
    assert wc.OUT.name == "wage_cut"
    assert wc.ARRAYS_DIR != wc.nr.ARRAYS_DIR
    assert wc.SEEDS == tuple(range(20))
