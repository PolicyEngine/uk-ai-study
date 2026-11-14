"""Fiscal Studies revision post-processing (analysis/revision_accounting.py),
checked on fake data with hand-computed answers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import revision_accounting as ra  # noqa: E402


def survivor_grid():
    # bill = 10 / 0.1 = 100; rows give net employee-earnings change in per cent
    return pd.DataFrame({
        "survivor_wage_effect_pct": [-5.0, -2.6, 0.0, 2.6, 5.0],
        "net_employment_income_change_pct": [-15.0, -12.5, -10.0, -7.5, -5.0],
        "exchequer_cost_bn": [12.0, 8.0, 5.0, 2.0, -1.0],
    })


def test_zero_crossing_interpolates_and_refuses_to_extrapolate():
    assert ra.zero_crossing([0, 1, 2], [2.0, 1.0, -1.0]) == pytest.approx(1.5)
    assert ra.zero_crossing([0, 1], [0.0, 3.0]) == 0.0
    with pytest.raises(ValueError):
        ra.zero_crossing([0, 1], [1.0, 2.0])


def test_breakeven_survivor_wage_between_grid_rows():
    # cost 2 at 2.6 and -1 at 5: zero at 2.6 + 2.4 * 2/3 = 4.2
    assert ra.breakeven_survivor_wage(survivor_grid().sample(frac=1, random_state=1)) == pytest.approx(4.2)


def test_breakeven_capital_is_linear_in_the_return_shock():
    out = ra.breakeven_capital(central_cost_bn=4.0, capital_channel_cost_bn=-8.0)
    assert out["cost_without_capital_bn"] == pytest.approx(12.0)
    assert out["pp"] == pytest.approx(0.6)  # 0.4 * 12 / 8
    assert out["uplift_pct"] == pytest.approx(100 * 0.6 / 1.005)
    with pytest.raises(ValueError):
        ra.breakeven_capital(4.0, 1.0)


def test_breakeven_duration_reads_retention_as_months():
    out = ra.breakeven_duration(full_year_cost_bn=6.0, half_retention_cost_bn=-6.0)
    assert out["retention"] == pytest.approx(0.25)
    assert out["months_out_of_work"] == pytest.approx(9.0)
    with pytest.raises(ValueError):
        ra.breakeven_duration(-1.0, -6.0)


def test_survivor_gain_uses_the_baseline_bill():
    # (-7.5 - -10) per cent of a 100bn bill
    assert ra.survivor_gain_bn(survivor_grid(), wage_bill_lost_bn=10.0, gross_share=0.1) == pytest.approx(2.5)


def factor(**kw):
    base = dict(wage_bill_lost_bn=100.0, survivor_gain_bn=20.0, employer_ni_net_bn=5.0,
                dividends_bn=50.0, interest_bn=10.0, channel_disp_bn=40.0, channel_wage_bn=-10.0,
                channel_capital_bn=-15.0, central_cost_bn=16.0, dividend_tax_on_uplift_bn=7.0,
                phi=1.0, payout=0.5, uk_household_share=1.0)
    base.update(kw)
    return ra.factor_account(**base)


def test_factor_account_flows_add_up():
    a = factor()
    assert a["net_labour_income_lost_bn"] == pytest.approx(80.0)
    assert a["profits_bn"] == pytest.approx(80.0)
    assert a["corporation_tax_bn"] == pytest.approx(20.0)
    assert a["post_tax_profits_bn"] == pytest.approx(60.0)
    assert a["household_dividends_bn"] == pytest.approx(30.0)
    rate = 7.0 / (ra.CAPITAL_SCALE * 50.0)
    assert a["dividend_tax_rate"] == pytest.approx(rate)
    assert a["interaction_bn"] == pytest.approx(1.0)  # 16 - (40 - 10 - 15)
    # closed: displacement + wage + interaction - CT - dividend tax
    assert a["closed_cost_bn"] == pytest.approx(40 - 10 + 1 - 20 - rate * 30)
    assert a["capital_uplift_assumed_bn"] == pytest.approx(ra.CAPITAL_SCALE * 60.0)


def test_factor_account_monotone_in_phi_and_ownership():
    assert factor(phi=0.0)["closed_cost_bn"] == pytest.approx(31.0)  # no profits: no offset
    assert factor(phi=1.0)["closed_cost_bn"] < factor(phi=0.5)["closed_cost_bn"]
    assert factor(uk_household_share=1.0)["closed_cost_bn"] < factor(uk_household_share=0.5)["closed_cost_bn"]


def test_absorption_rates():
    out = ra.absorption_rates(wage_bill_lost_bn=10.0, disp_gov_balance_bn=4.0, disp_narrow_cash_bn=3.0,
                              survivor_grid=survivor_grid(), gross_share=0.1)
    assert out["job_loss_gov_balance"] == pytest.approx(0.4)
    assert out["job_loss_household"] == pytest.approx(0.3)
    # 0 -> -2.6 removes 2.5 per cent of 100 = 2.5bn at a cost of 3bn
    assert out["wage_cut_earnings_bn"] == pytest.approx(2.5)
    assert out["wage_cut_gov_balance"] == pytest.approx(1.2)


def test_cost_per_displaced_is_a_per_draw_ratio():
    d = pd.DataFrame({"run": ["channel_displacement"] * 2 + ["central"],
                      "exchequer_cost_bn": [20.0, 30.0, 99.0], "displaced_weighted_m": [1.0, 2.0, 1.0]})
    out = ra.cost_per_displaced(d)
    # ratios 20,000 and 15,000: mean 17,500 (not 50/3 bn per m)
    assert out["mean_gbp"] == pytest.approx(17_500)
    assert out["draws"] == 2


def test_family_cost_per_worker_adds_back_other_channels():
    d = pd.DataFrame({"family": ["a", "a", "b"], "exchequer_cost_bn": [5.0, 15.0, 10.0],
                      "displaced_weighted_m": [1.0, 1.0, 2.0]})
    out = ra.family_displacement_cost_per_worker(d, other_channels_bn=-5.0)
    assert out["a"] == pytest.approx(15_000)  # (10 + 20)/2 thousand
    assert out["b"] == pytest.approx(7_500)


def test_count_adjustment_removes_a_pure_count_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 0.1, 50)
    d = pd.DataFrame({"family": "exposure", "displaced_weighted_m": x, "exchequer_cost_bn": 3 + 11 * x})
    slope, mu = ra.count_slope(d, "exchequer_cost_bn")
    assert slope == pytest.approx(11.0)
    # a draw that displaced 0.2m more than average is shifted back by 2.2bn
    assert ra.count_adjusted(30.0, mu + 0.2, slope, mu) == pytest.approx(27.8)


def test_mc_standard_error():
    assert ra.mc_standard_error(1.8, 50) == pytest.approx(1.8 / np.sqrt(50))


def test_emitter_writes_revision_macros_from_results():
    """The real results produce every revision macro (no simulation needed)."""
    import emit_tex_values as etv

    text = etv.OUT.read_text() if etv.OUT.exists() else ""
    for name in ("genBeWagePct", "genBeCapitalPp", "genBeDurMonths", "genFactorRows",
                 "genAbsJobGovPct", "genCostPerDisplacedGbp", "genIndexAdjRows", "genPolicyMcRows"):
        assert f"\\newcommand{{\\{name}}}" in text
        assert f"\\newcommand{{\\{name}}}{{\\GENPENDING" not in text
