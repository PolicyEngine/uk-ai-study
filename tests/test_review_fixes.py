"""Review of PR #18: quota rates, displaced-worker transition, decile-1
ranking sensitivity, cost decomposition, SPI scaling and the macro guard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from test_shocks import make_persons  # noqa: E402
from uk_ai_study.exposure import exposure_for_major_group  # noqa: E402
from uk_ai_study.runner import (  # noqa: E402
    COST_COMPONENTS,
    cost_decomposition,
    decile_mean_change,
    decile_rankings_from_inputs,
)
from uk_ai_study.shocks import (  # noqa: E402
    PRESETS,
    TRANSITION_ZEROED_VARIABLES,
    draw_displaced,
    group_quotas,
    realised_displacement_rate,
)


# --- fix 6: prescribed quota rates ----------------------------------------
def test_group_quotas_sum_to_the_aggregate_rate():
    persons = make_persons()
    q = group_quotas(persons, 0.07)
    total_emp = sum(v["employment"] for v in q.values())
    assert sum(v["quota"] for v in q.values()) == pytest.approx(0.07 * total_emp)
    # the least-exposed group takes no quota (min-shift normalisation)
    assert min(v["quota_rate"] for v in q.values()) == 0.0


def test_realised_rate_varies_around_the_prescribed_rate():
    persons = make_persons(n=4000)
    rates = [realised_displacement_rate(persons, draw_displaced(persons, PRESETS["central"], seed=s))
             for s in range(40)]
    assert np.mean(rates) == pytest.approx(0.07, abs=0.003)
    assert np.std(rates) > 0  # one draw is not the quota


# --- fix 8: transition and dead code --------------------------------------
def test_displaced_workers_stop_all_pension_contributions():
    for var in ("personal_pension_contributions", "employee_pension_contributions",
                "pension_contributions_via_salary_sacrifice"):
        assert var in TRANSITION_ZEROED_VARIABLES


def test_exposure_lookup_accepts_only_major_groups_one_to_nine():
    out = exposure_for_major_group(np.array([2, 2000, 0, np.nan]))
    assert np.isfinite(out[0]) and np.isnan(out[1:]).all()


# --- fix 2: decile rankings ------------------------------------------------
def _ranking_case():
    # ten one-person households; household 0 has £50k earnings but £45k of
    # personal pension contributions, so HBAI income puts it at the bottom
    n = 10
    hbai = np.linspace(10_000, 55_000, n)
    hbai[0] = 5_000.0
    ppc = np.zeros(n)
    ppc[0] = 45_000.0
    earnings = np.full(n, 20_000.0)
    earnings[0] = 40_000.0
    inputs = {
        "hbai": hbai, "equiv_hbai": hbai, "equivalisation": np.ones(n),
        "pension_contributions": ppc, "personal_pension_contributions": ppc,
        "earnings": earnings,
    }
    return decile_rankings_from_inputs(inputs, lambda v: v, np.ones(n))


def test_pre_pension_ranking_moves_the_large_contributor_up():
    r = _ranking_case()
    assert r["hbai"][0][0] == 1
    assert r["pre_pension"][0][0] == 8  # £50k before contributions


def test_exclusion_check_drops_only_large_contributors_above_earnings():
    r = _ranking_case()
    deciles, include = r["hbai_excl_large_ppc"]
    assert not include[0] and include[1:].all()
    change = decile_mean_change(np.arange(10.0), np.ones(10), deciles, include)
    assert np.isnan(change[1])  # decile 1 held only the excluded household


# --- fix 4: cost decomposition --------------------------------------------
def _totals(**kw):
    base = dict(gov_balance=0.0, income_tax=0.0, dividend_tax=0.0, savings_tax=0.0,
                other_income_tax=0.0, employee_ni=0.0, employer_ni=0.0,
                universal_credit=0.0, childcare=0.0, household_benefits=0.0,
                narrow_cash_balance=0.0)
    base.update(kw)
    return base


def test_cost_components_sum_to_the_gov_balance_cost():
    b = _totals(gov_balance=100e9, income_tax=50e9, dividend_tax=10e9, savings_tax=5e9,
                other_income_tax=35e9, employee_ni=20e9, employer_ni=30e9,
                universal_credit=40e9, childcare=5e9, narrow_cash_balance=30e9)
    s = _totals(gov_balance=92e9, income_tax=53e9, dividend_tax=14e9, savings_tax=6e9,
                other_income_tax=33e9, employee_ni=18e9, employer_ni=26e9,
                universal_credit=42e9, childcare=4.5e9, narrow_cash_balance=29e9)
    out = cost_decomposition(b, s)
    assert out["total"] == pytest.approx(8.0)
    assert sum(out[k] for k in COST_COMPONENTS) + out["other"] == pytest.approx(8.0)
    assert out["dividend_tax"] == pytest.approx(-4.0)  # extra dividend tax is a saving
    assert out["employer_ni"] == pytest.approx(4.0)
    assert out["excluding_employer_ni"] == pytest.approx(4.0)
    assert out["narrow_cash"] == pytest.approx(1.0)
    assert out["gov_balance_minus_narrow_cash"] == pytest.approx(7.0)


# --- fix 5: SPI scaling ----------------------------------------------------
def test_spi_scaling_reaches_spi_totals_on_taxpayers_only():
    import sensitivity_dividend_coverage as dc

    rng = np.random.default_rng(0)
    n = 5000
    ti = np.concatenate([rng.uniform(0, 12_000, 500), rng.lognormal(10.8, 1.2, n - 500)])
    ti[:3] = [2e6, 3e6, 5e6]  # someone in the top band
    w = rng.uniform(100, 1000, n)
    div = rng.lognormal(7, 1.5, n)
    for method in ("uniform", "banded"):
        f, table = dc.scaling_factors(div, w, ti, "dividend_income", method)
        taxpayer = ti >= dc.SPI_THRESHOLD
        assert (div * f * w)[taxpayer].sum() == pytest.approx(dc.SPI_DIVIDENDS_2026, rel=1e-9)
        assert (f[~taxpayer] == 1.0).all()
        assert len(table) == len(dc.SPI_DIVIDEND_BANDS_2026)
    # banded: each band hits its own SPI total
    f, table = dc.scaling_factors(div, w, ti, "dividend_income", "banded")
    idx = dc.band_index(ti, list(dc.SPI_DIVIDEND_BANDS_2026))
    top = len(table) - 1
    assert (div * f * w)[idx == top].sum() == pytest.approx(table[top]["spi_bn"] * 1e9)


def test_spi_band_rows_sum_to_the_file_totals():
    import sensitivity_dividend_coverage as dc

    assert sum(dc.SPI_DIVIDEND_BANDS_2026.values()) == pytest.approx(dc.SPI_DIVIDENDS_2026, rel=1e-9)
    assert sum(dc.SPI_SAVINGS_BANDS_2026.values()) == pytest.approx(dc.SPI_SAVINGS_2026, rel=1e-8)


# --- fix 7: emitter guard and signed twins ---------------------------------
def test_emitter_marks_a_partial_grid_pending(tmp_path, monkeypatch):
    import emit_tex_values as etv

    (tmp_path / "appendix").mkdir()
    rows = [{"unemployment_pct": u, "wage_pct": w, "capital_shock": c, "decile1_change_pct": 0.1}
            for u in range(4) for w in range(6) for c in (True, False)][:40]
    pd.DataFrame(rows).to_csv(tmp_path / "appendix/grid_deciles_capital.csv", index=False)
    monkeypatch.setattr(etv, "R", tmp_path)
    monkeypatch.setattr(etv, "_checked", {})
    with pytest.raises(etv.Pending, match="40 of 132"):
        etv.csv("appendix/grid_deciles_capital.csv")


def test_emitter_placeholders_and_twins(monkeypatch):
    import emit_tex_values as etv

    monkeypatch.setattr(etv, "lines", [])
    monkeypatch.setattr(etv, "missing", [])
    monkeypatch.setattr(etv, "emitted", set())

    def pending():
        raise etv.Pending("incomplete")
    etv.emit("genPresetRows", pending, "{}", "x")
    etv.emit("genAppGridHarshDOnePct", lambda: 1.25, "{:.1f}", "x")  # a loss of 1.25
    etv.emit("genGridGiniMinPp", lambda: -0.31, "{:.2f}", "x")
    text = "\n".join(etv.lines)
    assert "\\newcommand{\\genPresetRows}{\\GENPENDINGROW}" in text
    assert "\\newcommand{\\genAppGridHarshDOneSignedPct}{-1.2}" in text
    assert "\\newcommand{\\genGridGiniMinSignedPp}{-0.31}" in text
