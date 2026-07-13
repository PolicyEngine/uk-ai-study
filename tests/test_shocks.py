"""Unit tests for the shock mechanics (uk-ai-study#1, findings 4-6)."""

import numpy as np
import pandas as pd
import pytest

from uk_ai_study.shocks import ShockScenario, apply_shocks, draw_displaced


def make_persons(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    group = rng.integers(1, 10, n) * 1000.0
    exposure_by_group = {g * 1000.0: x for g, x in zip(range(1, 10), np.linspace(-0.7, 0.7, 9))}
    theta_by_group = {g * 1000.0: x for g, x in zip(range(1, 10), np.linspace(0.3, 0.9, 9))}
    return pd.DataFrame(
        {
            "age": rng.integers(18, 64, n),
            "employment_income": rng.lognormal(10, 0.6, n),
            "savings_interest_income": rng.lognormal(4, 1.0, n),
            "dividend_income": rng.lognormal(3, 1.0, n),
            "weight": rng.uniform(100, 2000, n),
            "soc_major_group": group,
            "exposure": np.vectorize(exposure_by_group.get)(group),
            "complementarity": np.vectorize(theta_by_group.get)(group),
        }
    )


def test_wage_conservation_per_seed():
    """JR16 eq 3.5: with employment-weighted theta_bar, the employment-
    weighted mean percentage wage change across workers equals the assumed
    wage uplift, per seed (finding 5)."""
    persons = make_persons()
    scenario = ShockScenario("t", displacement_rate=0.0, wage_uplift=0.026)
    for seed in range(5):
        shocked = apply_shocks(persons, scenario, seed=seed)
        base = persons["employment_income"].to_numpy()
        new = shocked["employment_income"].to_numpy()
        w = persons["weight"].to_numpy()
        workers = base > 0
        pct = (new[workers] - base[workers]) / base[workers]
        mean_pct = np.average(pct, weights=w[workers])
        assert mean_pct == pytest.approx(scenario.wage_uplift, rel=1e-9)


def test_quota_in_expectation():
    """Expected displaced weight equals displacement_rate x employee weight
    (eq 3.4), averaged over draws."""
    persons = make_persons()
    scenario = ShockScenario("t", displacement_rate=0.07, wage_uplift=0.0)
    w = persons["weight"].to_numpy()
    total = w[persons["employment_income"] > 0].sum()
    displaced_w = np.mean(
        [w[draw_displaced(persons, scenario, seed=s)].sum() for s in range(200)]
    )
    assert displaced_w == pytest.approx(0.07 * total, rel=0.02)


def test_equal_inclusion_regardless_of_weight():
    """Within a group, a person's displacement probability must not depend on
    their grossing weight (finding 6). Two persons, weights 1 and 9, quota 5:
    both should be included ~50% of the time."""
    persons = pd.DataFrame(
        {
            "age": [40, 40],
            "employment_income": [30000.0, 30000.0],
            "savings_interest_income": [0.0, 0.0],
            "dividend_income": [0.0, 0.0],
            "weight": [1.0, 9.0],
            "soc_major_group": [1000.0, 1000.0],
            "exposure": [0.5, 0.5],
            "complementarity": [0.6, 0.6],
        }
    )
    scenario = ShockScenario("t", displacement_rate=0.5, wage_uplift=0.0)
    hits = np.zeros(2)
    n = 4000
    for s in range(n):
        hits += draw_displaced(persons, scenario, seed=s)
    assert hits[0] / n == pytest.approx(0.5, abs=0.03)
    assert hits[1] / n == pytest.approx(0.5, abs=0.03)


def test_unmatched_employees_in_displacement_universe():
    """Employees without a SOC code are drawable (finding 7)."""
    persons = make_persons()
    persons.loc[persons.index[:400], "soc_major_group"] = np.nan
    scenario = ShockScenario("t", displacement_rate=0.30, wage_uplift=0.0)
    unmatched_hit = 0
    for s in range(20):
        d = draw_displaced(persons, scenario, seed=s)
        unmatched_hit += d[~np.isfinite(persons["soc_major_group"].to_numpy())].sum()
    assert unmatched_hit > 0


def test_displaced_earn_zero():
    persons = make_persons()
    scenario = ShockScenario("t", displacement_rate=0.10, wage_uplift=0.026)
    shocked = apply_shocks(persons, scenario, seed=0)
    displaced = shocked["displaced"].to_numpy()
    assert (shocked["employment_income"].to_numpy()[displaced] == 0).all()
