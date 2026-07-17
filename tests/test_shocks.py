"""Unit tests for the shock mechanics (uk-ai-study#1, findings 4-6)."""

import numpy as np
import pandas as pd
import pytest

from uk_ai_study.shocks import (
    ShockScenario,
    WAGE_MARGIN_PRESETS,
    WageMarginScenario,
    apply_shocks,
    apply_wage_margin_shock,
    draw_displaced,
)


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


def test_survivor_composition_wage_drift():
    """REVISION_PLAN item 2 (code audit #6): with displacement > 0, eq 3.5's
    theta_bar is the employment-weighted mean over ALL baseline workers, but
    the uplift is only paid to survivors. When theta correlates with wages
    and displacement is tilted toward low-theta (high-exposure) groups, the
    survivors' earnings-weighted mean theta exceeds theta_bar, so the
    realised aggregate uplift to the survivor wage bill drifts ABOVE the
    assumed rate. Quantify the drift and bound it."""
    n = 3000
    rng = np.random.default_rng(7)
    group = rng.integers(1, 10, n) * 1000.0
    g = group / 1000.0
    # theta rises with group; exposure falls with group (substitution vs
    # complementarity); wages rise with group -> theta correlates with wages
    theta_by_group = {gg: t for gg, t in zip(range(1, 10), np.linspace(0.3, 0.9, 9))}
    exposure_by_group = {gg: x for gg, x in zip(range(1, 10), np.linspace(0.9, 0.1, 9))}
    persons = pd.DataFrame(
        {
            "age": rng.integers(18, 64, n),
            "employment_income": rng.lognormal(9.5 + 0.15 * g, 0.3),
            "savings_interest_income": np.zeros(n),
            "dividend_income": np.zeros(n),
            "weight": rng.uniform(100, 2000, n),
            "soc_major_group": group,
            "exposure": np.vectorize(exposure_by_group.get)(g),
            "complementarity": np.vectorize(theta_by_group.get)(g),
        }
    )
    scenario = ShockScenario("t", displacement_rate=0.07, wage_uplift=0.026)
    drifts = []
    for seed in range(10):
        shocked = apply_shocks(persons, scenario, seed=seed)
        base = persons["employment_income"].to_numpy()
        new = shocked["employment_income"].to_numpy()
        w = persons["weight"].to_numpy()
        survivors = (base > 0) & ~shocked["displaced"].to_numpy()
        realised = float(
            ((new - base) * w)[survivors].sum() / (base * w)[survivors].sum()
        )
        drifts.append(realised / scenario.wage_uplift - 1.0)
        # the aggregate survivor uplift rate exceeds the assumed 2.6%...
        assert realised > scenario.wage_uplift
        # ...but by less than 50% relative
        assert realised < 1.5 * scenario.wage_uplift
    print(
        f"\nsurvivor-composition wage drift over 10 seeds: "
        f"mean {100 * np.mean(drifts):+.2f}% relative "
        f"(min {100 * min(drifts):+.2f}%, max {100 * max(drifts):+.2f}%) "
        f"vs assumed uplift {scenario.wage_uplift:.3f}"
    )


def test_displaced_earn_zero():
    persons = make_persons()
    scenario = ShockScenario("t", displacement_rate=0.10, wage_uplift=0.026)
    shocked = apply_shocks(persons, scenario, seed=0)
    displaced = shocked["displaced"].to_numpy()
    assert (shocked["employment_income"].to_numpy()[displaced] == 0).all()


# --- wage-margin family -----------------------------------------------------


def test_wage_margin_aggregate_earnings_equivalence():
    """With no uplift, the weighted aggregate fall in employee earnings
    equals aggregate_earnings_loss_share x baseline aggregate earnings."""
    persons = make_persons()
    scenario = WageMarginScenario("t", aggregate_earnings_loss_share=0.07, wage_uplift=0.0)
    shocked = apply_wage_margin_shock(persons, scenario)
    base = persons["employment_income"].to_numpy()
    new = shocked["employment_income"].to_numpy()
    w = persons["weight"].to_numpy()
    workers = base > 0
    loss = ((base - new) * w)[workers].sum()
    assert loss == pytest.approx(0.07 * (base * w)[workers].sum(), rel=1e-9)


def test_wage_margin_no_negative_incomes_and_no_job_loss():
    persons = make_persons()
    for scenario in WAGE_MARGIN_PRESETS.values():
        shocked = apply_wage_margin_shock(
            persons,
            scenario,
            # supply explicit per-group weights for the pss preset so the
            # test does not depend on the packaged csv
            gradient_weights={g: float(g) for g in range(1, 10)}
            if scenario.gradient == "pss"
            else None,
        )
        new = shocked["employment_income"].to_numpy()
        base = persons["employment_income"].to_numpy()
        assert (new >= 0).all()
        # nobody loses their job: all baseline workers still earn > 0, and
        # the displaced mask (which drives hours/pension zeroing and the
        # UNEMPLOYED transition downstream) is all False
        assert (new[base > 0] > 0).all()
        assert not shocked["displaced"].to_numpy().any()
        # non-earnings person inputs untouched (hours/pension are only ever
        # modified downstream for displaced == True)
        assert (shocked["age"].to_numpy() == persons["age"].to_numpy()).all()


def test_wage_margin_gradient_monotonicity():
    """Higher-exposure major groups take larger percentage cuts."""
    persons = make_persons()
    scenario = WageMarginScenario("t", wage_uplift=0.0)
    shocked = apply_wage_margin_shock(persons, scenario)
    base = persons["employment_income"].to_numpy()
    new = shocked["employment_income"].to_numpy()
    workers = base > 0
    pct_cut = pd.Series((base - new)[workers] / base[workers])
    group = persons["soc_major_group"][workers].reset_index(drop=True)
    by_group = pct_cut.groupby(group).mean()
    exposure = persons.groupby("soc_major_group")["exposure"].first()
    ordered = by_group.reindex(exposure.sort_values().index)
    assert (ordered.diff().dropna() >= -1e-12).all()
    assert ordered.iloc[-1] > ordered.iloc[0]


def test_wage_margin_composes_with_uplift():
    """cut + uplift = (cut alone) + (uplift alone), person by person, and the
    aggregate net change equals uplift-implied gain minus the loss share."""
    persons = make_persons()
    both = apply_wage_margin_shock(persons, WageMarginScenario("t", wage_uplift=0.026))
    cut_only = apply_wage_margin_shock(persons, WageMarginScenario("t", wage_uplift=0.0))
    uplift_only = apply_wage_margin_shock(
        persons, WageMarginScenario("t", aggregate_earnings_loss_share=0.0, wage_uplift=0.026)
    )
    base = persons["employment_income"].to_numpy()
    np.testing.assert_allclose(
        both["employment_income"].to_numpy() - base,
        (cut_only["employment_income"].to_numpy() - base)
        + (uplift_only["employment_income"].to_numpy() - base),
        rtol=1e-9,
    )
    # eq 3.5 conservation still holds for the uplift leg: weighted mean %
    # change is exactly +2.6% with no displacement
    w = persons["weight"].to_numpy()
    workers = base > 0
    pct = (uplift_only["employment_income"].to_numpy() - base)[workers] / base[workers]
    assert np.average(pct, weights=w[workers]) == pytest.approx(0.026, rel=1e-9)


def test_wage_margin_pss_missing_file_errors_clearly():
    persons = make_persons()
    scenario = WAGE_MARGIN_PRESETS["wage_margin_pss"]
    try:
        from uk_ai_study.shocks import load_pss_weights

        load_pss_weights()
    except FileNotFoundError as exc:
        assert "genai_expertise" in str(exc)
        with pytest.raises(FileNotFoundError):
            apply_wage_margin_shock(persons, scenario)
    else:
        # csv exists (built by the parallel task): the preset must just work
        shocked = apply_wage_margin_shock(persons, scenario)
        assert (shocked["employment_income"].to_numpy() >= 0).all()
