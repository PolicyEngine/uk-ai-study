"""ESRI JR16 shock mechanics (eqs 3.4 / 3.5) on a person table.

Employment shock (eq 3.4): the aggregate number of displaced workers is
``displacement_rate x employed``; it is allocated across SOC major groups in
proportion to ``employment x mean C-AIOE`` of the group, then realised by
random draws with UNIFORM ordering keys within each group (the survey weight
enters only through quota consumption, so a represented person's inclusion
probability does not depend on their record's grossing weight — #1,
finding 6).

Wage shock (eq 3.5): surviving workers receive percentage uplifts
proportional to complementarity (theta), normalised by the
EMPLOYMENT-weighted mean theta over baseline workers — JR16-literal, per the
estimand decision on uk-ai-study#1 (finding 5).

Capital shock: interest and dividend income scaled by the ratio of the
shocked to the baseline return (JR16: 1.005% -> 1.405%, i.e. +0.4pp on the
return, a factor of ~1.398).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASELINE_CAPITAL_RETURN = 0.01005
CAPITAL_RETURN_INCREASE = 0.004


@dataclass(frozen=True)
class ShockScenario:
    name: str
    displacement_rate: float
    wage_uplift: float
    capital_return_increase: float = CAPITAL_RETURN_INCREASE
    youth_displacement_multiplier: float = 1.0  # >1 tilts draws toward ages 16-24


#: Scenario presets (overridable). The capital shock (+0.4pp on the return)
#: is ON in every preset, as in all JR16 scenarios.
#: central — 7% displacement / +2.6% wages: JR16's central calibration, which
#:   converts Briggs & Kodnani (2023) task-exposure and productivity figures
#:   into displacement and wage rates (JR16 sec 3.2).
#: low — 1% displacement, no wage uplift. JR16 sec 3.2 attributes ~1% to
#:   Acemoglu (2025), but his 0.9-1.1% is a ten-year GDP figure, not an
#:   employment effect (uk-ai-study#1, finding 11) — read this as a
#:   sensitivity case, not an evidence-anchored lower bound.
#: high — Brynjolfsson, Chandar & Chen: 13% per early drafts; the Nov 2025
#:   version reports 16%. Cohort-specific relative decline treated as an
#:   economy-wide absolute rate — an upper bound in both respects.
#: central_youth_tilted — central with Klein Teeselink (2025) junior/total
#:   employment-effect ratio 5.8/4.5 as the youth multiplier.
PRESETS = {
    "central": ShockScenario("central", 0.07, 0.026),
    "low": ShockScenario("low", 0.01, 0.0),
    "high": ShockScenario("high", 0.13, 0.026),
    "central_youth_tilted": ShockScenario(
        "central_youth_tilted", 0.07, 0.026, youth_displacement_multiplier=5.8 / 4.5
    ),
}


def draw_displaced(
    persons: pd.DataFrame,
    scenario: ShockScenario,
    seed: int = 0,
) -> np.ndarray:
    """Boolean displaced mask per eq 3.4 (employees only)."""
    rng = np.random.default_rng(seed)
    employed = persons["employment_income"].to_numpy() > 0
    exposure = persons["exposure"].to_numpy()
    # JR16 normalises C-AIOE so the least-exposed sector scores 0 (and thus
    # receives no eq 3.4 job losses); the raw standardised score is negative
    # for low-exposure groups, which would corrupt the quota weights.
    exposure = exposure - exposure[employed].min()
    weight = persons["weight"].to_numpy()
    # Employees without an observed SOC code form their own pseudo-group
    # (carrying their mean-imputed exposure), so the displacement universe is
    # ALL employees and matches the wage-uplift universe (#1, finding 7).
    group = np.where(
        np.isfinite(persons["soc_major_group"].to_numpy()),
        persons["soc_major_group"].to_numpy(),
        -1.0,
    )
    total_quota = scenario.displacement_rate * float(weight[employed].sum())
    groups = np.unique(group[employed])
    # group quotas proportional to employment x mean exposure
    emp_w = {g: float(weight[employed & (group == g)].sum()) for g in groups}
    exp_g = {
        g: float(np.average(exposure[employed & (group == g)], weights=weight[employed & (group == g)]))
        for g in groups
    }
    raw = {g: emp_w[g] * exp_g[g] for g in groups}
    if sum(raw.values()) <= 0:
        # degenerate case (uniform exposure): allocate by employment alone
        raw = {g: emp_w[g] for g in groups}
        exposure = np.ones_like(exposure)
    scale = total_quota / sum(raw.values())
    displaced = np.zeros(len(persons), dtype=bool)

    age = persons["age"].to_numpy()
    for g in groups:
        members = np.flatnonzero(employed & (group == g))
        quota = raw[g] * scale
        if quota <= 0:
            continue
        # uniform ordering keys within group: exposure is constant within a
        # 1-digit group, and weighting the ordering as well as the quota
        # consumption would double-count the survey weight, making a
        # represented person's risk depend on their record's grossing
        # weight (#1, finding 6)
        p = np.ones(len(members))
        if scenario.youth_displacement_multiplier != 1.0:
            p = p * np.where(age[members] < 25, scenario.youth_displacement_multiplier, 1.0)
        p = p / p.sum()
        # draw members (weighted, without replacement) until the weighted
        # quota fills; the quota-crossing person is included with probability
        # equal to the remaining quota fraction, so the expected displaced
        # weight equals the quota exactly
        chosen = rng.choice(members, size=len(members), replace=False, p=p)
        cum = np.cumsum(weight[chosen])
        displaced[chosen[cum <= quota]] = True
        crossing = np.searchsorted(cum, quota)
        if crossing < len(chosen) and cum[crossing] > quota:
            shortfall = quota - (cum[crossing - 1] if crossing else 0.0)
            if rng.random() < shortfall / weight[chosen[crossing]]:
                displaced[chosen[crossing]] = True
    return displaced


def apply_shocks(
    persons: pd.DataFrame,
    scenario: ShockScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Shocked copy of the person table: employment, wage and capital shocks.

    Expects columns: employment_income, savings_interest_income,
    dividend_income, exposure, complementarity, soc_major_group, age, weight.
    """
    shocked = persons.copy()
    displaced = draw_displaced(persons, scenario, seed=seed)
    shocked["displaced"] = displaced

    employment = shocked["employment_income"].to_numpy(dtype=float)
    survivors = (employment > 0) & ~displaced

    # eq 3.5: person-level % wage change = wage_uplift * theta_i / theta_bar,
    # with theta_bar the EMPLOYMENT-weighted mean theta over baseline workers
    # (JR16-literal; deterministic across draws) — estimand decision on
    # uk-ai-study#1, finding 5
    theta = shocked["complementarity"].to_numpy(dtype=float)
    weight = shocked["weight"].to_numpy(dtype=float)
    baseline_workers = employment > 0
    theta_bar = float(
        (theta * weight)[baseline_workers].sum() / weight[baseline_workers].sum()
    )
    uplift = np.zeros_like(employment)
    if theta_bar > 0:
        uplift[survivors] = scenario.wage_uplift * (theta[survivors] / theta_bar) * employment[survivors]
    employment_shocked = np.where(displaced, 0.0, employment + uplift)
    shocked["employment_income"] = employment_shocked

    capital_factor = (BASELINE_CAPITAL_RETURN + scenario.capital_return_increase) / BASELINE_CAPITAL_RETURN
    for column in ("savings_interest_income", "dividend_income"):
        shocked[column] = shocked[column].to_numpy(dtype=float) * capital_factor
    return shocked


#: The transition contract (#1, finding 4 / decision 2): "displaced" means
#: fully out of work. Besides employment_income = 0, these person-level
#: inputs are zeroed so displaced workers do not remain in_work (hours > 0
#: keeps UC childcare, tax-free childcare and extended childcare paying),
#: do not keep deducting pension contributions from zero earnings, and do
#: not draw statutory pay.
TRANSITION_ZEROED_VARIABLES = (
    "hours_worked",
    "employee_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    "statutory_maternity_pay",
    "statutory_paternity_pay",
    "statutory_sick_pay",
)

SHOCKED_INCOME_VARIABLES = (
    "employment_income",
    "savings_interest_income",
    "dividend_income",
)


def build_shocked_simulation(dataset, baseline_sim, shocked_table, period):
    """One shared constructor for the shocked simulation (every pipeline).

    Sets the shocked income inputs from ``shocked_table`` and applies the
    full displacement transition to displaced persons.
    """
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    for column in SHOCKED_INCOME_VARIABLES:
        sim.set_input(column, period, shocked_table[column].to_numpy(dtype=float))
    displaced = shocked_table["displaced"].to_numpy()
    for var in TRANSITION_ZEROED_VARIABLES:
        values = baseline_sim.calculate(var, period=period, map_to="person").values.astype(float)
        values[displaced] = 0.0
        sim.set_input(var, period, values)
    status = baseline_sim.calculate("employment_status", period=period, map_to="person").values.astype(object)
    status[displaced] = "UNEMPLOYED"
    # A rejected set_input here would silently leave displaced workers
    # EMPLOYED (with zero hours), changing benefit entitlements in every
    # result — fail hard rather than warn.
    sim.set_input("employment_status", period, status)
    applied = sim.calculate("employment_status", period=period, map_to="person").values
    if not (applied[displaced].astype(str) == "UNEMPLOYED").all():
        raise RuntimeError(
            "employment_status transition not applied: displaced persons are "
            "not all UNEMPLOYED in the shocked simulation."
        )
    return sim
