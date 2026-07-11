"""ESRI JR16 shock mechanics (eqs 3.4 / 3.5) on a person table.

Employment shock (eq 3.4): the aggregate number of displaced workers is
``displacement_rate x employed``; it is allocated across SOC major groups in
proportion to ``employment x mean C-AIOE`` of the group, then realised by
weighted random draws within each group (probability proportional to
individual exposure).

Wage shock (eq 3.5): surviving workers receive a wage uplift whose aggregate
equals ``wage_uplift x surviving wage bill``, distributed across persons in
proportion to complementarity (theta) — AI complements, rather than
substitutes for, high-theta occupations.

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


#: Literature-anchored presets (overridable):
#: central — Briggs & Kodnani (2023) 7% displacement; +2.6% wages is JR16's
#:   adopted median wage-change estimate (JR16 fn.3, sec 3.2).
#: low — Acemoglu (2025, Economic Policy 40(121)), employment-only per JR16 fn.8.
#: high — Brynjolfsson, Chandar & Chen (2025), cohort-specific decline, upper bound.
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
    group = persons["soc_major_group"].to_numpy()

    total_quota = scenario.displacement_rate * float(weight[employed].sum())
    groups = np.unique(group[employed & np.isfinite(group)])
    # group quotas proportional to employment x mean exposure
    emp_w = {g: float(weight[employed & (group == g)].sum()) for g in groups}
    exp_g = {
        g: float(np.average(exposure[employed & (group == g)], weights=weight[employed & (group == g)]))
        for g in groups
    }
    raw = {g: emp_w[g] * exp_g[g] for g in groups}
    scale = total_quota / sum(raw.values())
    displaced = np.zeros(len(persons), dtype=bool)

    age = persons["age"].to_numpy()
    for g in groups:
        members = np.flatnonzero(employed & (group == g))
        quota = raw[g] * scale
        p = exposure[members] * weight[members]
        if quota <= 0 or p.sum() <= 0:
            continue
        if scenario.youth_displacement_multiplier != 1.0:
            p = p * np.where(age[members] < 25, scenario.youth_displacement_multiplier, 1.0)
        p = p / p.sum()
        order = rng.permutation(len(members))
        # draw members (weighted, without replacement) until the weighted quota fills
        chosen = rng.choice(members, size=len(members), replace=False, p=p)
        cum = np.cumsum(weight[chosen])
        displaced[chosen[cum <= quota]] = True
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

    # eq 3.5: uplift pool = wage_uplift x surviving wage bill, distributed by theta
    theta = shocked["complementarity"].to_numpy(dtype=float)
    weight = shocked["weight"].to_numpy(dtype=float)
    pool = scenario.wage_uplift * float((employment * weight)[survivors].sum())
    share_base = (theta * employment * weight)[survivors]
    uplift = np.zeros_like(employment)
    if share_base.sum() > 0:
        uplift[survivors] = pool * (theta * employment)[survivors] / float(share_base.sum()) * 1.0
    employment_shocked = np.where(displaced, 0.0, employment + uplift)
    shocked["employment_income"] = employment_shocked

    capital_factor = (BASELINE_CAPITAL_RETURN + scenario.capital_return_increase) / BASELINE_CAPITAL_RETURN
    for column in ("savings_interest_income", "dividend_income"):
        shocked[column] = shocked[column].to_numpy(dtype=float) * capital_factor
    return shocked
