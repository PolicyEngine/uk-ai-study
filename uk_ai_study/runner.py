"""Run shock scenarios through PolicyEngine UK and summarise deltas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.shocks import (
    PRESETS,
    RIPPLE_PRESETS,
    WAGE_MARGIN_PRESETS,
    MixedMarginScenario,
    RippleScenario,
    ShockScenario,
    WageMarginScenario,
    apply_ripple_shocks,
    apply_shocks,
    apply_wage_margin_shock,
    apply_mixed_margin_shock,
    build_shocked_simulation,
)

AGE_BANDS = ((16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200))

PERSON_VARIABLES = (
    "person_id",
    "age",
    "employment_income",
    "savings_interest_income",
    "dividend_income",
)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    displacement_rate: float
    wage_uplift: float
    exchequer_cost: float
    poverty_rate_change_bhc: float
    poverty_rate_change_ahc: float
    gini_baseline: float
    gini_shocked: float
    displaced_weighted: float
    decile_income_change: dict = field(default_factory=dict)
    age_band_displacement_share: dict = field(default_factory=dict)
    age_band_income_change: dict = field(default_factory=dict)


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    # bottom-code at zero: negative incomes make the Gini exceed 1
    values = np.clip(np.asarray(values, float), 0.0, None)
    order = np.argsort(values)
    v, w = values[order], np.asarray(weights, float)[order]
    cw = np.cumsum(w)
    cv = np.cumsum(v * w)
    if cv[-1] == 0:
        return 0.0
    return float(1 - 2 * np.sum((cv - v * w / 2) * w) / (cv[-1] * cw[-1]))


def _person_table(sim, period: int) -> pd.DataFrame:
    table = pd.DataFrame(
        {v: sim.calculate(v, period=period, map_to="person").values for v in PERSON_VARIABLES}
    )
    table["weight"] = sim.calculate("person_weight", period=period, map_to="person").values
    return table


def _metrics(sim, period: int) -> dict:
    # income concept: HBAI cash disposable income throughout (Gini, deciles,
    # changes), matching the in_poverty_* concept — #1, finding 2. The broad
    # household_net_income (in-kind benefits, indirect taxes) is used only
    # inside gov_balance.
    hh_w = sim.calculate("household_weight", period=period, map_to="household").values
    equiv = sim.calculate("equiv_hbai_household_net_income", period=period, map_to="household").values
    hh_count = sim.calculate("household_count_people", period=period, map_to="household").values
    return {
        "gov_balance": float((sim.calculate("gov_balance", period=period, map_to="household").values * hh_w).sum()),
        "poverty_bhc": float(np.average(
            sim.calculate("in_poverty_bhc", period=period, map_to="person").values,
            weights=sim.calculate("person_weight", period=period, map_to="person").values,
        )),
        "poverty_ahc": float(np.average(
            sim.calculate("in_poverty_ahc", period=period, map_to="person").values,
            weights=sim.calculate("person_weight", period=period, map_to="person").values,
        )),
        "gini": gini(equiv, hh_w * hh_count),
        "hni": sim.calculate("hbai_household_net_income", period=period, map_to="person").values,
    }


def run_scenario(
    dataset_path: str | Path,
    adult_tab_path: str | Path,
    scenario: ShockScenario | WageMarginScenario | MixedMarginScenario | str,
    period: int = 2026,
    seed: int = 0,
) -> ScenarioResult:
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    if isinstance(scenario, str):
        scenario = (
            PRESETS.get(scenario)
            or RIPPLE_PRESETS.get(scenario)
            or WAGE_MARGIN_PRESETS[scenario]
        )

    dataset = UKSingleYearDataset(file_path=str(dataset_path))
    baseline = Microsimulation(dataset=dataset)

    persons = _person_table(baseline, period)
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], adult_tab_path)
    exposure = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    theta = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(exposure), exposure, np.nanmean(exposure))
    persons["complementarity"] = np.where(np.isfinite(theta), theta, np.nanmean(theta))

    if isinstance(scenario, MixedMarginScenario):
        shocked_table = apply_mixed_margin_shock(persons, scenario, seed=seed)
    elif isinstance(scenario, WageMarginScenario):
        # deterministic (no draw): the seed is irrelevant to the cut
        shocked_table = apply_wage_margin_shock(persons, scenario)
    elif isinstance(scenario, RippleScenario):
        shocked_table = apply_ripple_shocks(persons, scenario, seed=seed)
    else:
        shocked_table = apply_shocks(persons, scenario, seed=seed)

    shocked = build_shocked_simulation(dataset, baseline, shocked_table, period)
    displaced = shocked_table["displaced"].to_numpy()

    base, shock = _metrics(baseline, period), _metrics(shocked, period)

    weight = persons["weight"].to_numpy()
    age = persons["age"].to_numpy()
    income_delta = shock["hni"] - base["hni"]
    # weighted deciles of baseline EQUIVALISED household disposable income
    # (HBAI concept, person-level, survey-weighted) — the JR16 convention
    equiv = baseline.calculate(
        "equiv_hbai_household_net_income", period=period, map_to="person"
    ).values
    order = np.argsort(equiv)
    cum = np.cumsum(weight[order])
    ranks = np.empty(len(equiv), dtype=float)
    ranks[order] = cum / cum[-1]
    deciles = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
    decile_change = {
        int(d): float(np.average(income_delta[deciles == d], weights=weight[deciles == d]))
        for d in range(1, 11)
    }
    band_share, band_income = {}, {}
    displaced_w = float(weight[displaced].sum())
    for lo, hi in AGE_BANDS:
        mask = (age >= lo) & (age <= hi)
        label = f"{lo}-{hi if hi < 200 else '+'}"
        band_share[label] = float(weight[mask & displaced].sum() / displaced_w) if displaced_w else 0.0
        band_income[label] = float(np.average(income_delta[mask], weights=weight[mask])) if mask.any() else 0.0

    return ScenarioResult(
        scenario=scenario.name,
        displacement_rate=getattr(scenario, "displacement_rate", 0.0),
        wage_uplift=scenario.wage_uplift,
        exchequer_cost=base["gov_balance"] - shock["gov_balance"],
        poverty_rate_change_bhc=shock["poverty_bhc"] - base["poverty_bhc"],
        poverty_rate_change_ahc=shock["poverty_ahc"] - base["poverty_ahc"],
        gini_baseline=base["gini"],
        gini_shocked=shock["gini"],
        displaced_weighted=displaced_w,
        decile_income_change=decile_change,
        age_band_displacement_share=band_share,
        age_band_income_change=band_income,
    )


def write_result(result: ScenarioResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(result), indent=2))
