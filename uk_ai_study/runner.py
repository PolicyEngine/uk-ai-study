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
    RippleScenario,
    ShockScenario,
    WageMarginScenario,
    apply_ripple_shocks,
    apply_shocks,
    apply_wage_margin_shock,
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
    # per-capita household disposable income change (£), person-weighted mean
    decile_income_change: dict = field(default_factory=dict)
    age_band_displacement_share: dict = field(default_factory=dict)
    # per-capita household disposable income change (£), person-weighted mean
    age_band_income_change: dict = field(default_factory=dict)
    # wage-margin runs only (R2-2): the paired gross-cut calibration target
    # (share of the baseline wage bill) and the realised aggregate cut
    gross_cut_share_target: float | None = None
    gross_cut_realised: float | None = None


def per_capita_household_income(
    household_income: np.ndarray, household_size: np.ndarray
) -> np.ndarray:
    """Per-capita household disposable income (£ per person).

    ``map_to="person"`` broadcasts the household TOTAL to each member; dividing
    by the (person-broadcast) household size gives the per-capita quantity used
    in the cash-change decile and age-band summaries. Raises on zero, negative,
    or non-finite household sizes.
    """
    household_income = np.asarray(household_income, dtype=float)
    household_size = np.asarray(household_size, dtype=float)
    if household_income.shape != household_size.shape:
        raise ValueError(
            f"shape mismatch: income {household_income.shape} vs size {household_size.shape}"
        )
    if household_size.size and (
        not np.all(np.isfinite(household_size)) or np.any(household_size < 1)
    ):
        raise ValueError("household_size must be finite and >= 1 for every person")
    return household_income / household_size


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
        # per-capita household disposable income (£): the person-broadcast
        # household total divided by household size (issue #6)
        "hni_pc": per_capita_household_income(
            sim.calculate("hbai_household_net_income", period=period, map_to="person").values,
            sim.calculate("household_count_people", period=period, map_to="person").values,
        ),
    }


def run_scenario(
    dataset_path: str | Path,
    adult_tab_path: str | Path,
    scenario: ShockScenario | str,
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
    # unmatched employees carry the EMPLOYMENT-WEIGHTED mean of the matched
    # employed (survey-weighted), as the paper states (R2-10)
    _emp = persons["employment_income"].to_numpy() > 0
    _w = persons["weight"].to_numpy()

    def _weighted_fill(values: np.ndarray) -> np.ndarray:
        ok = np.isfinite(values) & _emp
        mean = float(np.average(values[ok], weights=_w[ok])) if ok.any() else 0.0
        return np.where(np.isfinite(values), values, mean)

    persons["exposure"] = _weighted_fill(exposure)
    persons["complementarity"] = _weighted_fill(theta)

    if isinstance(scenario, WageMarginScenario):
        # the seed drives the paired central displacement draw the gross cut
        # is calibrated to (R2-2)
        shocked_table = apply_wage_margin_shock(persons, scenario, seed=seed)
    elif isinstance(scenario, RippleScenario):
        shocked_table = apply_ripple_shocks(persons, scenario, seed=seed)
    else:
        shocked_table = apply_shocks(persons, scenario, seed=seed)

    shocked = build_shocked_simulation(dataset, baseline, shocked_table, period)
    displaced = shocked_table["displaced"].to_numpy()

    base, shock = _metrics(baseline, period), _metrics(shocked, period)

    weight = persons["weight"].to_numpy()
    age = persons["age"].to_numpy()
    # per-capita household disposable income change (£ per person)
    income_delta = shock["hni_pc"] - base["hni_pc"]
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
        gross_cut_share_target=shocked_table.attrs.get("gross_cut_share_target"),
        gross_cut_realised=shocked_table.attrs.get("gross_cut_realised"),
    )


def write_result(result: ScenarioResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(result), indent=2))
