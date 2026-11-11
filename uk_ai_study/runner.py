"""Run shock scenarios through PolicyEngine UK and summarise deltas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from uk_ai_study.checkpoint import certified_data_build_id
from uk_ai_study.data import microsimulation, soc_major_group
from uk_ai_study.exposure import exposure_for_major_group
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
    realised_displacement_rate,
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
    # weighted baseline employees and the share of them this draw actually
    # displaced: one draw realises the prescribed quota only up to sampling
    # noise (seed 0 central: 6.76 per cent against 7)
    employees_weighted: float | None = None
    realised_displacement_rate: float | None = None
    # per-capita change by decile under the alternative rankings of
    # ``decile_rankings`` (the HBAI ranking above is the main measure)
    decile_income_change_by_ranking: dict = field(default_factory=dict)
    certified_data_build_id: str | None = None


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


#: Households whose personal pension contributions exceed both this floor and
#: their earnings are excluded in the decile-ranking exclusion check (the
#: same rule the review applied: a few such records sit in decile 1 only
#: because HBAI income subtracts the contributions).
LARGE_PPC_FLOOR = 10_000.0
DECILE_RANKINGS = ("hbai", "pre_pension", "hbai_excl_large_ppc")


def weighted_deciles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Deciles 1-10 of ``values`` with person weights (JR16 convention)."""
    # values keep their dtype: argsort can order ties differently in float32
    # and float64, and every script ranks the raw float32 model output
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = cum / cum[-1]
    return np.clip(np.ceil(ranks * 10).astype(int), 1, 10)


def household_to_person(sim, values_h: np.ndarray, period: int) -> np.ndarray:
    """Broadcast a household-level array to its members."""
    hh_ids = sim.calculate("household_id", period=period, map_to="household").values
    person_hh = sim.calculate("household_id", period=period, map_to="person").values
    index = pd.Series(np.arange(len(hh_ids)), index=hh_ids)
    return np.asarray(values_h)[index.loc[person_hh].to_numpy()]


def ranking_inputs(sim, period: int) -> dict:
    """Household arrays behind the alternative decile rankings."""

    def h(v):
        return np.asarray(sim.calculate(v, period=period, map_to="household").values, dtype=float)

    return {
        "hbai": h("hbai_household_net_income"),
        "equiv_hbai": h("equiv_hbai_household_net_income"),
        "equivalisation": h("household_equivalisation_bhc"),
        "pension_contributions": h("employee_pension_contributions")
        + h("personal_pension_contributions"),
        "personal_pension_contributions": h("personal_pension_contributions"),
        "earnings": h("employment_income") + h("self_employment_income"),
    }


def decile_rankings_from_inputs(inputs: dict, to_person, weights: np.ndarray) -> dict:
    """Person-level (deciles, include) for each ranking in DECILE_RANKINGS.

    ``hbai``: baseline equivalised HBAI income (the main measure).
    ``pre_pension``: the same income before employee and personal pension
    contributions, which HBAI subtracts, equivalised the same way.
    ``hbai_excl_large_ppc``: the main ranking, with households whose personal
    pension contributions exceed both LARGE_PPC_FLOOR and their earnings left
    out of every decile.
    """
    main_values = inputs.get("equiv_hbai_person")
    if main_values is None:
        main_values = to_person(inputs["equiv_hbai"])
    main = weighted_deciles(main_values, weights)
    eqv = inputs["equivalisation"]
    pre = np.divide(
        inputs["hbai"] + inputs["pension_contributions"], eqv,
        out=np.zeros_like(eqv), where=eqv > 0,
    )
    large = (inputs["personal_pension_contributions"] > LARGE_PPC_FLOOR) & (
        inputs["personal_pension_contributions"] > np.maximum(inputs["earnings"], 0.0)
    )
    everyone = np.ones(len(weights), dtype=bool)
    return {
        "hbai": (main, everyone),
        "pre_pension": (weighted_deciles(to_person(pre), weights), everyone),
        "hbai_excl_large_ppc": (main, ~to_person(large).astype(bool)),
        "_large_ppc_households": large,
    }


def decile_rankings(sim, period: int, weights: np.ndarray) -> dict:
    inputs = ranking_inputs(sim, period)
    # the main ranking uses the person-level model output exactly as every
    # other script does, so its deciles are identical to theirs
    inputs["equiv_hbai_person"] = sim.calculate(
        "equiv_hbai_household_net_income", period=period, map_to="person"
    ).values
    return decile_rankings_from_inputs(
        inputs, lambda v: household_to_person(sim, v, period), weights
    )


def decile_mean_change(delta, weights, deciles, include) -> dict:
    """Person-weighted mean of ``delta`` by decile over included persons."""
    out = {}
    for d in range(1, 11):
        m = (deciles == d) & include
        out[int(d)] = float(np.average(delta[m], weights=weights[m])) if m.any() else float("nan")
    return out


#: Exchequer-cost components (fix for the headline framing). Taxes are
#: person-level, spending household-level; ``other_income_tax`` is income tax
#: net of the dividend and savings income tax it contains. The cost of a
#: component is minus the change in a tax, or plus the change in a spending
#: line; ``other`` is the residual against the full gov_balance change, so
#: the components always sum to the headline cost.
CHILDCARE_VARIABLES = (
    "tax_free_childcare",
    "extended_childcare_entitlement",
    "universal_childcare_entitlement",
    "targeted_childcare_entitlement",
    "wales_childcare_offer",
)
COST_COMPONENTS = (
    "dividend_tax",
    "savings_tax",
    "other_income_tax",
    "employee_ni",
    "employer_ni",
    "universal_credit",
    "childcare",
)
TAX_COMPONENTS = ("dividend_tax", "savings_tax", "other_income_tax", "employee_ni", "employer_ni")


def cost_component_totals(sim, period: int) -> dict:
    """Weighted totals (£) behind the cost decomposition, one simulation."""
    pw = np.asarray(sim.calculate("person_weight", period=period, map_to="person").values, dtype=float)
    hw = np.asarray(sim.calculate("household_weight", period=period, map_to="household").values, dtype=float)

    def p(v):
        return float((np.asarray(sim.calculate(v, period=period, map_to="person").values, dtype=float) * pw).sum())

    def h(v):
        return float((np.asarray(sim.calculate(v, period=period, map_to="household").values, dtype=float) * hw).sum())

    income_tax = p("income_tax")
    dividend_tax = p("dividend_income_tax")
    savings_tax = p("savings_income_tax")
    employee_ni = p("national_insurance")
    benefits = h("household_benefits")
    return {
        "gov_balance": h("gov_balance"),
        "income_tax": income_tax,
        "dividend_tax": dividend_tax,
        "savings_tax": savings_tax,
        "other_income_tax": income_tax - dividend_tax - savings_tax,
        "employee_ni": employee_ni,
        "employer_ni": p("ni_employer"),
        "universal_credit": h("universal_credit"),
        "childcare": sum(h(v) for v in CHILDCARE_VARIABLES),
        "household_benefits": benefits,
        # the narrow cash basis of the JR16 grid: IT + employee NI - benefits
        "narrow_cash_balance": income_tax + employee_ni - benefits,
    }


def cost_decomposition(base: dict, shocked: dict) -> dict:
    """Exchequer cost (£bn, positive = cost) split by component.

    Returns the total (from gov_balance), each COST_COMPONENTS entry,
    ``other`` (the residual), ``excluding_employer_ni`` and the narrow cash
    basis with its gap to the gov_balance total.
    """
    total = (base["gov_balance"] - shocked["gov_balance"]) / 1e9
    out = {"total": total}
    for k in COST_COMPONENTS:
        delta = (shocked[k] - base[k]) / 1e9
        out[k] = -delta if k in TAX_COMPONENTS else delta
    out["other"] = total - sum(out[k] for k in COST_COMPONENTS)
    out["income_tax"] = out["dividend_tax"] + out["savings_tax"] + out["other_income_tax"]
    out["excluding_employer_ni"] = total - out["employer_ni"]
    out["narrow_cash"] = (base["narrow_cash_balance"] - shocked["narrow_cash_balance"]) / 1e9
    out["gov_balance_minus_narrow_cash"] = total - out["narrow_cash"]
    if not np.isclose(sum(out[k] for k in COST_COMPONENTS) + out["other"], total, rtol=0, atol=1e-9):
        raise AssertionError("cost components do not sum to the gov_balance cost")
    return out


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


def build_person_table(
    sim,
    period: int,
    *,
    extra_variables: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build the canonical shock person table used by every analysis.

    Occupation is Microcosm's ``soc2020_major_group`` (1-9, NaN when none is
    recorded). Employees without a major group receive the
    survey-employment-weighted exposure and complementarity means.  Keeping this construction here prevents the
    central runner and extension scripts from silently defining different
    versions of the same exposure scenario.
    """
    variables = PERSON_VARIABLES + tuple(
        v for v in extra_variables if v not in PERSON_VARIABLES
    )
    table = pd.DataFrame(
        {v: sim.calculate(v, period=period, map_to="person").values for v in variables}
    )
    table["weight"] = sim.calculate("person_weight", period=period, map_to="person").values
    table["soc_major_group"] = soc_major_group(sim, period)
    employed = table["employment_income"].to_numpy(dtype=float) > 0
    weights = table["weight"].to_numpy(dtype=float)

    def weighted_fill(values: np.ndarray) -> np.ndarray:
        ok = np.isfinite(values) & employed
        mean = (
            float(np.average(values[ok], weights=weights[ok])) if ok.any() else 0.0
        )
        return np.where(np.isfinite(values), values, mean)

    table["exposure"] = weighted_fill(
        exposure_for_major_group(table["soc_major_group"], "c_aioe")
    )
    table["complementarity"] = weighted_fill(
        exposure_for_major_group(table["soc_major_group"], "complementarity_theta")
    )
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
    scenario: ShockScenario | str,
    period: int = 2026,
    seed: int = 0,
) -> ScenarioResult:
    if isinstance(scenario, str):
        scenario = (
            PRESETS.get(scenario)
            or RIPPLE_PRESETS.get(scenario)
            or WAGE_MARGIN_PRESETS[scenario]
        )

    baseline = microsimulation()

    persons = build_person_table(baseline, period)

    if isinstance(scenario, WageMarginScenario):
        # the seed drives the paired central displacement draw the gross cut
        # is calibrated to (R2-2)
        shocked_table = apply_wage_margin_shock(persons, scenario, seed=seed)
    elif isinstance(scenario, RippleScenario):
        shocked_table = apply_ripple_shocks(persons, scenario, seed=seed)
    else:
        shocked_table = apply_shocks(persons, scenario, seed=seed)

    shocked = build_shocked_simulation(baseline, shocked_table, period)
    displaced = shocked_table["displaced"].to_numpy()

    base, shock = _metrics(baseline, period), _metrics(shocked, period)

    weight = persons["weight"].to_numpy()
    age = persons["age"].to_numpy()
    # per-capita household disposable income change (£ per person)
    income_delta = shock["hni_pc"] - base["hni_pc"]
    # weighted deciles of baseline EQUIVALISED household disposable income
    # (HBAI concept, person-level, survey-weighted) — the JR16 convention —
    # plus the pre-pension-contribution ranking and the exclusion check
    rankings = decile_rankings(baseline, period, weight)
    deciles = rankings["hbai"][0]
    decile_change = decile_mean_change(
        income_delta, weight, deciles, np.ones(len(weight), dtype=bool)
    )
    by_ranking = {
        name: decile_mean_change(income_delta, weight, *rankings[name])
        for name in DECILE_RANKINGS
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
        employees_weighted=float(weight[persons["employment_income"].to_numpy() > 0].sum()),
        realised_displacement_rate=realised_displacement_rate(persons, displaced),
        decile_income_change_by_ranking=by_ranking,
        certified_data_build_id=certified_data_build_id(),
    )


def write_result(result: ScenarioResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(result), indent=2))
