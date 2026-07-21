"""Robustness to mixed job-loss/wage adjustment and negative survivor wages.

Outputs:
  results/robustness/mixed_adjustment.csv
  results/robustness/survivor_wage_grid.csv
  results/robustness/mixed_wage_adjustment.json

The mixed grid varies lambda, the share of a common 7% labour adjustment
delivered through displacement rather than C-AIOE-graded wage cuts.  The
survivor-wage grid fixes the central 7% displacement draw and varies the
complementarity-graded wage effect from -5% to +5%.  Capital treatment and
all tax-benefit recomputation are unchanged from the main paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.runner import gini
from uk_ai_study.shocks import (
    MixedMarginScenario,
    ShockScenario,
    apply_mixed_margin_shock,
    apply_shocks,
    build_shocked_simulation,
)

DATA = Path("data")
OUT = Path("results/robustness")
H5 = DATA / "frs_2024_25.h5"
ADULT = DATA / "frs_2024_25/UKDA-9563-tab/tab/adult.tab"
PERIOD = 2026
SEED = 0
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SURVIVOR_WAGE_EFFECTS = (-0.05, -0.026, 0.0, 0.026, 0.05)


def main() -> None:
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = UKSingleYearDataset(file_path=str(H5))
    baseline = Microsimulation(dataset=dataset)

    def pcalc(variable: str) -> np.ndarray:
        return baseline.calculate(variable, period=PERIOD, map_to="person").values

    persons = pd.DataFrame(
        {
            "person_id": pcalc("person_id"),
            "age": pcalc("age"),
            "employment_income": pcalc("employment_income"),
            "savings_interest_income": pcalc("savings_interest_income"),
            "dividend_income": pcalc("dividend_income"),
            "weight": pcalc("person_weight"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
    exposure = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    theta = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(exposure), exposure, np.nanmean(exposure))
    persons["complementarity"] = np.where(np.isfinite(theta), theta, np.nanmean(theta))

    def metrics(sim) -> dict[str, float]:
        pw = sim.calculate("person_weight", period=PERIOD, map_to="person").values
        hw = sim.calculate("household_weight", period=PERIOD, map_to="household").values
        equiv = sim.calculate(
            "equiv_hbai_household_net_income", period=PERIOD, map_to="household"
        ).values
        count = sim.calculate(
            "household_count_people", period=PERIOD, map_to="household"
        ).values
        return {
            "gov_balance": float(
                (
                    sim.calculate("gov_balance", period=PERIOD, map_to="household").values
                    * hw
                ).sum()
            ),
            "poverty_bhc": float(
                np.average(
                    sim.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values,
                    weights=pw,
                )
            ),
            "poverty_ahc": float(
                np.average(
                    sim.calculate("in_poverty_ahc", period=PERIOD, map_to="person").values,
                    weights=pw,
                )
            ),
            "gini": gini(equiv, hw * count),
        }

    base_metrics = metrics(baseline)
    base_earnings = persons["employment_income"].to_numpy(dtype=float)
    weights = persons["weight"].to_numpy(dtype=float)
    base_wage_bill = float((base_earnings * weights).sum())

    def evaluate(label: str, table: pd.DataFrame) -> dict[str, float | str]:
        sim = build_shocked_simulation(dataset, baseline, table, PERIOD)
        shocked_metrics = metrics(sim)
        shocked_earnings = table["employment_income"].to_numpy(dtype=float)
        displaced = table["displaced"].to_numpy(dtype=bool)
        return {
            "scenario": label,
            "displaced_weighted_m": float(weights[displaced].sum() / 1e6),
            "net_employment_income_change_pct": float(
                100 * ((shocked_earnings - base_earnings) * weights).sum() / base_wage_bill
            ),
            "exchequer_cost_bn": float(
                (base_metrics["gov_balance"] - shocked_metrics["gov_balance"]) / 1e9
            ),
            "poverty_change_bhc_pp": float(
                100 * (shocked_metrics["poverty_bhc"] - base_metrics["poverty_bhc"])
            ),
            "poverty_change_ahc_pp": float(
                100 * (shocked_metrics["poverty_ahc"] - base_metrics["poverty_ahc"])
            ),
            "gini_change_pp": float(
                100 * (shocked_metrics["gini"] - base_metrics["gini"])
            ),
        }

    mixed_rows = []
    for lam in LAMBDAS:
        table = apply_mixed_margin_shock(
            persons,
            MixedMarginScenario(f"mixed_lambda_{lam:g}", displacement_share=lam),
            seed=SEED,
        )
        row = evaluate(f"mixed_lambda_{lam:g}", table)
        row["lambda_displacement_share"] = lam
        mixed_rows.append(row)
        print(row, flush=True)

    wage_rows = []
    for wage_effect in SURVIVOR_WAGE_EFFECTS:
        table = apply_shocks(
            persons,
            ShockScenario("survivor_wage_grid", 0.07, wage_effect),
            seed=SEED,
        )
        row = evaluate(f"survivor_wage_{wage_effect:+.3f}", table)
        row["survivor_wage_effect_pct"] = 100 * wage_effect
        wage_rows.append(row)
        print(row, flush=True)

    mixed_df = pd.DataFrame(mixed_rows)
    wage_df = pd.DataFrame(wage_rows)
    mixed_df.to_csv(OUT / "mixed_adjustment.csv", index=False)
    wage_df.to_csv(OUT / "survivor_wage_grid.csv", index=False)
    payload = {
        "description": "Reduced-form adjustment-margin robustness, FRS 2024-25, 2026, seed 0",
        "mixed_adjustment": mixed_rows,
        "survivor_wage_grid": wage_rows,
        "assumptions": {
            "aggregate_adjustment_share": 0.07,
            "mixed_wage_gradient": "C-AIOE",
            "standard_wage_uplift": 0.026,
            "capital_return_increase_pp": 0.4,
        },
    }
    (OUT / "mixed_wage_adjustment.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
