"""Referee M6 follow-up: does the R1-vs-R2 cost-effectiveness ranking survive
the half-earnings-retention displacement variant?

Central exposure-proportional shock with displaced workers keeping 50% of
baseline annual earnings while carrying full-year unemployed status/hours —
a 50%-earnings-retention hybrid, NOT a six-month unemployment spell (R2-10;
as in sensitivity_duration_takeup.py). Seeds 0-4.
  R1 wage insurance, retention-consistent: 50% of LOST earnings (= 50% x half
     a year's baseline pay), cap pro-rated to £7,500; post-simulation
     transfer, non-taxable, means-test disregarded (as in the main analysis).
  R2 UC circuit breaker: +20% on all four UC standard allowances, parameter
     reform on the retention-shocked simulation.
Ranking metric: £bn of gross/net cost per pp of BHC poverty averted relative
to the retention-shocked no-reform world.

Output: results/robustness/duration_reform_ranking.json (draws checkpointed
to duration_reform_draws.csv).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group  # noqa: E402
from uk_ai_study.shocks import PRESETS, TRANSITION_ZEROED_VARIABLES, apply_shocks  # noqa: E402
from policy_counterfactuals import (  # noqa: E402
    PERIOD, YEAR_SPAN, UC_UPLIFT, WAGE_INSURANCE_RATE, WAGE_INSURANCE_CAP,
    build_sim, hh_state, metrics_from_state, person_calc, hh_calc,
)

DATA = ROOT / "data"
OUT = ROOT / "results" / "robustness"
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
N_DRAWS = 5
RETENTION = 0.5
CSV = OUT / "duration_reform_draws.csv"


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk.system import system

    OUT.mkdir(parents=True, exist_ok=True)
    ds = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    baseline = Microsimulation(dataset=ds)
    persons = pd.DataFrame(
        {
            "person_id": person_calc(baseline, "person_id"),
            "age": person_calc(baseline, "age"),
            "employment_income": person_calc(baseline, "employment_income"),
            "savings_interest_income": person_calc(baseline, "savings_interest_income"),
            "dividend_income": person_calc(baseline, "dividend_income"),
            "weight": person_calc(baseline, "person_weight"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    # unmatched employees carry the EMPLOYMENT-WEIGHTED (survey-weight) mean
    # of the matched employed, as the paper states (R2-10)
    _emp = persons["employment_income"].to_numpy() > 0
    _w = persons["weight"].to_numpy()

    def _weighted_fill(values):
        ok = np.isfinite(values) & _emp
        mean = float(np.average(values[ok], weights=_w[ok])) if ok.any() else 0.0
        return np.where(np.isfinite(values), values, mean)

    persons["exposure"] = _weighted_fill(e)
    persons["complementarity"] = _weighted_fill(th)
    w = persons["weight"].to_numpy()
    base_emp = persons["employment_income"].to_numpy(dtype=float)

    base_arrays = {
        var: person_calc(baseline, var).astype(float) for var in TRANSITION_ZEROED_VARIABLES
    }
    base_arrays["employment_status"] = person_calc(baseline, "employment_status").astype(object)
    person_hh = person_calc(baseline, "household_id")
    hh_ids = hh_calc(baseline, "household_id")
    p2h = pd.Series(np.arange(len(hh_ids)), index=hh_ids).loc[person_hh].to_numpy()
    pw_by_hh = np.bincount(p2h, weights=w, minlength=len(hh_ids))
    del baseline

    sa = system.parameters.gov.dwp.universal_credit.standard_allowance.amount
    r2_reform = {
        f"gov.dwp.universal_credit.standard_allowance.amount.{k}": {
            YEAR_SPAN: round(float(sa.children[k](f"{PERIOD}-01-01")) * UC_UPLIFT, 2)
        }
        for k in ("SINGLE_YOUNG", "SINGLE_OLD", "COUPLE_YOUNG", "COUPLE_OLD")
    }

    done = set()
    if CSV.exists():
        done = set(pd.read_csv(CSV)["seed"])

    scenario = PRESETS["central"]
    for seed in range(N_DRAWS):
        if seed in done:
            continue
        table = apply_shocks(persons, scenario, seed=seed)
        displaced = table["displaced"].to_numpy()
        emp = table["employment_income"].to_numpy(dtype=float)
        table = table.copy()
        table["employment_income"] = np.where(displaced, RETENTION * base_emp, emp)

        sim0 = build_sim(ds, base_arrays, table)
        s0 = hh_state(sim0)
        del sim0
        m0 = metrics_from_state(s0, pw_by_hh)

        # R1 duration-consistent wage insurance
        transfer_p = np.where(
            displaced,
            np.minimum(WAGE_INSURANCE_RATE * (1 - RETENTION) * base_emp,
                       WAGE_INSURANCE_CAP * (1 - RETENTION)),
            0.0,
        )
        gross_r1 = float((transfer_p * w).sum())
        transfer_hh = np.bincount(p2h, weights=transfer_p, minlength=len(s0["hw"]))
        m1 = metrics_from_state(s0, pw_by_hh, extra_hh_income=transfer_hh)

        simr = build_sim(ds, base_arrays, table, reform=r2_reform)
        sr = hh_state(simr)
        del simr
        mr = metrics_from_state(sr, pw_by_hh)
        cost_r2 = s0["gov"] - sr["gov"]

        row = {
            "seed": seed,
            "r1_cost_bn": gross_r1 / 1e9,
            "r1_pov_bhc_pp": 100 * (m1["poverty_bhc"] - m0["poverty_bhc"]),
            "r2_cost_bn": cost_r2 / 1e9,
            "r2_pov_bhc_pp": 100 * (mr["poverty_bhc"] - m0["poverty_bhc"]),
        }
        row["r1_cost_per_pp_bn"] = row["r1_cost_bn"] / -row["r1_pov_bhc_pp"] if row["r1_pov_bhc_pp"] < 0 else None
        row["r2_cost_per_pp_bn"] = row["r2_cost_bn"] / -row["r2_pov_bhc_pp"] if row["r2_pov_bhc_pp"] < 0 else None
        pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
        print(f"duration reforms seed {seed} done", flush=True)

    d = pd.read_csv(CSV)
    summary = {
        "description": "R1 (retention-consistent wage insurance) vs R2 (UC +20% "
                       "standard allowance) under the half-earnings-retention "
                       "variant of the central shock (50% earnings retention, "
                       "full-year unemployed status; not a six-month spell); "
                       "poverty deltas vs the retention-shocked no-reform "
                       f"world; seeds 0-{N_DRAWS - 1}.",
        "n_draws": int(len(d)),
    }
    for c in d.columns:
        if c == "seed":
            continue
        summary[c] = {"mean": float(d[c].mean()), "sd": float(d[c].std(ddof=1))}
    (OUT / "duration_reform_ranking.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
