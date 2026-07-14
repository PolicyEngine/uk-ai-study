"""Referee M4: wage-tier-ratio sensitivity for the MEASURED incidence family.

Reruns the measured family with (high, low) wage-tier displacement multipliers
(3, 0.8) and (6, 0.7) alongside the baseline (9.6, 0.6), holding the junior
(5.8/4.5) and London (1.5) tilts and the central aggregate shock fixed.
20 draws (seeds 0-19) per variant; each draw checkpointed to
results/robustness/measured_wage_tier_draws.csv (rerun resumes); summary to
results/robustness/measured_wage_tier_sensitivity.json.
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
from uk_ai_study.runner import gini  # noqa: E402
from uk_ai_study.shocks import PRESETS, TRANSITION_ZEROED_VARIABLES  # noqa: E402
from measured_incidence import LONDON_MULT_CENTRAL, measured_table  # noqa: E402
from policy_counterfactuals import build_sim, person_calc, hh_calc  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "robustness"
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
N_DRAWS = 20
VARIANTS = {"3_0.8": (3.0, 0.8), "6_0.7": (6.0, 0.7), "9.6_0.6": (9.6, 0.6)}
CSV = OUT / "measured_wage_tier_draws.csv"
METRICS = ["exchequer_cost_bn", "poverty_change_bhc_pp", "gini_change_pp"]


def sim_metrics(s):
    pw = person_calc(s, "person_weight")
    hw = hh_calc(s, "household_weight")
    eq = hh_calc(s, "equiv_hbai_household_net_income")
    n = hh_calc(s, "household_count_people")
    return {
        "gov": float((hh_calc(s, "gov_balance") * hw).sum()),
        "pov_bhc": float(np.average(person_calc(s, "in_poverty_bhc"), weights=pw)),
        "gini": gini(eq, hw * n),
    }


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

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
            "region": person_calc(baseline, "region"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
    persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))
    w = persons["weight"].to_numpy()

    b = sim_metrics(baseline)
    base_arrays = {
        var: person_calc(baseline, var).astype(float) for var in TRANSITION_ZEROED_VARIABLES
    }
    base_arrays["employment_status"] = person_calc(baseline, "employment_status").astype(object)
    del baseline

    done = set()
    if CSV.exists():
        d = pd.read_csv(CSV)
        done = set(zip(d["variant"], d["seed"]))

    scenario = PRESETS["central"]
    for seed in range(N_DRAWS):
        for variant, (hi, lo) in VARIANTS.items():
            if (variant, seed) in done:
                continue
            table = measured_table(
                persons, scenario, LONDON_MULT_CENTRAL, seed=seed,
                high_wage_mult=hi, low_wage_mult=lo,
            )
            sim = build_sim(ds, base_arrays, table)
            m = sim_metrics(sim)
            del sim
            displaced = table["displaced"].to_numpy()
            row = {
                "variant": variant,
                "seed": seed,
                "exchequer_cost_bn": (b["gov"] - m["gov"]) / 1e9,
                "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - b["pov_bhc"]),
                "gini_change_pp": 100 * (m["gini"] - b["gini"]),
                "displaced_weighted_m": float(w[displaced].sum() / 1e6),
            }
            pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
            print(f"wage-tier {variant} seed {seed} done", flush=True)

    d = pd.read_csv(CSV)
    summary = {
        "description": "Measured family (Klein Teeselink) with alternative "
                       "high/low wage-tier displacement multipliers; junior "
                       "5.8/4.5 and London 1.5 tilts and central aggregate "
                       "shock held fixed; 20 draws (seeds 0-19).",
        "n_draws": N_DRAWS,
    }
    for variant in VARIANTS:
        g = d[d.variant == variant]
        summary[variant] = {
            c: {"mean": float(g[c].mean()), "sd": float(g[c].std(ddof=1)),
                "min": float(g[c].min()), "max": float(g[c].max())}
            for c in METRICS + ["displaced_weighted_m"]
        }
    (OUT / "measured_wage_tier_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
