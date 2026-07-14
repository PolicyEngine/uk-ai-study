"""Referee M6: duration and UC take-up sensitivities for the CENTRAL
(exposure-proportional) scenario. 20 draws (seeds 0-19) each.

(a) DURATION (6 months): displaced workers keep 50% of their baseline annual
    employment income instead of 0 (half a year out of work), evaluated
    IN-MODEL so taxes and means-tested benefits respond to the actual annual
    income. This is a documented hybrid: the annual model has no intra-year
    timing, so displaced persons carry the full displacement transition
    (hours zeroed, employment_status = UNEMPLOYED) while receiving half a
    year's earnings. The in-model run is preferred over the alternative
    "weight the annual result" (0.5 x full-year metrics deltas), which is
    exact for the exchequer only if the tax-benefit response were linear;
    the convex-combination comparator is reported alongside for reference.
    Survivor wage uplift and the capital shock are unchanged.

(b) UC TAKE-UP 70%: policyengine_uk models take-up through the benunit-level
    dataset input would_claim_uc (UC is only paid where it is True). Per
    seed, a Bernoulli(0.70) draw at benunit level replaces the dataset's
    stochastic take-up flags in BOTH the baseline and the shocked simulation
    (same draw), so the reported deltas are shock effects under 70% take-up,
    not take-up effects.

Each draw is checkpointed to results/robustness/duration_takeup_draws.csv;
summary to results/robustness/duration_takeup_sensitivity.json.
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
from uk_ai_study.shocks import PRESETS, TRANSITION_ZEROED_VARIABLES, apply_shocks  # noqa: E402
from policy_counterfactuals import PERIOD, build_sim, person_calc, hh_calc  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "robustness"
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
N_DRAWS = 20
EARNINGS_RETENTION = 0.5   # half a year of earnings kept by displaced
UC_TAKEUP = 0.70
CSV = OUT / "duration_takeup_draws.csv"
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
        "uc_bn": float((hh_calc(s, "universal_credit") * hw).sum() / 1e9),
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
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
    persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))
    w = persons["weight"].to_numpy()
    base_emp = persons["employment_income"].to_numpy(dtype=float)

    b = sim_metrics(baseline)
    base_arrays = {
        var: person_calc(baseline, var).astype(float) for var in TRANSITION_ZEROED_VARIABLES
    }
    base_arrays["employment_status"] = person_calc(baseline, "employment_status").astype(object)
    base_would_claim = baseline.calculate(
        "would_claim_uc", period=PERIOD, map_to="benunit"
    ).values.astype(bool)
    n_benunits = len(base_would_claim)
    baseline_takeup_mean_w = float(
        np.average(
            baseline.calculate("would_claim_uc", period=PERIOD, map_to="person").values,
            weights=w,
        )
    )
    del baseline

    done = set()
    if CSV.exists():
        d = pd.read_csv(CSV)
        done = set(zip(d["variant"], d["seed"]))

    scenario = PRESETS["central"]
    for seed in range(N_DRAWS):
        table = None
        for variant in ("duration_6m", "takeup_70"):
            if (variant, seed) in done:
                continue
            if table is None:
                table = apply_shocks(persons, scenario, seed=seed)
            displaced = table["displaced"].to_numpy()

            if variant == "duration_6m":
                t = table.copy()
                emp = t["employment_income"].to_numpy(dtype=float)
                t["employment_income"] = np.where(
                    displaced, EARNINGS_RETENTION * base_emp, emp
                )
                sim = build_sim(ds, base_arrays, t)
                m = sim_metrics(sim)
                del sim
                row_b = b
            else:
                rng = np.random.default_rng(10_000 + seed)
                claim = rng.random(n_benunits) < UC_TAKEUP
                # same take-up draw in baseline and shocked worlds
                sim0 = Microsimulation(dataset=ds)
                sim0.set_input("would_claim_uc", PERIOD, claim)
                row_b = sim_metrics(sim0)
                del sim0
                sim = build_sim(ds, base_arrays, table)
                sim.set_input("would_claim_uc", PERIOD, claim)
                m = sim_metrics(sim)
                del sim

            row = {
                "variant": variant,
                "seed": seed,
                "exchequer_cost_bn": (row_b["gov"] - m["gov"]) / 1e9,
                "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - row_b["pov_bhc"]),
                "gini_change_pp": 100 * (m["gini"] - row_b["gini"]),
                "uc_change_bn": m["uc_bn"] - row_b["uc_bn"],
                "baseline_pov_bhc": row_b["pov_bhc"],
                "displaced_weighted_m": float(w[displaced].sum() / 1e6),
            }
            pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
            print(f"{variant} seed {seed} done", flush=True)

    d = pd.read_csv(CSV)
    summary = {
        "n_draws": N_DRAWS,
        "duration_6m": {
            "description": "Central scenario; displaced workers keep 50% of "
                           "baseline annual earnings (6-month out-of-work "
                           "duration), in-model; hours/status transition "
                           "unchanged (documented hybrid).",
        },
        "takeup_70": {
            "description": "Central scenario under Bernoulli(0.70) UC take-up "
                           "(would_claim_uc set identically in baseline and "
                           "shocked simulations, redrawn per seed).",
            "dataset_baseline_would_claim_uc_person_weighted_mean": baseline_takeup_mean_w,
        },
    }
    for variant in ("duration_6m", "takeup_70"):
        g = d[d.variant == variant]
        summary[variant]["results"] = {
            c: {"mean": float(g[c].mean()), "sd": float(g[c].std(ddof=1)),
                "min": float(g[c].min()), "max": float(g[c].max())}
            for c in METRICS + ["uc_change_bn"]
        }

    # convex-combination comparator for duration: 0.5 x full-year central MC
    mc_path = OUT / "incidence_draws_five.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        mc = mc[mc.family == "exposure"]
        summary["duration_6m"]["convex_combination_comparator_0.5x_full_year"] = {
            c: {"mean": float(0.5 * mc[c].mean()), "sd": float(0.5 * mc[c].std(ddof=1))}
            for c in METRICS
        }
    (OUT / "duration_takeup_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
