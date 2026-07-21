"""Phase 1 Monte Carlo (REVISION_PLAN item 6): 20 draws (seeds 0-19) for

  (a) all five incidence families (exposure, junior, compression, uniform,
      Klein-anchored top-loaded) — mean/sd/min/max of exchequer cost, BHC poverty change and
      Gini change -> results/robustness/incidence_monte_carlo.json
  (b) every policy counterfactual in policy_counterfactuals.py (R1 under
      exposure/junior/uniform, R2/R3 under exposure), holding the reform
      definitions fixed and redrawing the shock
      -> results/robustness/policy_monte_carlo.json
  (c) the central scenario with the relative BHC poverty line
      (in_relative_poverty_bhc) added
      -> results/robustness/central_monte_carlo.json (recomputed; the
      "exposure" family IS the central preset, same apply_shocks call as
      uk_ai_study.runner.run_scenario)

Each (seed, family) draw is checkpointed to CSV as it completes
(incidence_draws_five.csv / policy_draws.csv); rerunning the script skips
completed draws and rebuilds the JSON summaries.
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

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.runner import gini
from uk_ai_study.shocks import PRESETS, TRANSITION_ZEROED_VARIABLES
from incidence_scenarios import shocked_table_for  # noqa: E402
from measured_incidence import LONDON_MULT_CENTRAL, measured_table  # noqa: E402
from policy_counterfactuals import (  # noqa: E402
    CAP_MULTIPLIER,
    TAPER_CUT_TO,
    UC_UPLIFT,
    WAGE_INSURANCE_CAP,
    WAGE_INSURANCE_RATE,
    YEAR_SPAN,
    build_sim,
    hh_state,
    metrics_from_state,
    person_calc,
    hh_calc,
)

DATA = ROOT / "data"
OUT = ROOT / "results" / "robustness"
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
PERIOD = 2026
N_DRAWS = 20
FAMILIES = ("exposure", "junior", "compression", "uniform", "klein_top_loaded")
FAMILIES_R1 = ("exposure", "junior", "uniform")
INC_CSV = OUT / "incidence_draws_five.csv"
POL_CSV = OUT / "policy_draws.csv"


def family_table(family: str, persons: pd.DataFrame, seed: int) -> pd.DataFrame:
    if family == "klein_top_loaded":
        return measured_table(persons, PRESETS["central"], LONDON_MULT_CENTRAL, seed=seed)
    return shocked_table_for(family, persons, seed=seed)


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def summarise(df: pd.DataFrame, cols) -> dict:
    return {
        c: {
            "mean": float(df[c].mean()),
            "sd": float(df[c].std(ddof=1)),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
        }
        for c in cols
    }


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
            "region": person_calc(baseline, "region"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
    persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))
    w = persons["weight"].to_numpy()
    base_emp = persons["employment_income"].to_numpy(dtype=float)

    def sim_metrics(s):
        pw = person_calc(s, "person_weight")
        hw = hh_calc(s, "household_weight")
        eq = hh_calc(s, "equiv_hbai_household_net_income")
        n = hh_calc(s, "household_count_people")
        return {
            "gov": float((hh_calc(s, "gov_balance") * hw).sum()),
            "pov_bhc": float(np.average(person_calc(s, "in_poverty_bhc"), weights=pw)),
            "relpov_bhc": float(
                np.average(person_calc(s, "in_relative_poverty_bhc"), weights=pw)
            ),
            "gini": gini(eq, hw * n),
        }

    b = sim_metrics(baseline)

    base_arrays = {
        var: person_calc(baseline, var).astype(float) for var in TRANSITION_ZEROED_VARIABLES
    }
    base_arrays["employment_status"] = person_calc(baseline, "employment_status").astype(object)
    person_hh = person_calc(baseline, "household_id")
    hh_ids = hh_calc(baseline, "household_id")
    hh_index = pd.Series(np.arange(len(hh_ids)), index=hh_ids)
    p2h = hh_index.loc[person_hh].to_numpy()
    pw_by_hh = np.bincount(p2h, weights=w, minlength=len(hh_ids))
    del baseline

    # reform parameter dicts, exactly as in policy_counterfactuals.main
    sa = system.parameters.gov.dwp.universal_credit.standard_allowance.amount
    r2_reform = {
        f"gov.dwp.universal_credit.standard_allowance.amount.{k}": {
            YEAR_SPAN: round(float(sa.children[k](f"{PERIOD}-01-01")) * UC_UPLIFT, 2)
        }
        for k in ("SINGLE_YOUNG", "SINGLE_OLD", "COUPLE_YOUNG", "COUPLE_OLD")
    }
    bc = system.parameters.gov.dwp.benefit_cap
    r3_reform = {
        "gov.dwp.universal_credit.means_test.reduction_rate": {YEAR_SPAN: TAPER_CUT_TO},
        **{
            f"gov.dwp.benefit_cap.{grp}.{loc}": {
                YEAR_SPAN: float(bc.children[grp].children[loc](f"{PERIOD}-01-01"))
                * CAP_MULTIPLIER
            }
            for grp in ("single", "non_single")
            for loc in ("in_london", "outside_london")
        },
    }
    reforms_exposure = {"R2_uc_circuit_breaker": r2_reform, "R3_cap_suspension_taper_cut": r3_reform}

    done_inc = set()
    if INC_CSV.exists():
        d = pd.read_csv(INC_CSV)
        # Migrate the pre-revision family name. Keeping those rows would mix
        # two calibrations in one checkpoint file after a clean branch rerun.
        if "measured" in set(d["family"]):
            d = d[d["family"] != "measured"].copy()
            d.to_csv(INC_CSV, index=False)
        done_inc = set(zip(d["seed"], d["family"]))
    done_pol = set()
    if POL_CSV.exists():
        d = pd.read_csv(POL_CSV)
        done_pol = set(zip(d["seed"], d["reform"], d["family"]))

    for seed in range(N_DRAWS):
        for family in FAMILIES:
            need_inc = (seed, family) not in done_inc
            pol_needed = [
                r for r in (["R1_wage_insurance"] if family in FAMILIES_R1 else [])
                + (list(reforms_exposure) if family == "exposure" else [])
                if (seed, r, family) not in done_pol
            ]
            if not need_inc and not pol_needed:
                continue

            table = family_table(family, persons, seed)
            displaced = table["displaced"].to_numpy()
            sim = build_sim(ds, base_arrays, table)
            m = sim_metrics(sim)
            s0 = hh_state(sim) if pol_needed else None
            del sim

            if need_inc:
                append_row(
                    INC_CSV,
                    {
                        "seed": seed,
                        "family": family,
                        "exchequer_cost_bn": (b["gov"] - m["gov"]) / 1e9,
                        "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - b["pov_bhc"]),
                        "relative_poverty_change_bhc_pp": 100 * (m["relpov_bhc"] - b["relpov_bhc"]),
                        "gini_change_pp": 100 * (m["gini"] - b["gini"]),
                        "displaced_weighted_m": float(w[displaced].sum() / 1e6),
                    },
                )
            if pol_needed:
                m0 = metrics_from_state(s0, pw_by_hh)
            if "R1_wage_insurance" in pol_needed:
                transfer_p = np.where(
                    displaced,
                    np.minimum(WAGE_INSURANCE_RATE * base_emp, WAGE_INSURANCE_CAP),
                    0.0,
                )
                gross = float((transfer_p * w).sum())
                transfer_hh = np.bincount(p2h, weights=transfer_p, minlength=len(s0["hw"]))
                m1 = metrics_from_state(s0, pw_by_hh, extra_hh_income=transfer_hh)
                append_row(
                    POL_CSV,
                    {
                        "seed": seed,
                        "reform": "R1_wage_insurance",
                        "family": family,
                        "exchequer_cost_bn": gross / 1e9,
                        "poverty_change_bhc_pp": 100 * (m1["poverty_bhc"] - m0["poverty_bhc"]),
                        "gini_change_pp": 100 * (m1["gini"] - m0["gini"]),
                    },
                )
            for name in pol_needed:
                if name == "R1_wage_insurance":
                    continue
                simr = build_sim(ds, base_arrays, table, reform=reforms_exposure[name])
                sr = hh_state(simr)
                del simr
                mr = metrics_from_state(sr, pw_by_hh)
                append_row(
                    POL_CSV,
                    {
                        "seed": seed,
                        "reform": name,
                        "family": family,
                        "exchequer_cost_bn": (s0["gov"] - sr["gov"]) / 1e9,
                        "poverty_change_bhc_pp": 100 * (mr["poverty_bhc"] - m0["poverty_bhc"]),
                        "gini_change_pp": 100 * (mr["gini"] - m0["gini"]),
                    },
                )
            print(f"seed {seed} family {family} done", flush=True)

    # ---- summaries ----
    inc = pd.read_csv(INC_CSV)
    expected = {(seed, family) for seed in range(N_DRAWS) for family in FAMILIES}
    observed = set(zip(inc["seed"], inc["family"]))
    if observed != expected or inc.duplicated(["seed", "family"]).any():
        raise ValueError("incidence checkpoint is incomplete, duplicated, or contains stale families")
    metrics_cols = ["exchequer_cost_bn", "poverty_change_bhc_pp", "gini_change_pp"]
    inc_summary = {
        fam: summarise(inc[inc.family == fam], metrics_cols) for fam in FAMILIES
    }
    (OUT / "incidence_monte_carlo.json").write_text(json.dumps(inc_summary, indent=2))

    pol = pd.read_csv(POL_CSV)
    pol_summary = {}
    for (reform, family), g in pol.groupby(["reform", "family"]):
        pol_summary[f"{reform}_{family}"] = summarise(g, metrics_cols)
    (OUT / "policy_monte_carlo.json").write_text(json.dumps(pol_summary, indent=2))

    # central = exposure family (identical mechanics to the central preset via
    # run_scenario); recompute central_monte_carlo.json with the relative-
    # poverty variant added
    central = inc[inc.family == "exposure"]
    central_summary = summarise(
        central, metrics_cols + ["relative_poverty_change_bhc_pp"]
    )
    central_summary["baseline_relative_poverty_bhc_rate"] = b["relpov_bhc"]
    (OUT / "central_monte_carlo.json").write_text(json.dumps(central_summary, indent=2))

    print(json.dumps({"incidence": inc_summary, "policy": pol_summary,
                      "central": central_summary}, indent=1))


if __name__ == "__main__":
    main()
