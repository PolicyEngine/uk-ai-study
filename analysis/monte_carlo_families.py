"""Monte Carlo analysis: 50 paired draws (seeds 0-49) for

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

  (d) for the exposure (central) family, the exchequer cost split by
      component on every draw (dividend, savings and other income tax,
      employee and employer NI, UC, childcare, other) with the narrow cash
      basis of the JR16 grid alongside
      -> results/robustness/central_cost_components_draws.csv and
         results/robustness/central_cost_decomposition.json

Each (seed, family) draw is checkpointed to CSV as it completes
(incidence_draws_five.csv / policy_draws.csv /
central_cost_components_draws.csv), stamped with the certified data build
and code fingerprint. An existing checkpoint is continued only with
``--resume`` (and refused if its stamp differs); otherwise the run starts
from scratch.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from uk_ai_study.checkpoint import Checkpoint, certified_data_build_id
from uk_ai_study.runner import (
    CHILDCARE_VARIABLES,
    COST_COMPONENTS,
    build_person_table,
    cost_component_totals,
    cost_decomposition,
    gini,
)
from uk_ai_study.shocks import PRESETS, TRANSITION_ZEROED_VARIABLES, realised_displacement_rate
from incidence_scenarios import (  # noqa: E402
    COMPRESSION_TOP_TERTILE_MULTIPLIER,
    shocked_table_for,
)
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

OUT = ROOT / "results" / "robustness"
PERIOD = 2026
N_DRAWS = 50
FAMILIES = ("exposure", "junior", "compression", "uniform", "klein_top_loaded")
FAMILIES_R1 = ("exposure", "junior", "uniform")
INC_CSV = OUT / "incidence_draws_five.csv"
POL_CSV = OUT / "policy_draws.csv"
COST_CSV = OUT / "central_cost_components_draws.csv"
COST_FIELDS = COST_COMPONENTS + (
    "other", "income_tax", "excluding_employer_ni", "narrow_cash",
    "gov_balance_minus_narrow_cash",
)


def family_table(
    family: str,
    persons: pd.DataFrame,
    seed: int,
    compression_multiplier: float = COMPRESSION_TOP_TERTILE_MULTIPLIER,
) -> pd.DataFrame:
    """Shocked table for one (family, seed) draw.

    Families use the R2-3 factorial design: shocked_table_for's default
    wage_axis="central" holds the person-level wage-uplift and capital
    channels fixed across families; only the displacement mask varies.
    ``compression_multiplier`` (R2-6a) is the compression draw tilt
    (1.5 / 2.0 / 3.0 sensitivity runs).
    """
    if family == "klein_top_loaded":
        return measured_table(persons, PRESETS["central"], LONDON_MULT_CENTRAL, seed=seed)
    return shocked_table_for(
        family, persons, seed=seed, compression_multiplier=compression_multiplier
    )


def summarise(df: pd.DataFrame, cols) -> dict:
    return {
        c: {
            "mean": float(df[c].mean()),
            "sd": float(df[c].std(ddof=1)),
            "mcse": float(df[c].std(ddof=1) / np.sqrt(len(df))),
            "q025": float(df[c].quantile(0.025)),
            "median": float(df[c].median()),
            "q975": float(df[c].quantile(0.975)),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
        }
        for c in cols
    }


def convergence_diagnostics(df: pd.DataFrame, cols, checkpoint: int = 20) -> dict:
    """Compare the original draw budget with the enlarged run."""
    if len(df) <= checkpoint:
        raise ValueError("convergence diagnostic requires more checkpoint draws")
    first = df.nsmallest(checkpoint, "seed")
    out = {}
    for col in cols:
        full_mean = float(df[col].mean())
        early_mean = float(first[col].mean())
        mcse = float(df[col].std(ddof=1) / np.sqrt(len(df)))
        delta = full_mean - early_mean
        out[col] = {
            "checkpoint_draws": checkpoint,
            "full_draws": int(len(df)),
            "checkpoint_mean": early_mean,
            "full_mean": full_mean,
            "mean_change": delta,
            "full_mcse": mcse,
            "passes_two_mcse": bool(abs(delta) <= 2 * mcse),
        }
    return out


def main():
    from uk_ai_study.data import microsimulation
    from policyengine_uk.system import system

    OUT.mkdir(parents=True, exist_ok=True)
    baseline = microsimulation()

    persons = build_person_table(baseline, PERIOD, extra_variables=("region",)
    )
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
            "pov_ahc": float(np.average(person_calc(s, "in_poverty_ahc"), weights=pw)),
            "relpov_bhc": float(
                np.average(person_calc(s, "in_relative_poverty_bhc"), weights=pw)
            ),
            "gini": gini(eq, hw * n),
        }

    b = sim_metrics(baseline)
    b_cost = cost_component_totals(baseline, PERIOD)

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

    inc_ckpt = Checkpoint(INC_CSV, ("seed", "family"), script=__file__)
    pol_ckpt = Checkpoint(POL_CSV, ("seed", "reform", "family"), script=__file__)
    cost_ckpt = Checkpoint(COST_CSV, ("seed",), script=__file__)
    done_inc, done_pol = inc_ckpt.done, pol_ckpt.done

    for seed in range(N_DRAWS):
        for family in FAMILIES:
            need_inc = (seed, family) not in done_inc
            need_cost = family == "exposure" and not cost_ckpt.is_done(seed)
            pol_needed = [
                r for r in (["R1_wage_insurance"] if family in FAMILIES_R1 else [])
                + (list(reforms_exposure) if family == "exposure" else [])
                if (seed, r, family) not in done_pol
            ]
            if not need_inc and not pol_needed and not need_cost:
                continue

            table = family_table(family, persons, seed)
            displaced = table["displaced"].to_numpy()
            sim = build_sim(base_arrays, table)
            m = sim_metrics(sim)
            s0 = hh_state(sim) if pol_needed else None
            costs = cost_component_totals(sim, PERIOD) if need_cost else None
            del sim
            gc.collect()

            if need_cost:
                cost_ckpt.append({"seed": seed, **cost_decomposition(b_cost, costs)})
            if need_inc:
                inc_ckpt.append(
                    {
                        "seed": seed,
                        "family": family,
                        "wage_axis": "central",
                        "exchequer_cost_bn": (b["gov"] - m["gov"]) / 1e9,
                        "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - b["pov_bhc"]),
                        "poverty_change_ahc_pp": 100 * (m["pov_ahc"] - b["pov_ahc"]),
                        "relative_poverty_change_bhc_pp": 100 * (m["relpov_bhc"] - b["relpov_bhc"]),
                        "gini_change_pp": 100 * (m["gini"] - b["gini"]),
                        "displaced_weighted_m": float(w[displaced].sum() / 1e6),
                        "realised_displacement_rate": realised_displacement_rate(persons, displaced),
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
                pol_ckpt.append(
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
                simr = build_sim(base_arrays, table, reform=reforms_exposure[name])
                sr = hh_state(simr)
                del simr
                gc.collect()
                mr = metrics_from_state(sr, pw_by_hh)
                pol_ckpt.append(
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
    inc = inc_ckpt.read()
    expected = {(seed, family) for seed in range(N_DRAWS) for family in FAMILIES}
    observed = set(zip(inc["seed"], inc["family"]))
    if observed != expected or inc.duplicated(["seed", "family"]).any():
        raise ValueError("incidence checkpoint is incomplete, duplicated, or contains stale families")
    incidence_metrics = [
        "exchequer_cost_bn",
        "poverty_change_bhc_pp",
        "poverty_change_ahc_pp",
        "gini_change_pp",
    ]
    inc_summary = {
        fam: summarise(inc[inc.family == fam], incidence_metrics)
        for fam in FAMILIES
    }
    inc_summary["_metadata"] = {
        "certified_data_build_id": certified_data_build_id(),
        "draws": N_DRAWS,
        "seeds": [0, N_DRAWS - 1],
        "interval_interpretation": (
            "Quantiles, SDs and MCSEs describe assignment-draw variation only; "
            "they are not survey-sampling confidence intervals."
        ),
        "convergence_vs_20_draws": {
            fam: convergence_diagnostics(inc[inc.family == fam], incidence_metrics)
            for fam in FAMILIES
        },
    }
    (OUT / "incidence_monte_carlo.json").write_text(json.dumps(inc_summary, indent=2))

    pol = pol_ckpt.read()
    expected_pol = {
        (seed, reform, family)
        for seed in range(N_DRAWS)
        for family in FAMILIES
        for reform in (["R1_wage_insurance"] if family in FAMILIES_R1 else [])
        + (list(reforms_exposure) if family == "exposure" else [])
    }
    if set(zip(pol["seed"], pol["reform"], pol["family"])) != expected_pol:
        raise ValueError("policy checkpoint is incomplete or duplicated")
    policy_metrics = ["exchequer_cost_bn", "poverty_change_bhc_pp", "gini_change_pp"]
    pol_summary = {}
    for (reform, family), g in pol.groupby(["reform", "family"]):
        pol_summary[f"{reform}_{family}"] = summarise(g, policy_metrics)
    (OUT / "policy_monte_carlo.json").write_text(json.dumps(pol_summary, indent=2))

    # central = exposure family (identical mechanics to the central preset via
    # run_scenario); recompute central_monte_carlo.json with the relative-
    # poverty variant added
    central = inc[inc.family == "exposure"]
    central_summary = summarise(
        central,
        incidence_metrics
        + ["relative_poverty_change_bhc_pp", "displaced_weighted_m", "realised_displacement_rate"],
    )
    central_summary["baseline_relative_poverty_bhc_rate"] = b["relpov_bhc"]
    central_summary["prescribed_displacement_rate"] = PRESETS["central"].displacement_rate
    central_summary["certified_data_build_id"] = certified_data_build_id()
    (OUT / "central_monte_carlo.json").write_text(json.dumps(central_summary, indent=2))

    # exchequer cost by component, central family, all draws
    cost = cost_ckpt.read()
    if sorted(cost["seed"]) != list(range(N_DRAWS)):
        raise ValueError("cost-component checkpoint is incomplete or duplicated")
    cost_summary = summarise(cost, list(COST_FIELDS) + ["total"])
    gap = (cost["total"] - inc[inc.family == "exposure"].set_index("seed")
           .loc[cost["seed"], "exchequer_cost_bn"].to_numpy()).abs().max()
    if gap > 1e-6:
        raise AssertionError(f"cost decomposition total differs from incidence cost by {gap}")
    cost_summary["_metadata"] = {
        "certified_data_build_id": certified_data_build_id(),
        "draws": N_DRAWS,
        "family": "exposure (central preset)",
        "sign": "positive = cost to the Exchequer (fall in gov_balance)",
        "components": {
            "dividend_tax": "dividend_income_tax (part of income tax)",
            "savings_tax": "savings_income_tax (part of income tax)",
            "other_income_tax": "income_tax less dividend and savings income tax",
            "employee_ni": "national_insurance (class 1 employee, 2, 3, 4)",
            "employer_ni": "ni_employer (class 1 employer and 1A)",
            "universal_credit": "universal_credit",
            "childcare": "+".join(CHILDCARE_VARIABLES),
            "other": "gov_balance cost less the named components (residual)",
        },
        "narrow_cash": (
            "income_tax + national_insurance - household_benefits, the basis of "
            "the JR16 shock grid (results/jr16/grid.csv); it omits employer NI "
            "and the other gov_balance lines"
        ),
    }
    (OUT / "central_cost_decomposition.json").write_text(json.dumps(cost_summary, indent=2))
    print(json.dumps({"incidence": inc_summary, "policy": pol_summary,
                      "central": central_summary}, indent=1))


if __name__ == "__main__":
    main()
