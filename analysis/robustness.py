"""Referee-demanded robustness: Monte Carlo CIs, exposure-measure
sensitivity, uniform-shock comparator, cross-sectional age x exposure
moments, and FRS-vs-ASHE occupation validation.

Outputs in results/robustness/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_ai_study.exposure import (
    attach_soc_major_group,
    exposure_for_major_group,
    load_major_group_exposure,
)
from uk_ai_study.runner import run_scenario
from uk_ai_study.shocks import PRESETS

DATA = Path("data")
OUT = Path("results/robustness")
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
H5 = DATA / "frs_2024_25.h5"
N_DRAWS = 20


def monte_carlo_central():
    rows = []
    for seed in range(N_DRAWS):
        r = run_scenario(H5, ADULT, "central", seed=seed)
        rows.append(
            {
                "seed": seed,
                "exchequer_cost_bn": r.exchequer_cost / 1e9,
                "poverty_change_bhc_pp": 100 * r.poverty_rate_change_bhc,
                "gini_change_pp": 100 * (r.gini_shocked - r.gini_baseline),
            }
        )
        print(f"draw {seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "central_draws.csv", index=False)
    summary = {
        c: {"mean": float(df[c].mean()), "sd": float(df[c].std(ddof=1)),
            "min": float(df[c].min()), "max": float(df[c].max())}
        for c in df.columns if c != "seed"
    }
    (OUT / "central_monte_carlo.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


def exposure_sensitivity():
    """Central scenario under alternative exposure measures.

    run_scenario derives exposure internally from c_aioe; here we monkeypatch
    the measure by passing precomputed columns through a custom runner call is
    overkill — instead we rerun with exposure arrays via runner internals.
    Simplest valid route: temporarily swap the measure name used by
    exposure_for_major_group calls in a local copy of the pipeline.
    """
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset
    from uk_ai_study.runner import AGE_BANDS, gini
    from uk_ai_study.shocks import apply_shocks, build_shocked_simulation

    dataset = UKSingleYearDataset(file_path=str(H5))
    results = {}
    for measure in ("c_aioe", "dsit_aioe", "eloundou_beta"):
        baseline = Microsimulation(dataset=dataset)
        calc = lambda v: baseline.calculate(v, period=2026, map_to="person").values
        persons = pd.DataFrame(
            {
                "person_id": calc("person_id"),
                "age": calc("age"),
                "employment_income": calc("employment_income"),
                "savings_interest_income": calc("savings_interest_income"),
                "dividend_income": calc("dividend_income"),
                "weight": calc("person_weight"),
            }
        )
        persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
        e = exposure_for_major_group(persons["soc_major_group"], measure)
        th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
        persons["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
        persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))

        shocked_table = apply_shocks(persons, PRESETS["central"], seed=0)
        sim = build_shocked_simulation(dataset, baseline, shocked_table, 2026)

        def m(s):
            hw = s.calculate("household_weight", period=2026, map_to="household").values
            eq = s.calculate("equiv_hbai_household_net_income", period=2026, map_to="household").values
            n = s.calculate("household_count_people", period=2026, map_to="household").values
            pw = s.calculate("person_weight", period=2026, map_to="person").values
            return {
                "gov": float((s.calculate("gov_balance", period=2026, map_to="household").values * hw).sum()),
                "pov": float(np.average(s.calculate("in_poverty_bhc", period=2026, map_to="person").values, weights=pw)),
                "gini": gini(eq, hw * n),
            }

        b, sh = m(baseline), m(sim)
        # decile gradient of displacement (top vs bottom decile share)
        equiv = baseline.calculate("equiv_hbai_household_net_income", period=2026, map_to="person").values
        w = persons["weight"].to_numpy()
        order = np.argsort(equiv)
        cw = np.cumsum(w[order]); ranks = np.empty(len(equiv)); ranks[order] = cw / cw[-1]
        dec = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
        disp = shocked_table["displaced"].to_numpy()
        results[measure] = {
            "exchequer_cost_bn": (b["gov"] - sh["gov"]) / 1e9,
            "poverty_change_bhc_pp": 100 * (sh["pov"] - b["pov"]),
            "gini_change_pp": 100 * (sh["gini"] - b["gini"]),
            "transition_share_decile1_pct": float(100 * w[(dec == 1) & disp].sum() / w[dec == 1].sum()),
            "transition_share_decile10_pct": float(100 * w[(dec == 10) & disp].sum() / w[dec == 10].sum()),
        }
        print(measure, results[measure], flush=True)
    (OUT / "exposure_sensitivity.json").write_text(json.dumps(results, indent=2))


def uniform_comparator():
    """Same aggregate shocks, allocated uniformly (JR16 sec 4.2.1)."""
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset
    from uk_ai_study.runner import gini
    from uk_ai_study.shocks import apply_shocks, build_shocked_simulation

    dataset = UKSingleYearDataset(file_path=str(H5))
    baseline = Microsimulation(dataset=dataset)
    calc = lambda v: baseline.calculate(v, period=2026, map_to="person").values
    persons = pd.DataFrame(
        {
            "person_id": calc("person_id"),
            "age": calc("age"),
            "employment_income": calc("employment_income"),
            "savings_interest_income": calc("savings_interest_income"),
            "dividend_income": calc("dividend_income"),
            "weight": calc("person_weight"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
    # uniform: constant exposure and theta for everyone with a SOC match
    persons["exposure"] = 1.0
    persons["complementarity"] = 1.0

    shocked_table = apply_shocks(persons, PRESETS["central"], seed=0)
    sim = build_shocked_simulation(dataset, baseline, shocked_table, 2026)

    def m(s):
        hw = s.calculate("household_weight", period=2026, map_to="household").values
        eq = s.calculate("equiv_hbai_household_net_income", period=2026, map_to="household").values
        n = s.calculate("household_count_people", period=2026, map_to="household").values
        pw = s.calculate("person_weight", period=2026, map_to="person").values
        return {
            "gov": float((s.calculate("gov_balance", period=2026, map_to="household").values * hw).sum()),
            "pov": float(np.average(s.calculate("in_poverty_bhc", period=2026, map_to="person").values, weights=pw)),
            "gini": gini(eq, hw * n),
        }

    b, sh = m(baseline), m(sim)
    out = {
        "exchequer_cost_bn": (b["gov"] - sh["gov"]) / 1e9,
        "poverty_change_bhc_pp": 100 * (sh["pov"] - b["pov"]),
        "gini_change_pp": 100 * (sh["gini"] - b["gini"]),
    }
    (OUT / "uniform_comparator.json").write_text(json.dumps(out, indent=2))
    print("uniform:", out)


def cross_sectional_moments():
    """Static UK analogues of the Canaries/H&L descriptives: employment and
    earnings by age band within exposure tertiles (cross-section only)."""
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    dataset = UKSingleYearDataset(file_path=str(H5))
    sim = Microsimulation(dataset=dataset)
    calc = lambda v: sim.calculate(v, period=2026, map_to="person").values
    persons = pd.DataFrame(
        {"person_id": calc("person_id"), "age": calc("age"),
         "employment_income": calc("employment_income"), "weight": calc("person_weight")}
    )
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    persons["exposure"] = e
    matched = (persons["employment_income"] > 0) & np.isfinite(e)
    w = persons["weight"].to_numpy()

    # employment-weighted exposure tertiles over matched employees
    vals = persons["exposure"].to_numpy()
    order = np.argsort(vals[matched])
    cw = np.cumsum(w[matched.to_numpy()][order])
    t1 = vals[matched][order][np.searchsorted(cw, cw[-1] / 3)]
    t2 = vals[matched][order][np.searchsorted(cw, 2 * cw[-1] / 3)]
    tertile = np.where(vals <= t1, 1, np.where(vals <= t2, 2, 3))

    bands = [(16, 24), (25, 34), (35, 44), (45, 54), (55, 64)]
    rows = []
    for lo, hi in bands:
        am = (persons["age"] >= lo) & (persons["age"] <= hi)
        for t in (1, 2, 3):
            m = matched & am & (tertile == t)
            rows.append(
                {"age_band": f"{lo}-{hi}", "exposure_tertile": t,
                 "employment_weighted_m": float(w[m].sum() / 1e6),
                 "mean_earnings": float(np.average(persons["employment_income"][m], weights=w[m])) if m.any() else 0.0,
                 "share_of_band_employment": float(w[m].sum() / w[matched & am].sum())}
            )
    pd.DataFrame(rows).to_csv(OUT / "age_exposure_moments.csv", index=False)
    print("moments written")


def validation_vs_ashe():
    """FRS major-group employment distribution vs ASHE 2025 jobs weights."""
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    dataset = UKSingleYearDataset(file_path=str(H5))
    sim = Microsimulation(dataset=dataset)
    calc = lambda v: sim.calculate(v, period=2026, map_to="person").values
    pid = pd.Series(calc("person_id"))
    empinc = calc("employment_income")
    w = calc("person_weight")
    soc = attach_soc_major_group(pid, ADULT)
    matched = (empinc > 0) & np.isfinite(soc)
    frs = pd.Series(w[matched]).groupby((soc[matched] / 1000).astype(int)).sum()
    frs_share = frs / frs.sum()
    ashe = load_major_group_exposure()["employment_jobs_thousands"]
    ashe_share = ashe / ashe.sum()
    out = pd.DataFrame({"frs_share": frs_share, "ashe_2025_share": ashe_share})
    out["diff_pp"] = 100 * (out["frs_share"] - out["ashe_2025_share"])
    out.to_csv(OUT / "frs_vs_ashe_occupation.csv")
    print(out.round(3))


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task in ("all", "fast"):
        cross_sectional_moments()
        validation_vs_ashe()
        uniform_comparator()
        exposure_sensitivity()
    if task in ("all", "mc"):
        monte_carlo_central()
