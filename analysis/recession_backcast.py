"""Historical-benchmark validation: a 2008-09-style recession backcast.

The 2008-10 downturn raised the UK unemployment rate by ~2.6pp (LFS: ~5.2%
in early 2008 to ~7.9% by early 2010). This script pushes a displacement
shock of that size through the SAME machinery as the AI scenarios —
uniform incidence (no exposure gradient, as recession job loss was not
AI-exposure-graded), NO wage uplift, NO capital-return shock — and compares
the simulated poverty/inequality response with what HBAI actually recorded
over 2007/08 -> 2010/11.

Observed history (IFS/DWP HBAI): absolute BHC poverty was roughly flat to
slightly FALLING and the Gini roughly flat to slightly falling, because
benefits were uprated with (then-high) inflation while median earnings
fell. The framework here freezes the poverty line at its baseline level,
applies no benefit uprating, and models a pure job-loss impact year, so it
should OVERSTATE the poverty response relative to history; the gap
quantifies what uprating/behavioural/duration channels absorbed.

Outputs results/robustness/recession_backcast.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uk_ai_study.runner import build_person_table, gini
from uk_ai_study.shocks import ShockScenario, apply_shocks, build_shocked_simulation

DATA = Path("data")
OUT = Path("results/robustness")
PERIOD = 2026
SEEDS = range(5)

#: LFS UK unemployment rate rose ~2.6pp between 2008Q1 and 2010Q1.
RECESSION_DISPLACEMENT_RATE = 0.026

SCENARIO = ShockScenario(
    "recession_2008_style",
    displacement_rate=RECESSION_DISPLACEMENT_RATE,
    wage_uplift=0.0,  # no AI complementarity uplift in a recession backcast
    capital_return_increase=0.0,  # no capital-return shock
)

#: Observed HBAI outcomes, 2007/08 -> 2010/11 (DWP HBAI / IFS Living
#: Standards, Poverty and Inequality series; 2010/11 vintage). Stated as
#: percentage-point changes over the three-year window spanning the
#: recession. Absolute poverty here is the HBAI 2010/11-rebased fixed
#: threshold; relative is 60% of contemporary median.
OBSERVED_HBAI_2008_10 = {
    "absolute_poverty_bhc_pp": -1.0,  # ~18% -> ~17%: fell slightly
    "relative_poverty_bhc_pp": -2.0,  # ~18.3% -> ~16.2%: fell (median fell)
    "relative_poverty_ahc_pp": -1.0,  # ~22.5% -> ~21.3%
    "gini_pp": -1.0,  # BHC Gini ~0.36 -> ~0.35 by 2010/11 (fell 2009/10-2010/11)
    "note": (
        "DWP HBAI / IFS: absolute BHC poverty roughly flat to slightly "
        "falling over 2007/08-2010/11; relative poverty FELL because the "
        "median fell while benefits were uprated with high RPI inflation; "
        "Gini roughly flat through 2009/10 then fell in 2010/11."
    ),
}


def relative_poverty_rate(equiv_person: np.ndarray, pw: np.ndarray) -> float:
    """Headcount below 60% of the contemporaneous weighted median."""
    order = np.argsort(equiv_person)
    cw = np.cumsum(pw[order])
    median = equiv_person[order][np.searchsorted(cw, 0.5 * cw[-1])]
    return float(np.average(equiv_person < 0.6 * median, weights=pw))


def metrics(sim):
    pw = sim.calculate("person_weight", period=PERIOD, map_to="person").values
    hw = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    eq_hh = sim.calculate(
        "equiv_hbai_household_net_income", period=PERIOD, map_to="household"
    ).values
    eq_p = sim.calculate(
        "equiv_hbai_household_net_income", period=PERIOD, map_to="person"
    ).values
    n = sim.calculate("household_count_people", period=PERIOD, map_to="household").values
    return {
        # in_poverty_bhc/ahc use PolicyEngine's absolute (fixed) poverty
        # thresholds — they do NOT move with the shocked distribution, i.e.
        # a baseline-fixed poverty line, as in the paper's headline numbers.
        "pov_bhc": float(
            np.average(
                sim.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values,
                weights=pw,
            )
        ),
        "pov_ahc": float(
            np.average(
                sim.calculate("in_poverty_ahc", period=PERIOD, map_to="person").values,
                weights=pw,
            )
        ),
        # relative poverty: threshold recomputed within each simulation
        # (60% of contemporaneous median), mirroring HBAI relative poverty
        "pov_rel_bhc": relative_poverty_rate(eq_p, pw),
        "gini": gini(eq_hh, hw * n),
    }


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    baseline = Microsimulation(dataset=ds)
    persons = build_person_table(
        baseline,
        PERIOD,
        DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab",
    )
    b = metrics(baseline)

    # uniform incidence: flatten the exposure gradient before the eq 3.4
    # draw, exactly as the "uniform" incidence family does
    flat = persons.copy()
    flat["exposure"] = 1.0
    w = persons["weight"].to_numpy()

    draws = []
    for seed in SEEDS:
        shocked = apply_shocks(flat, SCENARIO, seed=seed)
        sim = build_shocked_simulation(ds, baseline, shocked, PERIOD)
        m = metrics(sim)
        displaced = shocked["displaced"].to_numpy()
        rec = {
            "seed": seed,
            "displaced_weighted_m": float(w[displaced].sum() / 1e6),
            "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - b["pov_bhc"]),
            "poverty_change_ahc_pp": 100 * (m["pov_ahc"] - b["pov_ahc"]),
            "relative_poverty_change_bhc_pp": 100 * (m["pov_rel_bhc"] - b["pov_rel_bhc"]),
            "gini_change_pp": 100 * (m["gini"] - b["gini"]),
        }
        draws.append(rec)
        print(rec, flush=True)

    keys = [k for k in draws[0] if k != "seed"]
    mean = {k: float(np.mean([d[k] for d in draws])) for k in keys}
    sd = {k: float(np.std([d[k] for d in draws], ddof=1)) for k in keys}

    result = {
        "scenario": {
            "name": SCENARIO.name,
            "displacement_rate": SCENARIO.displacement_rate,
            "incidence": "uniform (no exposure gradient)",
            "wage_uplift": SCENARIO.wage_uplift,
            "capital_return_increase": SCENARIO.capital_return_increase,
            "seeds": list(SEEDS),
        },
        "baseline_levels_pct": {
            "poverty_bhc": 100 * b["pov_bhc"],
            "poverty_ahc": 100 * b["pov_ahc"],
            "relative_poverty_bhc": 100 * b["pov_rel_bhc"],
            "gini": 100 * b["gini"],
        },
        "draws": draws,
        "mean": mean,
        "sd_across_seeds": sd,
        "per_pp_of_displacement": {
            k: mean[k] / (100 * RECESSION_DISPLACEMENT_RATE)
            for k in keys
            if k != "displaced_weighted_m"
        },
        "observed_hbai_2007_08_to_2010_11": OBSERVED_HBAI_2008_10,
        "interpretation": (
            "The framework is an impact-effect calculator: it models the "
            "first-year mechanical income loss from displacement with the "
            "poverty line frozen, no benefit uprating, no unemployment-"
            "duration dynamics and no behavioural response. History over "
            "2008-10 combined a similar-sized employment shock with "
            "inflation-uprated benefits and a falling median, so measured "
            "poverty and inequality did not rise. The gap between the "
            "simulated and observed changes quantifies what those absorbing "
            "channels delivered."
        ),
    }
    (OUT / "recession_backcast.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"mean": mean, "per_pp": result["per_pp_of_displacement"]}, indent=2))


if __name__ == "__main__":
    main()
