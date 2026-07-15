"""Workstream 5B: central scenario under EVERY available exposure index.

Published "high exposure" employment shares range from ~13% to ~46% across
these indices; this script tests whether the paper's three qualitative
conclusions are index-robust:
  1. Gini rises;
  2. the displacement (transition) gradient rises with income
     (decile-10 share > decile-1 share);
  3. exchequer cost within +/-20% of the central (c_aioe) run.

Outputs results/robustness/index_sensitivity_full.json + index_sensitivity.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.runner import gini
from uk_ai_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

DATA = Path("data")
OUT = Path("results/robustness")
PERIOD = 2026
SEED = 0
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
MEASURES = ("c_aioe", "felten_aioe", "eloundou_beta", "dsit_aioe", "dsit_llm")
CENTRAL_MEASURE = "c_aioe"


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    baseline = Microsimulation(dataset=ds)
    calc = lambda v: baseline.calculate(v, period=PERIOD, map_to="person").values
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
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))

    def m(s):
        hw = s.calculate("household_weight", period=PERIOD, map_to="household").values
        eq = s.calculate("equiv_hbai_household_net_income", period=PERIOD, map_to="household").values
        n = s.calculate("household_count_people", period=PERIOD, map_to="household").values
        pw = s.calculate("person_weight", period=PERIOD, map_to="person").values
        return {
            "gov": float((s.calculate("gov_balance", period=PERIOD, map_to="household").values * hw).sum()),
            "pov": float(np.average(s.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values, weights=pw)),
            "gini": gini(eq, hw * n),
        }

    b = m(baseline)

    equiv = calc("equiv_hbai_household_net_income")
    w = persons["weight"].to_numpy()
    order = np.argsort(equiv)
    cw = np.cumsum(w[order])
    ranks = np.empty(len(equiv))
    ranks[order] = cw / cw[-1]
    dec = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)

    results = {}
    for measure in MEASURES:
        e = exposure_for_major_group(persons["soc_major_group"], measure)
        p = persons.copy()
        p["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
        table = apply_shocks(p, PRESETS["central"], seed=SEED)
        sim = build_shocked_simulation(ds, baseline, table, PERIOD)
        sh = m(sim)
        disp = table["displaced"].to_numpy()
        results[measure] = {
            "exchequer_cost_bn": (b["gov"] - sh["gov"]) / 1e9,
            "poverty_change_bhc_pp": 100 * (sh["pov"] - b["pov"]),
            "gini_change_pp": 100 * (sh["gini"] - b["gini"]),
            "transition_share_decile1_pct": float(100 * w[(dec == 1) & disp].sum() / w[dec == 1].sum()),
            "transition_share_decile10_pct": float(100 * w[(dec == 10) & disp].sum() / w[dec == 10].sum()),
            "displaced_weighted_m": float(w[disp].sum() / 1e6),
        }
        print(measure, {k: round(v, 3) for k, v in results[measure].items()}, flush=True)
        del sim, table

    central_cost = results[CENTRAL_MEASURE]["exchequer_cost_bn"]
    for measure, r in results.items():
        r["conclusions"] = {
            "gini_rises": r["gini_change_pp"] > 0,
            "transition_gradient_rises_with_income": (
                r["transition_share_decile10_pct"] > r["transition_share_decile1_pct"]
            ),
            "fiscal_cost_within_20pct_of_central": (
                abs(r["exchequer_cost_bn"] / central_cost - 1) <= 0.20
            ),
        }
        r["all_three_survive"] = all(r["conclusions"].values())

    payload = {
        "central_measure": CENTRAL_MEASURE,
        "seed": SEED,
        "scenario": "central (7% displacement, +2.6% wage, +0.4pp capital)",
        "results": results,
    }
    (OUT / "index_sensitivity_full.json").write_text(json.dumps(payload, indent=2))
    figure(results)


def figure(results):
    import figstyle as fs

    fs.apply_style()
    import matplotlib.pyplot as plt

    measures = list(results)
    x = np.arange(len(measures))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fs.TWOPANEL)
    ax1.bar(x, [results[k]["exchequer_cost_bn"] for k in measures], color=fs.BLUE)
    central = results[CENTRAL_MEASURE]["exchequer_cost_bn"]
    ax1.axhspan(central * 0.8, central * 1.2, color=fs.GRID, alpha=0.6, zorder=0,
                label="+/-20% of central")
    ax1.set_xticks(x, measures, rotation=20)
    ax1.set_ylabel("Exchequer cost (£bn/yr)")
    ax1.grid(axis="x", visible=False)
    fs.legend_below(ax1, 1)
    ax2.bar(x - 0.2, [results[k]["transition_share_decile1_pct"] for k in measures],
            width=0.4, color=fs.AQUA, label="Decile 1")
    ax2.bar(x + 0.2, [results[k]["transition_share_decile10_pct"] for k in measures],
            width=0.4, color=fs.BLUE, label="Decile 10")
    ax2.set_xticks(x, measures, rotation=20)
    ax2.set_ylabel("Displacement transition share (%)")
    ax2.grid(axis="x", visible=False)
    fs.legend_below(ax2, 2)
    fig.suptitle("Central scenario under alternative AI-exposure indices", fontsize=11)
    fs.save(fig, OUT / "index_sensitivity.png")


if __name__ == "__main__":
    main()
