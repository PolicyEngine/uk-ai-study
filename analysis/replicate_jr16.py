"""Regenerate the full JR16 result set (Figs 4.1-4.7) for the UK.

Outputs (results/jr16/):
  fig4_1_transition_by_decile.csv   - % of population transitioning to
                                      unemployment by decile of equivalised
                                      household disposable income (50 draws,
                                      95% CI)
  fig4_2_wage_gain_by_decile.csv    - % change in aggregate employment income
                                      by decile, non-transitioning population
  fig4_3_capital_by_decile.csv      - % change in aggregate capital income by
                                      decile
  fig4_4_decomposition.csv          - central scenario: change in market
                                      income / benefits / tax & NI /
                                      disposable income by decile, as % of
                                      baseline disposable income
  grid.csv                          - 11x6 grid (0-10% employment x 0-5%
                                      wage, capital +0.4pp always on): mean %
                                      change in household disposable income,
                                      % change in net Exchequer revenue
                                      (income tax + NI - benefits), Gini pp
                                      change

Usage:
  python analysis/replicate_jr16.py figs   # 4.1-4.4 (fast)
  python analysis/replicate_jr16.py grid   # 66 simulations (slow)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.shocks import (
    PRESETS,
    ShockScenario,
    apply_shocks,
    build_shocked_simulation,
    draw_displaced,
)
from uk_ai_study.runner import gini

DATA = Path("data")
OUT = Path("results/jr16")
PERIOD = 2026
N_DRAWS = 50


def build_person_table():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    dataset = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    sim = Microsimulation(dataset=dataset)

    def calc(v, entity="person"):
        return sim.calculate(v, period=PERIOD, map_to=entity).values

    persons = pd.DataFrame(
        {
            "person_id": calc("person_id"),
            "household_id": calc("household_id"),
            "age": calc("age"),
            "employment_income": calc("employment_income"),
            "savings_interest_income": calc("savings_interest_income"),
            "dividend_income": calc("dividend_income"),
            "weight": calc("person_weight"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(
        persons["person_id"], DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
    )
    exposure = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    theta = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(exposure), exposure, np.nanmean(exposure))
    persons["complementarity"] = np.where(np.isfinite(theta), theta, np.nanmean(theta))

    # deciles of equivalised household disposable income (HBAI concept,
    # person-level, weighted), fixed at baseline as in JR16 — #1, finding 2
    equiv = calc("equiv_hbai_household_net_income")
    order = np.argsort(equiv)
    cw = np.cumsum(persons["weight"].to_numpy()[order])
    ranks = np.empty(len(equiv), dtype=float)
    ranks[order] = cw / cw[-1]
    persons["decile"] = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
    return dataset, sim, persons


def figs_4_1_to_4_3(persons):
    scenario = PRESETS["central"]
    weight = persons["weight"].to_numpy()
    employment = persons["employment_income"].to_numpy()
    deciles = persons["decile"].to_numpy()

    # Fig 4.1: transition shares over N_DRAWS draws
    shares = np.zeros((N_DRAWS, 10))
    for s in range(N_DRAWS):
        displaced = draw_displaced(persons, scenario, seed=s)
        for d in range(1, 11):
            mask = deciles == d
            shares[s, d - 1] = weight[mask & displaced].sum() / weight[mask].sum()
    mean, se = shares.mean(0), shares.std(0, ddof=1) / np.sqrt(N_DRAWS)
    pd.DataFrame(
        {"decile": range(1, 11), "share_transitioning": mean,
         "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}
    ).to_csv(OUT / "fig4_1_transition_by_decile.csv", index=False)

    # Fig 4.2: % change in aggregate employment income, non-transitioning
    gains = np.zeros((N_DRAWS, 10))
    for s in range(N_DRAWS):
        shocked = apply_shocks(persons, scenario, seed=s)
        keep = ~shocked["displaced"].to_numpy()
        new_emp = shocked["employment_income"].to_numpy()
        for d in range(1, 11):
            mask = (deciles == d) & keep & (employment > 0)
            base = (employment[mask] * weight[mask]).sum()
            gains[s, d - 1] = ((new_emp[mask] - employment[mask]) * weight[mask]).sum() / base
    mean, se = gains.mean(0), gains.std(0, ddof=1) / np.sqrt(N_DRAWS)
    pd.DataFrame(
        {"decile": range(1, 11), "pct_change_employment_income": mean,
         "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}
    ).to_csv(OUT / "fig4_2_wage_gain_by_decile.csv", index=False)

    # Fig 4.3: % change in aggregate capital income (deterministic)
    capital = (persons["savings_interest_income"] + persons["dividend_income"]).to_numpy()
    factor = (0.01005 + scenario.capital_return_increase) / 0.01005
    rows = []
    for d in range(1, 11):
        mask = deciles == d
        base = (capital[mask] * weight[mask]).sum()
        rows.append({"decile": d, "pct_change_capital_income": (factor - 1) if base > 0 else 0.0,
                     "aggregate_capital_income": base})
    # NOTE: a uniform return shock is a uniform % increase for every recipient;
    # JR16's Fig 4.3 gradient comes from recipiency shares. Report both.
    total = sum(r["aggregate_capital_income"] for r in rows)
    for r in rows:
        r["share_of_all_capital_income"] = r["aggregate_capital_income"] / total
    pd.DataFrame(rows).to_csv(OUT / "fig4_3_capital_by_decile.csv", index=False)


def _household_frame(sim, period=PERIOD):
    """Component totals per HOUSEHOLD (person components aggregated up), so
    every column shares one weighting base — #1, finding 3."""

    def calc(v):
        return sim.calculate(v, period=period, map_to="household").values

    return pd.DataFrame(
        {
            "market": calc("employment_income") + calc("self_employment_income")
            + calc("private_pension_income") + calc("savings_interest_income")
            + calc("dividend_income"),
            "benefits": calc("hbai_benefits"),
            "tax": calc("income_tax") + calc("national_insurance"),
            "disposable": calc("hbai_household_net_income"),
        }
    )


def _household_deciles(sim, period=PERIOD):
    """Person-representative deciles at household level: households ranked by
    baseline equivalised HBAI income, weighted by household_weight x people."""
    equiv = sim.calculate("equiv_hbai_household_net_income", period=period, map_to="household").values
    hw = sim.calculate("household_weight", period=period, map_to="household").values
    n = sim.calculate("household_count_people", period=period, map_to="household").values
    w = hw * n
    order = np.argsort(equiv)
    cw = np.cumsum(w[order])
    ranks = np.empty(len(equiv), dtype=float)
    ranks[order] = cw / cw[-1]
    return np.clip(np.ceil(ranks * 10).astype(int), 1, 10), hw


def fig_4_4(dataset, baseline_sim, persons):
    shocked_table = apply_shocks(persons, PRESETS["central"], seed=0)
    shocked_sim = build_shocked_simulation(dataset, baseline_sim, shocked_table, PERIOD)

    base, shock = _household_frame(baseline_sim), _household_frame(shocked_sim)
    deciles, hw = _household_deciles(baseline_sim)
    rows = []
    for d in range(1, 11):
        m = deciles == d
        disp_base = (base["disposable"][m] * hw[m]).sum()
        row = {
            "decile": d,
            "market_income": ((shock["market"] - base["market"])[m] * hw[m]).sum() / disp_base,
            "benefits": ((shock["benefits"] - base["benefits"])[m] * hw[m]).sum() / disp_base,
            "tax_and_contributions": -((shock["tax"] - base["tax"])[m] * hw[m]).sum() / disp_base,
            "disposable_income": ((shock["disposable"] - base["disposable"])[m] * hw[m]).sum() / disp_base,
        }
        # explicit residual (perimeter items outside the three components:
        # e.g. council tax, pension deductions inside the HBAI concept)
        row["residual"] = row["disposable_income"] - (
            row["market_income"] + row["benefits"] + row["tax_and_contributions"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    # additivity: components + residual must equal disposable by construction
    gap = (out["market_income"] + out["benefits"] + out["tax_and_contributions"]
           + out["residual"] - out["disposable_income"]).abs().max()
    assert gap < 1e-12, f"decomposition not additive (max gap {gap})"
    out.to_csv(OUT / "fig4_4_decomposition.csv", index=False)
    print("fig 4.4 residual by decile (pp of baseline disposable):",
          (100 * out["residual"]).round(2).tolist())


GRID_COLUMNS = [
    "unemployment_pct",
    "wage_pct",
    "avg_disposable_income_change_pct",
    "net_revenue_change_bn",
    "net_revenue_change_pct_of_receipts",
    "gini_change_pp",
]


def run_grid(dataset, baseline_sim, persons, resume: bool = False):
    def hh_metrics(sim):
        hw = sim.calculate("household_weight", period=PERIOD, map_to="household").values
        hn = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values
        eq = sim.calculate("equiv_hbai_household_net_income", period=PERIOD, map_to="household").values
        np_ = sim.calculate("household_count_people", period=PERIOD, map_to="household").values
        pw = sim.calculate("person_weight", period=PERIOD, map_to="person").values
        it = (sim.calculate("income_tax", period=PERIOD, map_to="person").values * pw).sum()
        ni = (sim.calculate("national_insurance", period=PERIOD, map_to="person").values * pw).sum()
        bh = (sim.calculate("household_benefits", period=PERIOD, map_to="household").values * hw).sum()
        return {
            "mean_hni": float(np.average(hn, weights=hw)),
            "gini": gini(eq, hw * np_),
            "net_revenue": float(it + ni - bh),
            "receipts": float(it + ni),
        }

    base = hh_metrics(baseline_sim)
    out_path = OUT / "grid.csv"
    # resume is opt-in and fingerprint-aware: a stale-schema CSV is never
    # silently reused (#1, finding 8)
    done = set()
    rows = []
    if resume and out_path.exists():
        prev = pd.read_csv(out_path)
        if list(prev.columns) == GRID_COLUMNS:
            done = set(zip(prev["unemployment_pct"], prev["wage_pct"]))
            rows = prev.to_dict("records")
        else:
            print("existing grid.csv has a different schema; recomputing all cells")

    for u in range(0, 11):
        for w in range(0, 6):
            if (u, w) in done:
                continue
            scenario = ShockScenario(f"u{u}_w{w}", u / 100, w / 100)
            shocked_table = apply_shocks(persons, scenario, seed=0)
            sim = build_shocked_simulation(dataset, baseline_sim, shocked_table, PERIOD)
            m = hh_metrics(sim)
            delta_rev = m["net_revenue"] - base["net_revenue"]
            rows.append(
                {
                    "unemployment_pct": u,
                    "wage_pct": w,
                    "avg_disposable_income_change_pct": 100 * (m["mean_hni"] / base["mean_hni"] - 1),
                    # change in net revenue (IT + NI - household benefits) in
                    # £bn, and re-expressed against gross IT+NI receipts; the
                    # % change against the (negative) net base was sign-flipped
                    # and is no longer written — #1, findings 8/9
                    "net_revenue_change_bn": delta_rev / 1e9,
                    "net_revenue_change_pct_of_receipts": 100 * delta_rev / base["receipts"],
                    "gini_change_pp": 100 * (m["gini"] - base["gini"]),
                }
            )
            pd.DataFrame(rows)[GRID_COLUMNS].to_csv(out_path, index=False)
            print(f"u={u} w={w} done", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "figs"
    dataset, baseline_sim, persons = build_person_table()
    if mode == "figs":
        figs_4_1_to_4_3(persons)
        fig_4_4(dataset, baseline_sim, persons)
        print("figs 4.1-4.4 written")
    elif mode == "grid":
        run_grid(dataset, baseline_sim, persons, resume="--resume" in sys.argv)
        print("grid complete")


if __name__ == "__main__":
    main()
