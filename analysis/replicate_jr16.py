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
from uk_ai_study.shocks import PRESETS, ShockScenario, apply_shocks, draw_displaced
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

    # deciles of equivalised household disposable income (person-level,
    # weighted), fixed at baseline as in JR16
    equiv = calc("equiv_household_net_income")
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


def _decile_frame(sim, persons, period=PERIOD):
    def calc(v):
        return sim.calculate(v, period=period, map_to="person").values

    return pd.DataFrame(
        {
            "market": calc("employment_income") + calc("self_employment_income")
            + calc("private_pension_income") + calc("savings_interest_income")
            + calc("dividend_income"),
            "benefits": calc("household_benefits"),
            "tax": calc("income_tax") + calc("national_insurance"),
            "disposable": calc("household_net_income"),
        }
    )


def fig_4_4(dataset, baseline_sim, persons):
    from policyengine_uk import Microsimulation

    shocked_table = apply_shocks(persons, PRESETS["central"], seed=0)
    shocked_sim = Microsimulation(dataset=dataset)
    for col in ("employment_income", "savings_interest_income", "dividend_income"):
        shocked_sim.set_input(col, PERIOD, shocked_table[col].to_numpy(dtype=float))

    base, shock = _decile_frame(baseline_sim, persons), _decile_frame(shocked_sim, persons)
    weight = persons["weight"].to_numpy()
    deciles = persons["decile"].to_numpy()
    rows = []
    for d in range(1, 11):
        m = deciles == d
        disp_base = (base["disposable"][m] * weight[m]).sum()
        rows.append(
            {
                "decile": d,
                "market_income": ((shock["market"] - base["market"])[m] * weight[m]).sum() / disp_base,
                "benefits": ((shock["benefits"] - base["benefits"])[m] * weight[m]).sum() / disp_base,
                "tax_and_contributions": -((shock["tax"] - base["tax"])[m] * weight[m]).sum() / disp_base,
                "disposable_income": ((shock["disposable"] - base["disposable"])[m] * weight[m]).sum() / disp_base,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "fig4_4_decomposition.csv", index=False)


def run_grid(dataset, baseline_sim, persons):
    from policyengine_uk import Microsimulation

    def hh_metrics(sim):
        hw = sim.calculate("household_weight", period=PERIOD, map_to="household").values
        hn = sim.calculate("household_net_income", period=PERIOD, map_to="household").values
        eq = sim.calculate("equiv_household_net_income", period=PERIOD, map_to="household").values
        np_ = sim.calculate("household_count_people", period=PERIOD, map_to="household").values
        pw = sim.calculate("person_weight", period=PERIOD, map_to="person").values
        it = sim.calculate("income_tax", period=PERIOD, map_to="person").values
        ni = sim.calculate("national_insurance", period=PERIOD, map_to="person").values
        bh = sim.calculate("household_benefits", period=PERIOD, map_to="household").values
        net_rev = (it * pw).sum() + (ni * pw).sum() - (bh * hw).sum()
        return {
            "mean_hni": float(np.average(hn, weights=hw)),
            "gini": gini(eq, hw * np_),
            "net_revenue": float(net_rev),
        }

    base = hh_metrics(baseline_sim)
    out_path = OUT / "grid.csv"
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev["unemployment_pct"], prev["wage_pct"]))
    rows = [] if not done else pd.read_csv(out_path).to_dict("records")

    for u in range(0, 11):
        for w in range(0, 6):
            if (u, w) in done:
                continue
            scenario = ShockScenario(f"u{u}_w{w}", u / 100, w / 100)
            shocked_table = apply_shocks(persons, scenario, seed=0)
            sim = Microsimulation(dataset=dataset)
            for col in ("employment_income", "savings_interest_income", "dividend_income"):
                sim.set_input(col, PERIOD, shocked_table[col].to_numpy(dtype=float))
            m = hh_metrics(sim)
            rows.append(
                {
                    "unemployment_pct": u,
                    "wage_pct": w,
                    "avg_disposable_income_change_pct": 100 * (m["mean_hni"] / base["mean_hni"] - 1),
                    # WARNING: baseline net_revenue (IT + NI - benefits incl.
                    # state pension) is NEGATIVE, so this ratio is
                    # sign-flipped and misleading; the paper instead reports
                    # the change relative to gross IT+NI receipts (the
                    # *_of_receipts column in the committed grid.csv, whose
                    # generating revision is not yet committed — see
                    # uk-ai-study#1, findings 8/9).
                    "net_revenue_change_pct": 100 * (m["net_revenue"] / base["net_revenue"] - 1),
                    "gini_change_pp": 100 * (m["gini"] - base["gini"]),
                }
            )
            pd.DataFrame(rows).to_csv(out_path, index=False)
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
        run_grid(dataset, baseline_sim, persons)
        print("grid complete")


if __name__ == "__main__":
    main()
