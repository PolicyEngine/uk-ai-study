"""Workstream 2: the labour->profits tax-composition channel (OBR FRS-2026
Box 4.1 style pricing).

Static accounting layered ON TOP of the JR16 microsimulation grid. In each
displacement scenario a wage bill W_lost disappears from labour taxation.
Composition counterfactual: a fraction phi of W_lost reappears as corporate
profits taxed at the UK corporation tax main rate (25%), instead of being
taxed as labour at the ACTUAL average effective IT+NICs rate on the
displaced workers' earnings.

Effective labour rate approximation (documented): person-level income_tax +
national_insurance are attributed to employment income in proportion to the
person's employment-income share of their total gross income (employment +
self-employment + private pension + savings interest + dividends). This is a
pro-rata average-rate attribution, not a marginal calculation; it ignores
the ordering rules of the UK schedule (savings/dividend bands sit on top of
earned income), so it slightly OVERSTATES the tax attributable to labour for
people with large capital incomes. Employee NIC is levied on earnings only,
so pro-rata attribution is exact for NIC whenever employment income is the
only earned income.

Second channel: a share of the recycled profits flows back to households as
dividends. Dividends are paid out of POST-corporation-tax profit, so the
recycled amount is W_lost * phi * (1 - CT_MAIN_RATE) * payout, distributed
pro-rata over the observed household dividend distribution (payout ratio
treated as a parameter, 0.5 — roughly the long-run FTSE dividend payout
ratio; no UK-specific AI-firm evidence, so it is a parameter, not an
estimate). ONE extra PolicyEngine simulation prices the phi=0.5 case
against the no-recycling central scenario.

Caveats printed into the outputs: static accounting; incidence of CT (on
wages, prices, or shareholders) NOT modelled; no behavioural response; CT
base assumed equal to the accounting profit (no capital allowances, losses,
or profit shifting).

Outputs (results/tax_composition/):
  composition_grid.csv       - u x phi grid of revenue accounting
  recycling_case.json        - phi=0.5 dividend-recycling simulation
  revenue_shortfall_phi.png  - net revenue shortfall (GBP bn) vs phi

Usage: python analysis/tax_composition.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import figstyle
from replicate_jr16 import PERIOD, build_person_table
from uk_ai_study.runner import gini
from uk_ai_study.shocks import (
    PRESETS,
    ShockScenario,
    apply_shocks,
    build_shocked_simulation,
    draw_displaced,
)

OUT = Path("results/tax_composition")
CT_MAIN_RATE = 0.25
PHIS = [0.25, 0.5, 0.75, 1.0]
PAYOUT_RATIO = 0.5  # parameter, see module docstring
SEED = 0  # same seed as the jr16 grid, so draws line up cell-for-cell

INCOME_COMPONENTS = (
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "savings_interest_income",
    "dividend_income",
)


def labour_tax_attribution(baseline_sim, persons):
    """Person-level IT+NICs attributable to employment income (pro-rata)."""

    def calc(v):
        return baseline_sim.calculate(v, period=PERIOD, map_to="person").values.astype(float)

    gross = sum(np.clip(calc(v), 0, None) for v in INCOME_COMPONENTS)
    emp = np.clip(calc("employment_income"), 0, None)
    share = np.divide(emp, gross, out=np.zeros_like(emp), where=gross > 0)
    it_ni = calc("income_tax") + calc("national_insurance")
    return it_ni * share  # per-person GBP attributable to labour


def composition_grid(persons, labour_tax, grid_path):
    """u x phi accounting grid; labour-side net revenue comes from the
    existing jr16 grid (w=0 column) where available."""
    weight = persons["weight"].to_numpy()
    emp = persons["employment_income"].to_numpy()

    jr16 = None
    if grid_path.exists():
        g = pd.read_csv(grid_path)
        jr16 = g[g["wage_pct"] == 0].set_index("unemployment_pct")[
            "net_revenue_change_bn"
        ]

    rows = []
    for u in range(0, 11):
        if u == 0:
            w_lost = 0.0
            t_lab = np.nan
            lab_tax_lost = 0.0
        else:
            scenario = ShockScenario(f"u{u}", u / 100, 0.0)
            displaced = draw_displaced(persons, scenario, seed=SEED)
            w_lost = float((emp * weight)[displaced].sum())
            lab_tax_lost = float((labour_tax * weight)[displaced].sum())
            t_lab = lab_tax_lost / w_lost
        # net revenue change (IT+NI-benefits) from the microsimulation,
        # displacement-only column (wage uplift 0), same seed
        net_rev_bn = float(jr16.loc[u]) if jr16 is not None and u in jr16.index else np.nan
        for phi in PHIS:
            ct = phi * CT_MAIN_RATE * w_lost
            rows.append(
                {
                    "unemployment_pct": u,
                    "phi": phi,
                    "w_lost_bn": w_lost / 1e9,
                    "effective_labour_tax_rate": t_lab,
                    "labour_tax_lost_bn": lab_tax_lost / 1e9,
                    "ct_recouped_bn": ct / 1e9,
                    # microsim labour-side net revenue change plus the CT
                    # recoup; negative = shortfall
                    "net_revenue_change_with_ct_bn": net_rev_bn + ct / 1e9,
                    "microsim_net_revenue_change_bn": net_rev_bn,
                    # pure composition delta: taxing phi*W_lost at 25% CT
                    # instead of the whole W_lost at the effective labour rate
                    "delta_vs_full_labour_taxation_bn": (phi * CT_MAIN_RATE * w_lost - lab_tax_lost) / 1e9,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "composition_grid.csv", index=False)
    return out


def shortfall_figure(grid):
    figstyle.apply_style()
    fig, ax = __import__("matplotlib.pyplot", fromlist=["plt"]).subplots(
        figsize=figstyle.SINGLE
    )
    us = [1, 3, 5, 7, 10]
    for u, colour in zip(us, figstyle.SERIES):
        sub = grid[grid["unemployment_pct"] == u].sort_values("phi")
        shortfall = -sub["net_revenue_change_with_ct_bn"]
        ax.plot(sub["phi"], shortfall, marker="o", color=colour,
                label=f"{u}% displacement")
    ax.set_xlabel(
        r"$\phi$: share of displaced wage bill returning as taxed corporate profit"
    )
    ax.set_ylabel("Net revenue shortfall (£bn per year)")
    ax.set_xticks(PHIS)
    ax.axhline(0, color=figstyle.BASELINE, linewidth=0.8)
    figstyle.legend_below(ax, ncol=3)
    figstyle.save(fig, OUT / "revenue_shortfall_phi.png")


def hh_metrics(sim):
    hw = sim.calculate("household_weight", period=PERIOD, map_to="household").values
    eq = sim.calculate("equiv_hbai_household_net_income", period=PERIOD, map_to="household").values
    n = sim.calculate("household_count_people", period=PERIOD, map_to="household").values
    pw = sim.calculate("person_weight", period=PERIOD, map_to="person").values
    return {
        "gini": gini(eq, hw * n),
        "poverty_bhc": float(np.average(
            sim.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values, weights=pw)),
        "poverty_ahc": float(np.average(
            sim.calculate("in_poverty_ahc", period=PERIOD, map_to="person").values, weights=pw)),
        "hni_person": sim.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values,
    }


def recycling_case(dataset, baseline_sim, persons, labour_tax):
    """phi=0.5 dividend-recycling simulation vs the no-recycling central
    scenario. ONE extra Microsimulation (run sequentially, freed after use)."""
    scenario = PRESETS["central"]
    shocked_table = apply_shocks(persons, scenario, seed=SEED)
    displaced = shocked_table["displaced"].to_numpy()
    weight = persons["weight"].to_numpy()
    w_lost = float((persons["employment_income"].to_numpy() * weight)[displaced].sum())
    lab_tax_lost = float((labour_tax * weight)[displaced].sum())

    pw = persons["weight"].to_numpy()
    deciles = persons["decile"].to_numpy()

    # no-recycling central scenario
    sim = build_shocked_simulation(dataset, baseline_sim, shocked_table, PERIOD)
    central = hh_metrics(sim)
    del sim

    # recycling: distribute W_lost * phi * (1 - CT) * payout pro-rata over
    # the observed dividend distribution (baseline shares == shocked shares,
    # since the capital shock scales every dividend by the same factor).
    # Dividends come out of post-corporation-tax profit (R2-8): CT at the
    # 25% main rate is charged on phi*W_lost, and only the after-CT residual
    # is available for distribution.
    phi = 0.5
    recycled = w_lost * phi * (1 - CT_MAIN_RATE) * PAYOUT_RATIO
    div = shocked_table["dividend_income"].to_numpy(dtype=float)
    agg_div = float((div * weight).sum())
    recycled_table = shocked_table.copy()
    recycled_table["dividend_income"] = div * (1 + recycled / agg_div)
    sim = build_shocked_simulation(dataset, baseline_sim, recycled_table, PERIOD)
    recyc = hh_metrics(sim)
    del sim

    base = hh_metrics(baseline_sim)

    def decile_means(m):
        return {
            int(d): float(np.average(m["hni_person"][deciles == d], weights=pw[deciles == d]))
            for d in range(1, 11)
        }

    dm_base, dm_c, dm_r = decile_means(base), decile_means(central), decile_means(recyc)
    result = {
        "scenario": "central (7% displacement, +2.6% wages, +0.4pp capital return), seed 0",
        "phi": phi,
        "payout_ratio": PAYOUT_RATIO,
        "displaced_wage_bill_bn": w_lost / 1e9,
        "effective_labour_tax_rate_on_displaced": lab_tax_lost / w_lost,
        "recycled_dividends_bn": recycled / 1e9,
        "ct_on_recycled_profits_bn": phi * CT_MAIN_RATE * w_lost / 1e9,
        "baseline": {"gini": base["gini"], "poverty_bhc": base["poverty_bhc"], "poverty_ahc": base["poverty_ahc"]},
        "central_no_recycling": {k: central[k] for k in ("gini", "poverty_bhc", "poverty_ahc")},
        "recycling_phi05": {k: recyc[k] for k in ("gini", "poverty_bhc", "poverty_ahc")},
        "recycling_minus_central": {
            "gini_pp": 100 * (recyc["gini"] - central["gini"]),
            "poverty_bhc_pp": 100 * (recyc["poverty_bhc"] - central["poverty_bhc"]),
            "poverty_ahc_pp": 100 * (recyc["poverty_ahc"] - central["poverty_ahc"]),
        },
        "decile_mean_hbai_net_income": {
            "baseline": dm_base,
            "central_no_recycling": dm_c,
            "recycling_phi05": dm_r,
            "recycling_minus_central_gbp": {d: dm_r[d] - dm_c[d] for d in dm_c},
        },
        "caveats": [
            "Static accounting on top of the microsimulation; incidence of corporation tax not modelled.",
            "Effective labour tax rate is a pro-rata attribution of IT+NICs to employment income; ignores schedule ordering of savings/dividend bands.",
            "CT base assumed equal to accounting profit (no allowances, losses or profit shifting).",
            "Payout ratio 0.5 is a parameter, not an estimate; recycled dividends are paid out of post-corporation-tax profit (x(1-0.25)) and allocated pro-rata to existing dividend holders only.",
            "Dividend tax on the recycled dividends IS captured by the simulation; CT is added outside it.",
        ],
    }
    (OUT / "recycling_case.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset, baseline_sim, persons = build_person_table()
    labour_tax = labour_tax_attribution(baseline_sim, persons)

    grid = composition_grid(persons, labour_tax, Path("results/jr16/grid.csv"))
    shortfall_figure(grid)

    result = recycling_case(dataset, baseline_sim, persons, labour_tax)

    c = grid[(grid["unemployment_pct"] == 7) & (grid["phi"] == 0.5)].iloc[0]
    print(json.dumps({
        "central_w_lost_bn": c["w_lost_bn"],
        "central_effective_labour_tax_rate": c["effective_labour_tax_rate"],
        "central_labour_tax_lost_bn": c["labour_tax_lost_bn"],
        "central_phi05_ct_recouped_bn": c["ct_recouped_bn"],
        "central_phi05_delta_vs_full_labour_bn": c["delta_vs_full_labour_taxation_bn"],
        "recycling_minus_central": result["recycling_minus_central"],
    }, indent=2))


if __name__ == "__main__":
    main()
