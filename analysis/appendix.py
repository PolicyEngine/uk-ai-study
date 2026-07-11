"""JR16-appendix analogues for the UK (Figures B.1-B.11, Table 3.2).

Tasks:
  fast    - baseline decile distributions (B.1-B.5), exposure/complementarity
            incidence by decile (3.2/3.3), job-loss table by major group
            (Table 3.2), uniform-vs-AI decile figure (B.7), alternative-index
            decile figure (B.8)
  decomp  - 20-draw CIs on the fig 4.4 decomposition
  grids   - decile-faceted grid (B.9) and no-capital grid (B.11), 66 sims each

Outputs in results/appendix/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.shocks import PRESETS, ShockScenario, apply_shocks

DATA = Path("data")
OUT = Path("results/appendix")
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
H5 = DATA / "frs_2024_25.h5"
PERIOD = 2026
NAVY, RED, BLUE, GREEN, GOLD = "#1f3557", "#b03a3a", "#4a90c4", "#4a7d4a", "#e8a33d"


def setup():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    dataset = UKSingleYearDataset(file_path=str(H5))
    baseline = Microsimulation(dataset=dataset)

    def calc(v, entity="person"):
        return baseline.calculate(v, period=PERIOD, map_to=entity).values

    persons = pd.DataFrame(
        {
            "person_id": calc("person_id"),
            "age": calc("age"),
            "employment_income": calc("employment_income"),
            "self_employment_income": calc("self_employment_income"),
            "private_pension_income": calc("private_pension_income"),
            "savings_interest_income": calc("savings_interest_income"),
            "dividend_income": calc("dividend_income"),
            "income_tax": calc("income_tax"),
            "national_insurance": calc("national_insurance"),
            "household_benefits": calc("household_benefits"),
            "household_net_income": calc("household_net_income"),
            "weight": calc("person_weight"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], ADULT)
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
    persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))
    persons["exposure_raw"] = e

    equiv = calc("equiv_household_net_income")
    w = persons["weight"].to_numpy()
    order = np.argsort(equiv)
    cw = np.cumsum(w[order])
    ranks = np.empty(len(equiv))
    ranks[order] = cw / cw[-1]
    persons["decile"] = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
    return dataset, baseline, persons


def bar_by_decile(values_bn, title, ylabel, fname, color=NAVY, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(1, 11), values_bn, color=color)
    for x, y in zip(range(1, 11), values_bn):
        ax.text(x, y + max(values_bn) * 0.02, fmt.format(y), ha="center", fontsize=8)
    ax.set_xlabel("Decile of equivalised household disposable income")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(1, 11))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def fast(dataset, baseline, persons):
    w = persons["weight"].to_numpy()
    dec = persons["decile"].to_numpy()

    # --- B.1-B.5 baseline distributions (weighted aggregates, GBP bn/yr)
    comps = {
        "b1_market_income_less_capital": (
            persons["employment_income"] + persons["self_employment_income"]
            + persons["private_pension_income"], NAVY,
            "Aggregate market income less capital income"),
        "b2_capital_income": (
            persons["savings_interest_income"] + persons["dividend_income"], GOLD,
            "Aggregate capital income (interest + dividends)"),
        "b3_benefits": (persons["household_benefits"], BLUE, "Aggregate benefits"),
        "b4_tax_and_ni": (
            persons["income_tax"] + persons["national_insurance"], GREEN,
            "Aggregate income tax and National Insurance"),
        "b5_disposable_income": (
            persons["household_net_income"], NAVY, "Aggregate disposable income"),
    }
    rows = {}
    for name, (series, color, title) in comps.items():
        agg = np.array([float((series.to_numpy() * w)[dec == d].sum()) / 1e9 for d in range(1, 11)])
        rows[name] = agg
        bar_by_decile(agg, title + " — UK baseline, 2026", "£ billion per year",
                      f"{name}.png", color=color)
    pd.DataFrame(rows, index=range(1, 11)).rename_axis("decile").to_csv(OUT / "baseline_distributions.csv")

    # --- 3.2/3.3 analogues: exposure & complementarity incidence by decile
    matched = (persons["employment_income"] > 0) & np.isfinite(persons["exposure_raw"])
    for measure, fname, title in [
        ("exposure_raw", "exposure_incidence_by_decile.png",
         "Workers by AI-exposure tertile within each income decile"),
        ("complementarity", "complementarity_incidence_by_decile.png",
         "Workers by complementarity tertile within each income decile"),
    ]:
        vals = persons[measure].to_numpy()
        mv, mw = vals[matched], w[matched.to_numpy()]
        order = np.argsort(mv)
        cwm = np.cumsum(mw[order])
        t1 = mv[order][np.searchsorted(cwm, cwm[-1] / 3)]
        t2 = mv[order][np.searchsorted(cwm, 2 * cwm[-1] / 3)]
        tert = np.where(vals <= t1, 1, np.where(vals <= t2, 2, 3))
        shares = np.zeros((3, 10))
        for d in range(1, 11):
            m = matched & (dec == d)
            tot = w[m].sum()
            for t in (1, 2, 3):
                shares[t - 1, d - 1] = w[m & (tert == t)].sum() / tot
        fig, ax = plt.subplots(figsize=(8, 4.2))
        bottom = np.zeros(10)
        for t, (label, color) in enumerate([("Low tertile", BLUE), ("Middle", "#c9d6e4"), ("High tertile", RED)]):
            ax.bar(range(1, 11), shares[t], bottom=bottom, label=label, color=color)
            bottom += shares[t]
        ax.set_xlabel("Decile of equivalised household disposable income")
        ax.set_ylabel("Share of employed")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=8, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)

    # --- Table 3.2 analogue: job loss per major group, central scenario
    from uk_ai_study.exposure import load_major_group_exposure
    table = load_major_group_exposure()
    shocked = apply_shocks(persons, PRESETS["central"], seed=0)
    disp = shocked["displaced"].to_numpy()
    g = persons["soc_major_group"].to_numpy()
    emp = persons["employment_income"].to_numpy() > 0
    recs = []
    for grp in range(1, 10):
        m = emp & (g == grp * 1000)
        recs.append(
            {"soc_major_group": grp,
             "title": table.loc[grp, "major_group_title"],
             "c_aioe": round(float(table.loc[grp, "c_aioe"]), 3),
             "employment_m": round(float(w[m].sum() / 1e6), 2),
             "job_loss_pct": round(100 * float(w[m & disp].sum() / w[m].sum()), 1)}
        )
    pd.DataFrame(recs).to_csv(OUT / "job_loss_by_major_group.csv", index=False)
    print(pd.DataFrame(recs).to_string(index=False))

    # --- B.7: AI vs uniform shock, disposable income change by decile
    from policyengine_uk import Microsimulation

    def decile_change(persons_variant, label):
        shocked_table = apply_shocks(persons_variant, PRESETS["central"], seed=0)
        sim = Microsimulation(dataset=dataset)
        for col in ("employment_income", "savings_interest_income", "dividend_income"):
            sim.set_input(col, PERIOD, shocked_table[col].to_numpy(dtype=float))
        hni_b = baseline.calculate("household_net_income", period=PERIOD, map_to="person").values
        hni_s = sim.calculate("household_net_income", period=PERIOD, map_to="person").values
        out = []
        for d in range(1, 11):
            m = dec == d
            base = float((hni_b[m] * w[m]).sum())
            out.append(100 * float(((hni_s - hni_b)[m] * w[m]).sum()) / base)
        return np.array(out)

    ai = decile_change(persons, "ai")
    uniform_persons = persons.copy()
    uniform_persons["exposure"] = 1.0
    uniform_persons["complementarity"] = 1.0
    uni = decile_change(uniform_persons, "uniform")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(1, 11)
    ax.bar(x - 0.2, ai, width=0.4, label="AI shock (exposure-allocated)", color=NAVY)
    ax.bar(x + 0.2, uni, width=0.4, label="Uniform shock", color="#9aa8bd")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Decile of equivalised household disposable income")
    ax.set_ylabel("% change in disposable income")
    ax.set_title("Central scenario vs the same shock allocated uniformly", fontsize=10)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "b7_uniform_vs_ai_decile.png", dpi=150)
    plt.close(fig)
    pd.DataFrame({"decile": x, "ai_shock_pct": ai, "uniform_shock_pct": uni}).to_csv(
        OUT / "b7_uniform_vs_ai_decile.csv", index=False)

    # --- B.8: alternative exposure index (eloundou_beta), decile change
    alt = persons.copy()
    e2 = exposure_for_major_group(persons["soc_major_group"], "eloundou_beta")
    alt["exposure"] = np.where(np.isfinite(e2), e2, np.nanmean(e2))
    alt_change = decile_change(alt, "alt")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, ai, width=0.4, label="C-AIOE (main)", color=NAVY)
    ax.bar(x + 0.2, alt_change, width=0.4, label="Eloundou et al. beta", color=GOLD)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Decile of equivalised household disposable income")
    ax.set_ylabel("% change in disposable income")
    ax.set_title("Central scenario under alternative AI-exposure indices", fontsize=10)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "b8_alternative_index_decile.png", dpi=150)
    plt.close(fig)
    pd.DataFrame({"decile": x, "c_aioe_pct": ai, "eloundou_pct": alt_change}).to_csv(
        OUT / "b8_alternative_index_decile.csv", index=False)
    print("fast appendix outputs written")


def decomp_ci(dataset, baseline, persons, n_draws=20):
    """20-draw decomposition of disposable-income change by decile."""
    from policyengine_uk import Microsimulation

    w = persons["weight"].to_numpy()
    dec = persons["decile"].to_numpy()
    hni_b = baseline.calculate("household_net_income", period=PERIOD, map_to="person").values
    draws = np.zeros((n_draws, 10))
    for s in range(n_draws):
        shocked_table = apply_shocks(persons, PRESETS["central"], seed=s)
        sim = Microsimulation(dataset=dataset)
        for col in ("employment_income", "savings_interest_income", "dividend_income"):
            sim.set_input(col, PERIOD, shocked_table[col].to_numpy(dtype=float))
        hni_s = sim.calculate("household_net_income", period=PERIOD, map_to="person").values
        for d in range(1, 11):
            m = dec == d
            draws[s, d - 1] = 100 * float(((hni_s - hni_b)[m] * w[m]).sum()) / float((hni_b[m] * w[m]).sum())
        print(f"decomp draw {s} done", flush=True)
    mean, se = draws.mean(0), draws.std(0, ddof=1) / np.sqrt(n_draws)
    pd.DataFrame(
        {"decile": range(1, 11), "disposable_change_pct": mean,
         "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}
    ).to_csv(OUT / "decomposition_ci.csv", index=False)
    print("decomposition CIs written")


def grids(dataset, baseline, persons):
    """B.9 decile-faceted grid + B.11 no-capital grid."""
    from policyengine_uk import Microsimulation
    from uk_ai_study.runner import gini

    w = persons["weight"].to_numpy()
    dec = persons["decile"].to_numpy()
    hni_b = baseline.calculate("household_net_income", period=PERIOD, map_to="person").values
    hw = baseline.calculate("household_weight", period=PERIOD, map_to="household").values
    eq_b = baseline.calculate("equiv_household_net_income", period=PERIOD, map_to="household").values
    nppl = baseline.calculate("household_count_people", period=PERIOD, map_to="household").values
    gini_b = gini(eq_b, hw * nppl)
    mean_b = float(np.average(
        baseline.calculate("household_net_income", period=PERIOD, map_to="household").values, weights=hw))

    rows = []
    for u in range(0, 11):
        for wg in range(0, 6):
            for capital in (True, False):
                scenario = ShockScenario(
                    f"u{u}w{wg}c{int(capital)}", u / 100, wg / 100,
                    capital_return_increase=0.004 if capital else 0.0)
                shocked_table = apply_shocks(persons, scenario, seed=0)
                sim = Microsimulation(dataset=dataset)
                for col in ("employment_income", "savings_interest_income", "dividend_income"):
                    sim.set_input(col, PERIOD, shocked_table[col].to_numpy(dtype=float))
                hni_s = sim.calculate("household_net_income", period=PERIOD, map_to="person").values
                eq_s = sim.calculate("equiv_household_net_income", period=PERIOD, map_to="household").values
                hh_s = sim.calculate("household_net_income", period=PERIOD, map_to="household").values
                rec = {
                    "unemployment_pct": u, "wage_pct": wg, "capital_shock": capital,
                    "avg_disposable_income_change_pct": 100 * (float(np.average(hh_s, weights=hw)) / mean_b - 1),
                    "gini_change_pp": 100 * (gini(eq_s, hw * nppl) - gini_b),
                }
                for d in range(1, 11):
                    m = dec == d
                    rec[f"decile{d}_change_pct"] = 100 * float(((hni_s - hni_b)[m] * w[m]).sum()) / float((hni_b[m] * w[m]).sum())
                rows.append(rec)
            pd.DataFrame(rows).to_csv(OUT / "grid_deciles_capital.csv", index=False)
            print(f"u={u} w={wg} done (both capital variants)", flush=True)
    print("grids complete")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    task = sys.argv[1] if len(sys.argv) > 1 else "fast"
    dataset, baseline, persons = setup()
    if task == "fast":
        fast(dataset, baseline, persons)
    elif task == "decomp":
        decomp_ci(dataset, baseline, persons)
    elif task == "grids":
        grids(dataset, baseline, persons)
