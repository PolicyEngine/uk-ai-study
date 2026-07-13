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

from figstyle import (
    AQUA, BLUE, GREEN, INK, INK2, LIGHT_BLUE, MUTED, RED, SINGLE, YELLOW,
    apply_style, decile_ax, save,
)

DATA = Path("data")
OUT = Path("results/appendix")
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
H5 = DATA / "frs_2024_25.h5"
PERIOD = 2026


def setup():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset
    from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group

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

    equiv = calc("equiv_hbai_household_net_income")
    w = persons["weight"].to_numpy()
    order = np.argsort(equiv)
    cw = np.cumsum(w[order])
    ranks = np.empty(len(equiv))
    ranks[order] = cw / cw[-1]
    persons["decile"] = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
    return dataset, baseline, persons


def bar_by_decile(values_bn, title, ylabel, fname, color=BLUE, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.bar(range(1, 11), values_bn, color=color)
    for x, y in zip(range(1, 11), values_bn):
        ax.text(x, y + max(values_bn) * 0.02, fmt.format(y), ha="center",
                va="bottom", fontsize=8, color=INK2)
    decile_ax(ax, ylabel)
    ax.set_title(title)
    save(fig, OUT / fname)


# --- CSV-only redraw helpers (no simulation needed) ---------------------

BASELINE_META = {
    "b1_market_income_less_capital": (BLUE, "Aggregate market income less capital income"),
    "b2_capital_income": (YELLOW, "Aggregate capital income (interest + dividends)"),
    "b3_benefits": (AQUA, "Aggregate benefits (HBAI)"),
    "b4_tax_and_ni": (GREEN, "Aggregate income tax and National Insurance"),
    "b5_disposable_income": (BLUE, "Aggregate disposable income (HBAI)"),
}


def plot_baseline_distributions(df):
    for name, (color, title) in BASELINE_META.items():
        bar_by_decile(df[name].to_numpy(), title + " — UK baseline, 2026",
                      "£ billion per year", f"{name}.png", color=color)


def _paired_decile_bars(x, y1, y2, label1, label2, color1, color2, title, fname):
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.bar(x - 0.2, y1, width=0.38, label=label1, color=color1)
    ax.bar(x + 0.2, y2, width=0.38, label=label2, color=color2)
    ax.axhline(0, color=INK, lw=0.8)
    decile_ax(ax, "Change in disposable income (%)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    save(fig, OUT / fname)


def plot_b7(df):
    _paired_decile_bars(
        df["decile"].to_numpy(), df["ai_shock_pct"], df["uniform_shock_pct"],
        "AI shock (exposure-allocated)", "Uniform shock", BLUE, LIGHT_BLUE,
        "Central scenario vs the same shock allocated uniformly",
        "b7_uniform_vs_ai_decile.png")


def plot_b8(df):
    _paired_decile_bars(
        df["decile"].to_numpy(), df["c_aioe_pct"], df["eloundou_pct"],
        "C-AIOE (main)", "Eloundou et al. beta", BLUE, YELLOW,
        "Central scenario under alternative AI-exposure indices",
        "b8_alternative_index_decile.png")


def redraw():
    """Restyle-only re-render of every appendix figure that can be redrawn
    from committed CSVs (B.1-B.5, B.7, B.8). The exposure/complementarity
    tertile figures need simulation microdata and are not redrawn here."""
    plot_baseline_distributions(pd.read_csv(OUT / "baseline_distributions.csv"))
    plot_b7(pd.read_csv(OUT / "b7_uniform_vs_ai_decile.csv"))
    plot_b8(pd.read_csv(OUT / "b8_alternative_index_decile.csv"))
    print("redrew B.1-B.5, B.7, B.8 from CSVs")


def fast(dataset, baseline, persons):
    from uk_ai_study.exposure import exposure_for_major_group
    from uk_ai_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

    w = persons["weight"].to_numpy()
    dec = persons["decile"].to_numpy()

    # --- B.1-B.5 baseline distributions (weighted aggregates, GBP bn/yr)
    # Built at HOUSEHOLD level: broadcasting household totals to persons and
    # person-weight-summing counts each household once per member (#1,
    # finding 3 — the £683bn benefits / £3.3tn disposable artefact).
    def hcalc(v):
        return baseline.calculate(v, period=PERIOD, map_to="household").values

    hw = hcalc("household_weight")
    equiv_h = hcalc("equiv_hbai_household_net_income")
    n_h = hcalc("household_count_people")
    order_h = np.argsort(equiv_h)
    cw_h = np.cumsum((hw * n_h)[order_h])
    ranks_h = np.empty(len(equiv_h))
    ranks_h[order_h] = cw_h / cw_h[-1]
    dec_h = np.clip(np.ceil(ranks_h * 10).astype(int), 1, 10)

    comps = {
        "b1_market_income_less_capital": (
            hcalc("employment_income") + hcalc("self_employment_income")
            + hcalc("private_pension_income")),
        "b2_capital_income": hcalc("savings_interest_income") + hcalc("dividend_income"),
        "b3_benefits": hcalc("hbai_benefits"),
        "b4_tax_and_ni": hcalc("income_tax") + hcalc("national_insurance"),
        "b5_disposable_income": hcalc("hbai_household_net_income"),
    }
    rows = {}
    for name, series in comps.items():
        rows[name] = np.array([float((series * hw)[dec_h == d].sum()) / 1e9 for d in range(1, 11)])
    df_base = pd.DataFrame(rows, index=range(1, 11)).rename_axis("decile")
    df_base.to_csv(OUT / "baseline_distributions.csv")
    plot_baseline_distributions(df_base)

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
        fig, ax = plt.subplots(figsize=SINGLE)
        bottom = np.zeros(10)
        for t, (label, color) in enumerate(
            [("Low tertile", "#cde2fb"), ("Middle", "#6da7ec"), ("High tertile", "#184f95")]
        ):
            ax.bar(range(1, 11), shares[t], bottom=bottom, label=label,
                   color=color, edgecolor="white", linewidth=0.8)
            bottom += shares[t]
        decile_ax(ax, "Share of employed")
        ax.set_title(title)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        save(fig, OUT / fname)

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
        sim = build_shocked_simulation(dataset, baseline, shocked_table, PERIOD)
        hni_b = baseline.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values
        hni_s = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values
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

    x = np.arange(1, 11)
    df_b7 = pd.DataFrame({"decile": x, "ai_shock_pct": ai, "uniform_shock_pct": uni})
    df_b7.to_csv(OUT / "b7_uniform_vs_ai_decile.csv", index=False)
    plot_b7(df_b7)

    # --- B.8: alternative exposure index (eloundou_beta), decile change
    alt = persons.copy()
    e2 = exposure_for_major_group(persons["soc_major_group"], "eloundou_beta")
    alt["exposure"] = np.where(np.isfinite(e2), e2, np.nanmean(e2))
    alt_change = decile_change(alt, "alt")
    df_b8 = pd.DataFrame({"decile": x, "c_aioe_pct": ai, "eloundou_pct": alt_change})
    df_b8.to_csv(OUT / "b8_alternative_index_decile.csv", index=False)
    plot_b8(df_b8)
    print("fast appendix outputs written")


def decomp_ci(dataset, baseline, persons, n_draws=20):
    """20-draw decomposition of disposable-income change by decile."""
    from policyengine_uk import Microsimulation
    from uk_ai_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

    w = persons["weight"].to_numpy()
    dec = persons["decile"].to_numpy()
    hni_b = baseline.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values
    draws = np.zeros((n_draws, 10))
    for s in range(n_draws):
        shocked_table = apply_shocks(persons, PRESETS["central"], seed=s)
        sim = build_shocked_simulation(dataset, baseline, shocked_table, PERIOD)
        hni_s = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values
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
    from uk_ai_study.shocks import ShockScenario, apply_shocks, build_shocked_simulation

    w = persons["weight"].to_numpy()
    dec = persons["decile"].to_numpy()
    hni_b = baseline.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values
    hw = baseline.calculate("household_weight", period=PERIOD, map_to="household").values
    eq_b = baseline.calculate("equiv_hbai_household_net_income", period=PERIOD, map_to="household").values
    nppl = baseline.calculate("household_count_people", period=PERIOD, map_to="household").values
    gini_b = gini(eq_b, hw * nppl)
    mean_b = float(np.average(
        baseline.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values, weights=hw))

    rows = []
    for u in range(0, 11):
        for wg in range(0, 6):
            for capital in (True, False):
                scenario = ShockScenario(
                    f"u{u}w{wg}c{int(capital)}", u / 100, wg / 100,
                    capital_return_increase=0.004 if capital else 0.0)
                shocked_table = apply_shocks(persons, scenario, seed=0)
                sim = build_shocked_simulation(dataset, baseline, shocked_table, PERIOD)
                hni_s = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values
                eq_s = sim.calculate("equiv_hbai_household_net_income", period=PERIOD, map_to="household").values
                hh_s = sim.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values
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
    apply_style()
    task = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if task == "redraw":  # CSV-only, no simulation
        redraw()
        sys.exit(0)
    dataset, baseline, persons = setup()
    if task == "fast":
        fast(dataset, baseline, persons)
    elif task == "decomp":
        decomp_ci(dataset, baseline, persons)
    elif task == "grids":
        grids(dataset, baseline, persons)
