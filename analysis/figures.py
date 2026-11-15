"""Render the main-text (fig 4.1-4.7), grid appendix (B.9, B.11,
decomposition CI) and incidence-family figures from the committed
CSVs/JSONs. No simulations are run here — presentation only.

Usage: python analysis/figures.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import MEDIUM_DARK_GRAY  # noqa: E402
from figstyle import (
    AQUA,
    BLUE,
    DECILE_AXIS,
    DIVERGING,
    DPI,
    FACETS,
    GREEN,
    HEATMAP,
    INK,
    INK2,
    LIGHT_BLUE,
    MUTED,
    RED,
    SINGLE,
    TWOPANEL,
    VIOLET,
    YELLOW,
    apply_style,
    decile_ax,
    save,
    legend_below,
)

JR16 = Path("results/jr16")
APPENDIX = Path("results/appendix")
INCIDENCE = Path("results/incidence")
ROBUSTNESS = Path("results/robustness")


def _bar_labels(ax, xs, ys, fmt="{:.1f}", pad_frac=0.02):
    span = max(abs(min(min(ys), 0)), max(ys)) or 1.0
    for x, y in zip(xs, ys):
        ax.text(x, y + span * pad_frac, fmt.format(y), ha="center",
                va="bottom", fontsize=8, color=INK2)


def fig4_1():
    d = pd.read_csv(JR16 / "fig4_1_transition_by_decile.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    y = 100 * d["share_transitioning"]
    err = [100 * (d["share_transitioning"] - d["ci_low"]),
           100 * (d["ci_high"] - d["share_transitioning"])]
    ax.bar(d["decile"], y, yerr=err, capsize=3, color=BLUE,
           error_kw={"ecolor": INK2, "lw": 1})
    _bar_labels(ax, d["decile"], y, fmt="{:.1f}", pad_frac=0.04)
    decile_ax(ax, "Share transitioning to unemployment (%)")
    save(fig, JR16 / "fig4_1_transition.png")


def fig4_2():
    d = pd.read_csv(JR16 / "fig4_2_wage_gain_by_decile.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    y = 100 * d["pct_change_employment_income"]
    ax.bar(d["decile"], y, color=BLUE)
    _bar_labels(ax, d["decile"], y, fmt="{:.2f}", pad_frac=0.004)
    decile_ax(ax, "Change in employment income (%)\nnon-transitioning population")
    # zero baseline: bar heights must be proportional to the values
    ax.set_ylim(0, y.max() * 1.12)
    save(fig, JR16 / "fig4_2_wages.png")


def fig4_3():
    d = pd.read_csv(JR16 / "fig4_3_capital_by_decile.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    y = 100 * d["share_of_all_capital_income"]
    ax.bar(d["decile"], y, color=BLUE)
    _bar_labels(ax, d["decile"], y)
    decile_ax(ax, "Share of all interest and dividend income (%)")
    save(fig, JR16 / "fig4_3_capital.png")


def fig4_4():
    d = pd.read_csv(JR16 / "fig4_4_decomposition.csv")
    comps = [
        ("disposable_income", "Disposable income", BLUE),
        ("market_income", "Market income", RED),
        ("benefits", "Benefits", AQUA),
        ("tax_and_contributions", "Tax & contributions", YELLOW),
    ]
    if "residual" in d.columns:
        # a lighter grey than market income (GRAY), so the two series differ
        comps.append(("residual", "Other (residual)", MEDIUM_DARK_GRAY))
    wd = 0.8 / len(comps)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, (col, label, color) in enumerate(comps):
        ax.bar(d["decile"] + (k - (len(comps) - 1) / 2) * wd, 100 * d[col],
               width=wd * 0.92, label=label, color=color)
    ax.axhline(0, color=INK, lw=0.8)
    decile_ax(ax, "Change (% of baseline disposable income)")
    legend_below(ax, ncol=len(comps))
    save(fig, JR16 / "fig4_4_decomposition.png")


def load_grid():
    """run_grid writes net_revenue_change_bn / _pct_of_receipts directly
    (see replicate_jr16.py); this remains only as the read point."""
    return pd.read_csv(JR16 / "grid.csv")


def _heatmap(ax, piv, lim, annotate=True):
    im = ax.imshow(piv.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    if annotate:
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                dark = abs(v) > 0.62 * lim
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=7.5, color="white" if dark else INK)
    ax.grid(visible=False)
    return im


def grid_heatmaps(g):
    for col, title, fname in [
        ("avg_disposable_income_change_pct",
         "Average change in household disposable income (%)",
         "fig4_5_disposable_grid.png"),
        ("net_revenue_change_pct_of_receipts",
         "Change in net Exchequer revenue (% of income tax + NI receipts)",
         "fig4_6_exchequer_grid.png"),
        ("gini_change_pp", "Change in Gini index (pp)", "fig4_7_gini_grid.png"),
    ]:
        piv = g.pivot(index="wage_pct", columns="unemployment_pct",
                      values=col).sort_index(ascending=False)
        lim = np.abs(piv.values).max()
        fig, ax = plt.subplots(figsize=HEATMAP)
        im = _heatmap(ax, piv, lim)
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("Displacement scenario (% of employees)")
        ax.set_ylabel("Wage scenario (% increase)")
        fig.colorbar(im, ax=ax, shrink=0.8, label=title)
        save(fig, JR16 / fname)


def decomposition_ci():
    d = pd.read_csv(APPENDIX / "decomposition_ci.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    err = [d["disposable_change_pct"] - d["ci_low"],
           d["ci_high"] - d["disposable_change_pct"]]
    ax.bar(d["decile"], d["disposable_change_pct"], yerr=err, capsize=3,
           color=BLUE, error_kw={"ecolor": INK2, "lw": 1})
    ax.axhline(0, color=INK, lw=0.8)
    decile_ax(ax, "Change in disposable income (%)")
    save(fig, APPENDIX / "decomposition_ci.png")


def b9_b11():
    g = pd.read_csv(APPENDIX / "grid_deciles_capital.csv")

    gc = g[g["capital_shock"] == True]  # noqa: E712
    fig, axes = plt.subplots(2, 5, figsize=FACETS, sharex=True, sharey=True)
    lim = max(abs(gc[f"decile{d}_change_pct"]).max() for d in range(1, 11))
    for d in range(1, 11):
        ax = axes[(d - 1) // 5][(d - 1) % 5]
        piv = gc.pivot(index="wage_pct", columns="unemployment_pct",
                       values=f"decile{d}_change_pct").sort_index(ascending=False)
        im = _heatmap(ax, piv, lim, annotate=False)
        ax.set_title(f"Decile {d}", fontsize=10)
        ax.set_xticks(range(0, 11, 2), piv.columns[::2])
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.tick_params(labelsize=8)
    fig.supxlabel("Displacement scenario (% of employees)", fontsize=10)
    fig.supylabel("Wage scenario (% increase)", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Change in disposable income (%)")
    fig.savefig(APPENDIX / "b9_grid_by_decile.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    gn = g[g["capital_shock"] == False]  # noqa: E712
    piv = gn.pivot(index="wage_pct", columns="unemployment_pct",
                   values="avg_disposable_income_change_pct").sort_index(ascending=False)
    lim = np.abs(piv.values).max()
    fig, ax = plt.subplots(figsize=HEATMAP)
    im = _heatmap(ax, piv, lim)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("Displacement scenario (% of employees)")
    ax.set_ylabel("Wage scenario (% increase)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Average change in household disposable income (%)")
    save(fig, APPENDIX / "b11_grid_no_capital.png")


FAMILY_STYLE = {
    "exposure": ("Exposure-proportional", BLUE),
    "junior": ("Junior-concentrated", AQUA),
    "compression": ("Expertise compression", YELLOW),
    "uniform": ("Uniform", GREEN),
    "klein_top_loaded": ("Top-loaded stress test", VIOLET),
}


SHORT_LABEL = {"exposure": "Exposure-proportional", "junior": "Junior-concentrated",
               "compression": "Expertise compression", "uniform": "Uniform",
               "klein_top_loaded": "Top-loaded"}
# (dx in £bn, dy in pp, ha, va) offsets for the point labels
POINT_LABEL = {"exposure": (0.9, -0.30, "left", "top"),
               "junior": (0.0, 0.62, "center", "bottom"),
               "compression": (0.5, 0.12, "left", "bottom"),
               "uniform": (0.0, -0.30, "center", "top"),
               "klein_top_loaded": (0.0, -0.32, "center", "top")}
LEADER = {"exposure", "junior"}

def incidence_families():
    """Two-panel comparison of the five incidence families (same aggregate
    shock): decile transition shares (left) and the exchequer-cost /
    poverty-change trade-off (right)."""
    fams = {}
    for name in FAMILY_STYLE:
        with open(INCIDENCE / f"{name}.json") as f:
            fams[name] = json.load(f)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=TWOPANEL,
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    deciles = np.arange(1, 11)
    handles, labels = [], []
    for name, (label, color) in FAMILY_STYLE.items():
        shares = [fams[name]["decile_transition_share_pct"][str(d)] for d in deciles]
        line, = axl.plot(deciles, shares, color=color, lw=2, marker="o", ms=5,
                         markerfacecolor=color, markeredgecolor="white",
                         markeredgewidth=1, label=label)
        handles.append(line)
        labels.append(label)
    decile_ax(axl, "Share transitioning to unemployment (%)")
    axl.set_ylim(bottom=0)

    # Right panel: 50-draw means with +/-1 SD bars (the basis of the text and
    # Table tab:incidence), not the seed-0 point, which is a bottom-tail draw
    # for poverty and reorders the families. Identified by the shared legend.
    draws = pd.read_csv(ROBUSTNESS / "incidence_draws_five.csv")
    stats = draws.groupby("family")[["exchequer_cost_bn", "poverty_change_bhc_pp"]].agg(["mean", "std"])
    for name, (label, color) in FAMILY_STYLE.items():
        x, xs = stats.loc[name, ("exchequer_cost_bn", "mean")], stats.loc[name, ("exchequer_cost_bn", "std")]
        y, ys = stats.loc[name, ("poverty_change_bhc_pp", "mean")], stats.loc[name, ("poverty_change_bhc_pp", "std")]
        axr.errorbar(x, y, xerr=xs, yerr=ys, fmt="none", ecolor=color, elinewidth=1, alpha=0.45, zorder=2)
        axr.scatter(x, y, s=45, color=color, edgecolor="white", lw=0.8, zorder=3)
        # direct labels: exposure and junior sit almost on top of each other
        # (their 50-draw means differ by ~£0.3bn and ~0.03pp), so label both
        dx, dy, ha, va = POINT_LABEL.get(name, (0.6, 0.06, "left", "bottom"))
        axr.annotate(SHORT_LABEL[name], (x, y), xytext=(x + dx, y + dy), ha=ha, va=va,
                     fontsize=8.5, color=INK2, zorder=4,
                     arrowprops=(dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=4)
                                 if name in LEADER else None))
    axr.set_xlabel("Exchequer cost (£ billion per year)")
    axr.set_ylabel("Change in absolute BHC poverty (pp)")
    lo = (stats[("exchequer_cost_bn", "mean")] - stats[("exchequer_cost_bn", "std")]).min()
    hi = (stats[("exchequer_cost_bn", "mean")] + stats[("exchequer_cost_bn", "std")]).max()
    pad = 0.06 * (hi - lo)
    axr.set_xlim(min(0, lo - pad), hi + pad)
    axr.set_ylim(bottom=0)
    axr.grid(axis="x", visible=True)

    # One shared legend box serving both panels.
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(INCIDENCE / "incidence_families.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def appendix_restyle():
    """House-style redraw of the CSV-backed appendix charts (B.1-B.5, B.7,
    B.8): no in-figure titles (the caption carries them), blue for single
    series, green for the uniform rule as in the incidence figure. Reads the
    CSVs analysis/appendix.py wrote; runs no simulation and leaves that
    stage's code (and so its fingerprinted values) untouched."""
    base = pd.read_csv(APPENDIX / "baseline_distributions.csv")
    for name in ["b1_market_income_less_capital", "b2_capital_income", "b3_benefits",
                 "b4_tax_and_ni", "b5_disposable_income"]:
        fig, ax = plt.subplots(figsize=SINGLE)
        y = base[name].to_numpy()
        ax.bar(base["decile"], y, color=BLUE)
        _bar_labels(ax, base["decile"], y)
        decile_ax(ax, "£ billion per year")
        save(fig, APPENDIX / f"{name}.png")
    for csv, (c1, l1), (c2, l2, col2), fname in [
        ("b7_uniform_vs_ai_decile.csv", ("ai_shock_pct", "AI shock (exposure-allocated)"),
         ("uniform_shock_pct", "Uniform shock", GREEN), "b7_uniform_vs_ai_decile.png"),
        ("b8_alternative_index_decile.csv", ("c_aioe_pct", "C-AIOE (main)"),
         ("eloundou_pct", "Eloundou et al. beta", YELLOW), "b8_alternative_index_decile.png"),
    ]:
        d = pd.read_csv(APPENDIX / csv)
        fig, ax = plt.subplots(figsize=SINGLE)
        ax.bar(d["decile"] - 0.2, d[c1], width=0.38, label=l1, color=BLUE)
        ax.bar(d["decile"] + 0.2, d[c2], width=0.38, label=l2, color=col2)
        ax.axhline(0, color=INK, lw=0.8)
        decile_ax(ax, "Change in disposable income (%)")
        legend_below(ax, ncol=2)
        save(fig, APPENDIX / fname)


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    apply_style()

    fig4_1()
    fig4_2()
    fig4_3()
    fig4_4()
    grid_heatmaps(load_grid())
    decomposition_ci()
    b9_b11()
    incidence_families()
    appendix_restyle()
    print("figures written to results/jr16, results/appendix and results/incidence")


if __name__ == "__main__":
    main()
