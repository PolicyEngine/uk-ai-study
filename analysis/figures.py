"""Render the main-text (fig 4.1-4.7) and grid appendix (B.9, B.11,
decomposition CI) figures from the committed CSVs, and add the revenue
columns to grid.csv that the paper quotes.

Recovered from the original generating session (these were run as ad hoc
snippets and never committed — see uk-ai-study#1, finding 8).

Usage: python analysis/figures.py [--data-dir DATA] [--period 2026]

The bar/heatmap figures only need the CSVs under results/. The grid revenue
columns (net_revenue_change_bn, net_revenue_change_pct_of_receipts) need a
baseline PolicyEngine simulation; they are recomputed only if missing from
results/jr16/grid.csv, or always with --recompute-revenue.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

JR16 = Path("results/jr16")
APPENDIX = Path("results/appendix")
NAVY = "#1f3557"


def fig4_1():
    d = pd.read_csv(JR16 / "fig4_1_transition_by_decile.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    err = [d["share_transitioning"] - d["ci_low"], d["ci_high"] - d["share_transitioning"]]
    ax.bar(d["decile"], 100 * d["share_transitioning"], yerr=[100 * e for e in err], capsize=3, color=NAVY)
    for x, y in zip(d["decile"], d["share_transitioning"]):
        ax.text(x, 100 * y + 0.2, f"{100*y:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Decile of equivalised household disposable income")
    ax.set_ylabel("% transitioning to unemployment")
    ax.set_xticks(range(1, 11))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(JR16 / "fig4_1_transition.png", dpi=150)
    plt.close(fig)


def fig4_2():
    d = pd.read_csv(JR16 / "fig4_2_wage_gain_by_decile.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(d["decile"], 100 * d["pct_change_employment_income"], color=NAVY)
    for x, y in zip(d["decile"], d["pct_change_employment_income"]):
        ax.text(x, 100 * y + 0.03, f"{100*y:.2f}%", ha="center", fontsize=8)
    ax.set_xlabel("Decile of equivalised household disposable income")
    ax.set_ylabel("% change in employment income\n(non-transitioning population)")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 3.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(JR16 / "fig4_2_wages.png", dpi=150)
    plt.close(fig)


def fig4_3():
    d = pd.read_csv(JR16 / "fig4_3_capital_by_decile.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(d["decile"], 100 * d["share_of_all_capital_income"], color="#e8a33d")
    for x, y in zip(d["decile"], d["share_of_all_capital_income"]):
        ax.text(x, 100 * y + 0.6, f"{100*y:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Decile of equivalised household disposable income")
    ax.set_ylabel("Share of all interest + dividend income (%)")
    ax.set_xticks(range(1, 11))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(JR16 / "fig4_3_capital.png", dpi=150)
    plt.close(fig)


def fig4_4():
    d = pd.read_csv(JR16 / "fig4_4_decomposition.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    wd = 0.2
    comps = [
        ("disposable_income", "Disposable income", "#1f3557"),
        ("market_income", "Market income", "#b03a3a"),
        ("benefits", "Benefits", "#4a90c4"),
        ("tax_and_contributions", "Tax & contributions", "#4a7d4a"),
    ]
    if "residual" in d.columns:
        comps.append(("residual", "Other (residual)", "#9a9a9a"))
        wd = 0.16
    for k, (col, label, color) in enumerate(comps):
        ax.bar(d["decile"] + (k - (len(comps) - 1) / 2) * wd, 100 * d[col], width=wd, label=label, color=color)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Decile of equivalised household disposable income")
    ax.set_ylabel("% of baseline disposable income")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8, ncol=len(comps))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(JR16 / "fig4_4_decomposition.png", dpi=150)
    plt.close(fig)


def load_grid():
    """run_grid now writes net_revenue_change_bn / _pct_of_receipts directly
    (see replicate_jr16.py); this remains only as the read point."""
    return pd.read_csv(JR16 / "grid.csv")


def grid_heatmaps(g):
    for col, title, fname in [
        ("avg_disposable_income_change_pct", "Avg change in household disposable income (%)", "fig4_5_disposable_grid.png"),
        ("net_revenue_change_pct_of_receipts", "Change in net Exchequer revenue (% of income tax + NI receipts)", "fig4_6_exchequer_grid.png"),
        ("gini_change_pp", "Change in Gini index (pp)", "fig4_7_gini_grid.png"),
    ]:
        piv = g.pivot(index="wage_pct", columns="unemployment_pct", values=col).sort_index(ascending=False)
        lim = np.abs(piv.values).max()
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(11), piv.columns)
        ax.set_yticks(range(6), piv.index)
        ax.set_xlabel("Unemployment scenario (% increase)")
        ax.set_ylabel("Wage scenario (% increase)")
        ax.set_title(title + " — UK")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i,j]:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(JR16 / fname, dpi=150)
        plt.close(fig)


def decomposition_ci():
    d = pd.read_csv(APPENDIX / "decomposition_ci.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    err = [d["disposable_change_pct"] - d["ci_low"], d["ci_high"] - d["disposable_change_pct"]]
    ax.bar(d["decile"], d["disposable_change_pct"], yerr=err, capsize=3, color=NAVY)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Decile of equivalised household disposable income")
    ax.set_ylabel("% change in disposable income")
    ax.set_title("Disposable income change by decile, central scenario (mean of 20 draws, 95% CI)", fontsize=10)
    ax.set_xticks(range(1, 11))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(APPENDIX / "decomposition_ci.png", dpi=150)
    plt.close(fig)


def b9_b11():
    g = pd.read_csv(APPENDIX / "grid_deciles_capital.csv")

    gc = g[g["capital_shock"] == True]  # noqa: E712
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    lim = max(abs(gc[f"decile{d}_change_pct"]).max() for d in range(1, 11))
    for d in range(1, 11):
        ax = axes[(d - 1) // 5][(d - 1) % 5]
        piv = gc.pivot(index="wage_pct", columns="unemployment_pct", values=f"decile{d}_change_pct").sort_index(ascending=False)
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"Decile {d}", fontsize=9)
        ax.set_xticks(range(0, 11, 2), piv.columns[::2], fontsize=7)
        ax.set_yticks(range(6), piv.index, fontsize=7)
    fig.supxlabel("Unemployment scenario (% increase)")
    fig.supylabel("Wage scenario (%)")
    fig.suptitle("Change in disposable income by baseline decile across the scenario grid (%)", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.savefig(APPENDIX / "b9_grid_by_decile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    gn = g[g["capital_shock"] == False]  # noqa: E712
    piv = gn.pivot(index="wage_pct", columns="unemployment_pct", values="avg_disposable_income_change_pct").sort_index(ascending=False)
    lim = np.abs(piv.values).max()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(11), piv.columns)
    ax.set_yticks(range(6), piv.index)
    ax.set_xlabel("Unemployment scenario (% increase)")
    ax.set_ylabel("Wage scenario (% increase)")
    ax.set_title("Avg change in household disposable income (%), NO capital shock — UK")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(APPENDIX / "b11_grid_no_capital.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    fig4_1()
    fig4_2()
    fig4_3()
    fig4_4()
    grid_heatmaps(load_grid())
    decomposition_ci()
    b9_b11()
    print("figures written to results/jr16 and results/appendix")


if __name__ == "__main__":
    main()
