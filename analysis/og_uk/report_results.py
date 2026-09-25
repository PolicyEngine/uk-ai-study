"""Reproduce the AI-vs-baseline result tables in the PR description.

Prints the year-by-year % gap for every macro series, GDP growth for both
paths, the labour-share path, and the comparison against Anthropic's
substantial scenario (Korinek et al. 2026, Table 3).

Usage: python analysis/og_uk/report_results.py [--oguk-dir PATH]
"""

import argparse
import json
from pathlib import Path

import numpy as np

VARS = [("C", "Consumption"), ("I", "Investment"), ("G", "Government"),
        ("total_tax_revenue", "Tax revenue"), ("D", "Debt"), ("Y", "GDP")]

# Korinek, Jones, Sacher, Cotter & McCrory (2026), Anthropic Institute WP
# 2026-02, Table 3, "substantial" column, 2030.
ANTHROPIC_SUBSTANTIAL = {
    "labour_share_fall_pp": 3.9,     # 60.0c -> 56.1c
    "measured_tfp_pct": 3.1,         # panel d
    "gdp_above_no_ai_pct": 8.3,
    "gdp_growth_pct": 5.4,
    "capital_stock_pct": 13.8,
    "unemployment_pct": 4.6,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oguk-dir", default=str(Path.home() / "ogmodels" / "OG-UK"))
    ap.add_argument("--results", default=str(Path(__file__).parent / "results" / "ai_scenario_1sector.json"))
    args = ap.parse_args()

    d = json.load(open(args.results))
    params = json.load(open(Path(args.oguk_dir) / "oguk" / "oguk_default_parameters.json"))
    gn = np.asarray(params["g_n"]["value"] if isinstance(params["g_n"], dict)
                    else params["g_n"]).ravel()
    gy = 0.011
    years = [2026, 2027, 2028, 2029, 2030]
    trend = np.cumprod(np.concatenate([[1.0], 1 + gy + gn[1:5]]))

    lvl = {k: {v: (np.array(d[k][v])[:5] * trend) for v, _ in VARS} for k in ("baseline", "ai")}

    print("AI vs BASELINE, % gap by year\n")
    print(f"{'variable':<16}" + "".join(f"{y:>9}" for y in years))
    for v, label in VARS:
        print(f"{label:<16}" + "".join(
            f"{(lvl['ai'][v][i] / lvl['baseline'][v][i] - 1) * 100:>+8.2f}%" for i in range(5)))

    print("\nGDP GROWTH\n")
    for k, name in (("baseline", "Baseline"), ("ai", "With AI")):
        Y = np.array(d[k]["Y"])[:5]
        g = (np.diff(np.log(Y)) + gy + gn[1:5]) * 100
        print(f"  {name:<10}" + "".join(f"{x:>8.2f}%" for x in g))

    print("\nLABOUR SHARE (w*L/Y)\n")
    print(f"{'path':<16}" + "".join(f"{y:>9}" for y in years))
    for k, name in (("baseline", "Baseline"), ("ai", "With AI")):
        sl = (np.array(d[k]["w"]) * np.array(d[k]["L"]) / np.array(d[k]["Y"]))[:5]
        print(f"{name:<16}" + "".join(f"{x:>9.4f}" for x in sl))
    print("  flat over time because epsilon=1: at Cobb-Douglas the labour share is")
    print("  exactly 1-gamma, so this is purely the automation channel.")

    print("\nAGAINST ANTHROPIC SUBSTANTIAL (Table 3, 2030)\n")
    A = ANTHROPIC_SUBSTANTIAL
    sl_b = float(d["baseline"]["ss_labour_share"])
    sl_a = float(d["ai"]["ss_labour_share"])
    gdp_gap = (lvl["ai"]["Y"][4] / lvl["baseline"]["Y"][4] - 1) * 100
    Y = np.array(d["ai"]["Y"])[:5]
    g_ai = ((np.diff(np.log(Y)) + gy + gn[1:5]) * 100)[-1]
    K_gap = (np.array(d["ai"]["K"])[4] / np.array(d["baseline"]["K"])[4] - 1) * 100
    print(f"{'outcome':<26}{'Anthropic':>11}{'OG-UK':>10}")
    print(f"{'labour-share fall (pp)':<26}{A['labour_share_fall_pp']:>10.1f}{(sl_b - sl_a) * 100:>10.1f}   matched by construction")
    print(f"{'measured TFP (%)':<26}{A['measured_tfp_pct']:>10.1f}{3.1:>10.1f}   input, matched")
    print(f"{'GDP vs no-AI (%)':<26}{A['gdp_above_no_ai_pct']:>10.1f}{gdp_gap:>10.1f}   we overshoot")
    print(f"{'GDP growth (%)':<26}{A['gdp_growth_pct']:>10.1f}{g_ai:>10.2f}   we undershoot")
    print(f"{'capital stock (%)':<26}{A['capital_stock_pct']:>10.1f}{K_gap:>10.1f}")
    print(f"{'unemployment (%)':<26}{A['unemployment_pct']:>10.1f}{'—':>10}   not modelled")
    print("\nBoth discrepancies have one cause: gamma steps once and is fully in force")
    print("from 2026, where Anthropic's automation builds to 2030. The level arrives")
    print("immediately, leaving no growth behind it. Needs time-varying gamma.")


if __name__ == "__main__":
    main()
