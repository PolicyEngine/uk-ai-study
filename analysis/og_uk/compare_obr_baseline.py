"""Reproduce the OG-UK vs OBR baseline comparison in the PR description.

Reads results/ai_scenario_1sector.json (baseline arm) and prints:
  - real GDP growth 2027-2030 and its mean, against OBR's 1.6%
  - the component decomposition (productivity, labour supply, potential output)

OBR figures, March 2026 Economic and fiscal outlook:
  para 1.2   productivity growth 1.0% medium term; labour supply 0.5% by 2030
  para 1.10  GDP growth averages 1.6% a year from 2027 to 2030
  para 2.10  potential output growth 1.2% in 2026 rising to 1.5% in 2030
https://assets.publishing.service.gov.uk/media/69a6d7b62e1f4fbda4252208/economic-and-fiscal-outlook-march-2026-web-accessible.pdf

Usage: python analysis/og_uk/compare_obr_baseline.py [--oguk-dir PATH]
"""

import argparse
import json
from pathlib import Path

import numpy as np

OBR = {
    "gdp_growth_2027_30": 1.60,
    "productivity_medium_term": 1.0,
    "labour_supply_2030": 0.5,
    "potential_output_2030": 1.5,
    "potential_output_2026": 1.2,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oguk-dir", default=str(Path.home() / "ogmodels" / "OG-UK"),
                    help="OG-UK checkout, for oguk_default_parameters.json")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results" / "ai_scenario_1sector.json"))
    args = ap.parse_args()

    res = json.load(open(args.results))
    params = json.load(open(Path(args.oguk_dir) / "oguk" / "oguk_default_parameters.json"))

    def par(name):
        v = params[name]
        return np.asarray(v["value"] if isinstance(v, dict) and "value" in v else v).ravel()

    g_n = par("g_n")
    g_y = float(par("g_y_annual")[0])

    # Model Y is detrended by productivity AND population: real growth is
    # dlog(Y_hat) + g_y + g_n. Omitting g_n understates growth.
    Y = np.array(res["baseline"]["Y"])[:5]
    growth = (np.diff(np.log(Y)) + g_y + g_n[1:5]) * 100
    years = [2027, 2028, 2029, 2030]

    print("OG-UK BASELINE vs OBR (March 2026 EFO) — real GDP growth\n")
    print(f"{'year':>6}{'OG-UK':>9}")
    for y, x in zip(years, growth):
        print(f"{y:>6}{x:>8.2f}%")
    print(f"\n  OG-UK 2027-30 mean : {growth.mean():.2f}%")
    print(f"  OBR  2027-30 mean  : {OBR['gdp_growth_2027_30']:.2f}%   (EFO para 1.10)")
    print(f"  difference         : {growth.mean() - OBR['gdp_growth_2027_30']:+.2f}pp")

    print("\nCOMPONENTS — where the agreement breaks down\n")
    print(f"{'':<34}{'OBR':>8}{'OG-UK':>9}{'gap':>9}")
    rows = [
        ("productivity growth, medium term", OBR["productivity_medium_term"], g_y * 100),
        ("labour supply growth by 2030", OBR["labour_supply_2030"], g_n[4] * 100),
        ("potential output growth 2030", OBR["potential_output_2030"], g_y * 100 + g_n[4] * 100),
        ("potential output growth 2026", OBR["potential_output_2026"], g_y * 100 + g_n[0] * 100),
    ]
    for label, obr, oguk in rows:
        print(f"{label:<34}{obr:>7.1f}%{oguk:>8.2f}%{oguk - obr:>+8.2f}pp")

    print(
        "\nNOTE: g_y_annual is mis-sourced. oguk/api.py cites OBR POTENTIAL OUTPUT\n"
        "growth, but OG-Core defines g_y_annual as labour-augmenting technological\n"
        "change (PRODUCTIVITY growth). OBR separates them: 1.0% + 0.5% = 1.5%.\n"
        "OG-UK already carries labour in g_n, so this double-counts labour supply.\n"
        f"Setting g_y_annual = 0.010 gives potential output growth "
        f"{1.0 + g_n[4] * 100:.2f}% in 2030 against OBR's 1.5%."
    )


if __name__ == "__main__":
    main()
