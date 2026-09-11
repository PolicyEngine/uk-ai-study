"""Score the Korinek et al. (2026) scenarios through PolicyEngine UK.

Writes results/anthropic/<scenario>.json plus a duration sensitivity, since
the stock-to-flow conversion is the largest single source of uncertainty.

Usage: python analysis/anthropic_scenarios.py [--data-dir data] [--period 2026]
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from uk_ai_study.anthropic import ANTHROPIC_PRESETS
from uk_ai_study.runner import run_scenario, write_result

#: Expected unemployment duration (years) for the stock-to-flow conversion.
#: 0.5 is the survey median re-employment time (Appendix B); 1.0 and 2.0 are
#: the slower-adjustment cases their explorer also admits.
DURATIONS = (0.5, 1.0, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = Path(args.data_dir)
    dataset = data / "frs_2024_25.h5"
    adult_tab = data / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
    out = Path("results/anthropic")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, scenario in ANTHROPIC_PRESETS.items():
        result = run_scenario(
            dataset, adult_tab, scenario, period=args.period, seed=args.seed
        )
        write_result(result, out / f"{name}.json")
        rows.append((name, scenario, result))
        print(
            f"{name}: exchequer £{result.exchequer_cost/1e9:+.1f}bn, "
            f"poverty BHC {result.poverty_rate_change_bhc*100:+.2f}pp, "
            f"gini {result.gini_shocked - result.gini_baseline:+.4f}, "
            f"displaced {result.displaced_weighted/1e6:.2f}m"
        )

    sensitivity = {}
    for d in DURATIONS:
        for name, scenario in ANTHROPIC_PRESETS.items():
            if scenario.knowledge_displacement_flow <= 0 and d != DURATIONS[0]:
                continue
            variant = replace(scenario, name=f"{name}_d{d}", duration_years=d)
            result = run_scenario(
                dataset, adult_tab, variant, period=args.period, seed=args.seed
            )
            sensitivity[variant.name] = {
                "duration_years": d,
                "knowledge_flow": variant.knowledge_displacement_flow,
                "exchequer_cost": result.exchequer_cost,
                "poverty_rate_change_bhc": result.poverty_rate_change_bhc,
                "gini_change": result.gini_shocked - result.gini_baseline,
                "displaced_weighted": result.displaced_weighted,
            }
            print(
                f"  [d={d}y] {name}: £{result.exchequer_cost/1e9:+.1f}bn, "
                f"{result.poverty_rate_change_bhc*100:+.2f}pp"
            )
    (out / "duration_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))


if __name__ == "__main__":
    main()
