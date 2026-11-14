"""Run all presets and write results to results/.

Usage: python analysis/run_all.py [--period 2026] [--scenarios ...]
"""

import argparse
from pathlib import Path

from uk_ai_study.runner import run_scenario, write_result
from uk_ai_study.shocks import PRESETS, RIPPLE_PRESETS, WAGE_MARGIN_PRESETS

ALL_PRESETS = list(PRESETS) + list(RIPPLE_PRESETS) + list(WAGE_MARGIN_PRESETS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--scenarios", nargs="*", default=ALL_PRESETS)
    args = parser.parse_args()

    results = Path("results")
    results.mkdir(exist_ok=True)

    for name in args.scenarios:
        result = run_scenario(name, period=args.period)
        write_result(result, results / f"{name}.json")
        print(
            f"{name}: exchequer £{result.exchequer_cost/1e9:.1f}bn, "
            f"poverty BHC {result.poverty_rate_change_bhc*100:+.2f}pp, "
            f"gini {result.gini_shocked - result.gini_baseline:+.4f}"
        )


if __name__ == "__main__":
    main()
