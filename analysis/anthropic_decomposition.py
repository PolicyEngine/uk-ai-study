"""Decompose the Anthropic extreme scenario into its three channels.

The headline poverty result is driven by one of: displacement (job loss),
wage divergence (-11.5% knowledge / +33.6% other among survivors), or the
capital shock. Turning each on alone identifies which, and whether the
channels are additive.

Usage: python analysis/anthropic_decomposition.py
"""

import json
from dataclasses import replace
from pathlib import Path

from uk_ai_study.anthropic import ANTHROPIC_PRESETS
from uk_ai_study.runner import run_scenario

DATA = Path("data")
DATASET = DATA / "frs_2024_25.h5"
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"

#: Channel isolations. ``no_capital`` sets the labour share to baseline, so
#: the capital factor is 1.0; ``no_wage`` zeroes both group wage changes;
#: ``no_displacement`` sets both unemployment targets to their counterfactual.
CHANNELS = {
    "full": {},
    "displacement_only": {
        "knowledge_wage_change": 0.0,
        "other_wage_change": 0.0,
        "labour_share": 0.60,
    },
    "wage_only": {
        "knowledge_unemployment": 0.029,
        "other_unemployment": 0.054,
        "labour_share": 0.60,
    },
    "capital_only": {
        "knowledge_unemployment": 0.029,
        "other_unemployment": 0.054,
        "knowledge_wage_change": 0.0,
        "other_wage_change": 0.0,
    },
    "no_capital": {"labour_share": 0.60},
}


def main() -> None:
    base = ANTHROPIC_PRESETS["anthropic_extreme"]
    out = Path("results/anthropic")
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for label, overrides in CHANNELS.items():
        scenario = replace(base, name=f"extreme_{label}", **overrides)
        r = run_scenario(DATASET, ADULT, scenario, period=2026, seed=0)
        rows[label] = {
            "exchequer_cost": r.exchequer_cost,
            "poverty_rate_change_bhc": r.poverty_rate_change_bhc,
            "gini_change": r.gini_shocked - r.gini_baseline,
            "displaced_weighted": r.displaced_weighted,
        }
        print(
            f"{label:20} £{r.exchequer_cost/1e9:+8.1f}bn  "
            f"poverty {r.poverty_rate_change_bhc*100:+6.2f}pp  "
            f"gini {r.gini_shocked - r.gini_baseline:+.4f}  "
            f"displaced {r.displaced_weighted/1e6:.2f}m"
        )
    parts = sum(
        rows[c]["poverty_rate_change_bhc"]
        for c in ("displacement_only", "wage_only", "capital_only")
    )
    print(
        f"\nadditivity check: channels sum to {parts*100:+.2f}pp vs "
        f"full {rows['full']['poverty_rate_change_bhc']*100:+.2f}pp"
    )
    (out / "extreme_decomposition.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
