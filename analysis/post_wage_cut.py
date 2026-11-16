#!/usr/bin/env python3
"""Post-processing for the wage-cut household run (no simulation).

Pure functions on ``results/wage_cut/wage_cut_draws.csv`` (written by
``analysis/wage_cut_household.py``) and the paired central draws of
``results/robustness/dividend_coverage_draws.csv``, unit-tested in
tests/test_post_wage_cut.py:

* ``margin_comparison`` - the Table 3 wage-margin scenario against the
  central job-loss scenario, seed by seed. Each wage-cut draw removes the
  earnings of the central displacement draw of the same seed, so the two
  rows are paired.
* ``cut_absorption`` - pence of cost per pound of earnings removed by wage
  cuts alone (no uplift, no capital shock), on the government balance, on
  the household basis (income tax, employee NI and benefits) and for
  employer NI.
* ``job_absorption`` - pence of cost per pound of earnings removed by
  displacement alone on the government balance, from the channel split of
  the same seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEEDS = tuple(range(20))


class Incomplete(Exception):
    """A wage-cut variant does not have all its seeds."""


def variant_rows(draws: pd.DataFrame, variant: str) -> pd.DataFrame:
    d = draws[draws["variant"].astype(str) == variant]
    if set(d["seed"].astype(int)) != set(SEEDS) or len(d) != len(SEEDS):
        raise Incomplete(f"wage-cut run {variant}: {len(d)} of {len(SEEDS)} seeds")
    return d.assign(seed=d["seed"].astype(int)).set_index("seed").sort_index()


def _run_rows(existing: pd.DataFrame, run: str) -> pd.DataFrame:
    e = existing[existing["run"].astype(str) == run]
    e = e.assign(seed=e["seed"].astype(int)).set_index("seed").sort_index()
    if list(e.index) != list(SEEDS):
        raise Incomplete(f"{run}: {len(e)} of {len(SEEDS)} paired seeds")
    return e


def _se(x: pd.Series) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def margin_comparison(draws: pd.DataFrame, existing: pd.DataFrame) -> dict:
    """Wage cuts (variant ``table3``) against job loss (run ``central``)."""
    w = variant_rows(draws, "table3")
    c = _run_rows(existing, "central")
    out = {"draws": len(w), "gross_cut_share_pct": float(w["gross_cut_share_pct"].mean())}
    for col, key in [("exchequer_cost_bn", "cost"), ("poverty_change_bhc_pp", "pov_bhc"),
                     ("poverty_change_ahc_pp", "pov_ahc"), ("gini_change_pp", "gini")]:
        diff = w[col].astype(float) - c[col].astype(float)
        out[key] = {"job": float(c[col].mean()), "job_se": _se(c[col]),
                    "wage": float(w[col].mean()), "wage_se": _se(w[col]),
                    "diff": float(diff.mean()), "diff_se": _se(diff),
                    "diff_min": float(diff.min()), "diff_max": float(diff.max())}
    out["relative_pov_bhc_wage"] = float(w["relative_poverty_change_bhc_pp"].mean())
    return out


def cut_absorption(draws: pd.DataFrame) -> dict:
    """Cost per pound of earnings removed by wage cuts alone (ratio of means)."""
    k = variant_rows(draws, "cut_only")
    gross = float(k["gross_cut_bn"].mean())
    per_draw = k["exchequer_cost_bn"].astype(float) / k["gross_cut_bn"].astype(float)
    return {"gov_balance": float(k["exchequer_cost_bn"].mean()) / gross,
            "household": float(k["household_basis_cost_bn"].mean()) / gross,
            "employer_ni": -float(k["employer_ni_change_bn"].mean()) / gross,
            "uc_bn": float(k["uc_change_bn"].mean()),
            "gross_cut_bn": gross,
            "gov_balance_min": float(per_draw.min()), "gov_balance_max": float(per_draw.max())}


def job_absorption(draws: pd.DataFrame, existing: pd.DataFrame) -> dict:
    """Government-balance cost per pound of earnings removed by displacement
    alone: the channel split's displacement run over the wage bill of the same
    seed's displaced workers (the gross cut of the paired wage-cut draw)."""
    k = variant_rows(draws, "cut_only")
    j = _run_rows(existing, "channel_displacement")
    per_draw = j["exchequer_cost_bn"].astype(float) / k["gross_cut_bn"].astype(float)
    return {"gov_balance": float(j["exchequer_cost_bn"].mean()) / float(k["gross_cut_bn"].mean()),
            "gov_balance_min": float(per_draw.min()), "gov_balance_max": float(per_draw.max()),
            "draws": len(j)}
