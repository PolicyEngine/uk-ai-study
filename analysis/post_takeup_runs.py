#!/usr/bin/env python3
"""Post-processing for the UC take-up and new-style JSA text (no simulation).

Two groups of pure functions, unit-tested in tests/test_post_takeup_runs.py:

From run 15's saved draws (results/next_runs/obr_benchmark_draws.csv), so
available now:

* ``obr_per_head`` - the OBR's welfare addition per additional unemployed
  person (its 1.4-point rise applied to Microcosm's labour force), run 15's
  UC per displaced worker, the mean earnings lost per displaced worker, and
  the benefit change per pound of earnings lost on two perimeters: the
  model's household benefit total (which nets lost childcare support and
  statutory pay) and cash benefits only.

From the take-up runs (analysis/displaced_takeup_runs.py,
results/takeup/takeup_draws.csv), pending until those runs finish:

* ``variant_means`` - one scenario/variant's means, accepted only when every
  seed is present.
* ``paired_change`` - a variant minus the existing run, seed by seed (the
  displaced sets are identical by construction).
* ``absorption_variants`` - pence of cost per pound of earnings lost in the
  seed-0 displacement-only case, on the government balance and on the
  household basis of tax_composition.py, for each variant.

Dolls, Fuest and Peichl: the UK row of the working paper's Table 3
(unemployment shock) is stored in ``DOLLS_WP_UK_UNEMPLOYMENT``. The
published version (J. Public Econ. 2012) could not be checked here, so the
text attributes the UK figures to the working paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Dolls, Fuest and Peichl (2010), NBER WP 16275, Table 3: decomposition of
#: the income stabilisation coefficient for the unemployment shock, UK row.
#: (0.186 in Table 1 is the UK demand-stabilisation coefficient C2_US, not a
#: benefit share.)
DOLLS_WP_UK_UNEMPLOYMENT = {"income_tax": 0.191, "state_tax": 0.000, "social_insurance": 0.061,
                            "benefits": 0.163, "total": 0.415}
TAKEUP_SEEDS = {"disp_only": (0,), "obr": tuple(range(20)), "central": tuple(range(20))}


class Incomplete(Exception):
    """A take-up scenario/variant does not have all its seeds."""


def obr_per_head(draws: pd.DataFrame, labour_force_m: float, rise_pp: float, welfare_bn: float) -> dict:
    """Per-head comparison of run 15 with the OBR (pounds; pence per pound)."""
    m = draws.mean(numeric_only=True)
    cash = m.benefits_change_bn - m.childcare_change_bn - m.statutory_pay_change_bn
    extra_unemployed_m = rise_pp / 100 * labour_force_m
    return {
        "obr_extra_unemployed_m": extra_unemployed_m,
        "obr_welfare_per_extra_unemployed_gbp": welfare_bn * 1e3 / extra_unemployed_m,
        "uc_per_displaced_gbp": m.uc_change_bn * 1e3 / m.displaced_weighted_m,
        "mean_lost_earnings_gbp": m.wage_bill_lost_bn * 1e3 / m.displaced_weighted_m,
        "benefits_pence": 100 * m.benefits_change_bn / m.wage_bill_lost_bn,
        "cash_benefits_pence": 100 * cash / m.wage_bill_lost_bn,
        "cash_benefits_bn": cash,
    }


def variant_rows(draws: pd.DataFrame, scenario: str, variant: str) -> pd.DataFrame:
    d = draws[(draws["scenario"].astype(str) == scenario) & (draws["variant"].astype(str) == variant)]
    want = set(TAKEUP_SEEDS[scenario])
    if set(d["seed"].astype(int)) != want or len(d) != len(want):
        raise Incomplete(f"take-up runs {scenario}/{variant}: {len(d)} of {len(want)} seeds")
    return d.sort_values("seed").reset_index(drop=True)


def variant_means(draws: pd.DataFrame, scenario: str, variant: str) -> pd.Series:
    d = variant_rows(draws, scenario, variant)
    out = d.mean(numeric_only=True)
    out["uc_per_displaced_gbp"] = out["uc_change_bn"] * 1e3 / out["displaced_weighted_m"]
    out["cash_benefits_pence"] = 100 * out["cash_benefits_change_bn"] / out["wage_bill_lost_bn"]
    return out


def paired_change(draws: pd.DataFrame, scenario: str, variant: str, existing: pd.DataFrame, col: str,
                  existing_col: str | None = None) -> float:
    """Mean over seeds of (variant - existing run) for ``col``; the existing
    run's draws must cover the same seeds with the same displaced count."""
    d = variant_rows(draws, scenario, variant).set_index("seed")
    e = existing.set_index("seed").reindex(d.index)
    if e["displaced_weighted_m"].isna().any() or not np.allclose(
            e["displaced_weighted_m"].astype(float), d["displaced_weighted_m"].astype(float), rtol=1e-9):
        raise ValueError(f"take-up {scenario}/{variant} does not pair with the existing draws")
    return float((d[col].astype(float) - e[existing_col or col].astype(float)).mean())


def absorption_variants(draws: pd.DataFrame, wage_bill_bn: float) -> dict:
    """Pence per pound of earnings lost, seed-0 displacement-only, by variant:
    the fall in the government balance and the fall in IT + NI - benefits
    (the bases of revision_accounting.absorption_rates)."""
    out = {}
    for v in ("takeup1", "jsa"):
        r = variant_rows(draws, "disp_only", v).iloc[0]
        if not np.isclose(float(r["wage_bill_lost_bn"]), wage_bill_bn, rtol=1e-6):
            raise ValueError("disp_only wage bill differs from the absorption-rate wage bill")
        out[v] = {"gov_balance": float(r["exchequer_cost_bn"]) / wage_bill_bn,
                  "household": -float(r["narrow_cash_change_bn"]) / wage_bill_bn}
    return out
