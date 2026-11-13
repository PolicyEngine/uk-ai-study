#!/usr/bin/env python3
"""Post-processing of runs 15 and 16 for the paper (no simulation).

Kept out of ``analysis/next_runs.py`` on purpose: that file's code
fingerprint stamps the run-15/16 checkpoints, so any edit there would stale
them. Everything here reads finished outputs.

Pure functions (unit-tested in ``tests/test_post_next_runs.py``):

* ``complete_variant`` - one run-16 variant's draws, accepted only when all
  20 seeds are present with the current data build and next_runs.py
  fingerprint. Run 16 writes variant by variant, so a finished variant is
  final while later variants are still running.
* ``public_paybill`` - the pay bill of the displaced public-sector employees
  in the central draw (seeds 0-19). Every run-16 variant shares the central
  displaced set seed by seed, so any complete variant gives it; the function
  checks they agree.
* ``capital_breakeven`` - the capital-income uplift at which the central
  cost is zero, interpolated between the central uplift (+39.8 per cent,
  ``dividend_coverage_draws.csv``) and the simulated +60 per cent variant,
  paired over seeds 0-19.
* ``obr_lines`` - run 15's government-balance change split into the lines
  the OBR reports (receipts, welfare) plus childcare support and statutory
  pay, which ``household_benefits`` also carries.
* ``decile_stabilisation`` - run 15's flows by baseline decile from the
  household arrays (``../ai-study-household-arrays/obr_seed*.npz``).

CLI: ``python analysis/post_next_runs.py`` rebuilds
``results/next_runs/obr_household_deciles.csv`` from the household arrays
(and checks it against ``obr_benchmark_deciles.csv``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

R = ROOT / "results"
NR = R / "next_runs"
ARRAYS_DIR = ROOT.parent / "ai-study-household-arrays"
BACKUP_ARRAYS_DIR = Path.home() / "uk-ai-study-rerun-backup" / "household-arrays"
SEEDS = tuple(range(20))
#: capital-income uplift of the central scenario (per cent): 1.405/1.005 - 1
CENTRAL_UPLIFT_PCT = 100 * (1.405 / 1.005 - 1)
BASELINE_RETURN_PCT = 1.005
PE_PARAMS = Path("policyengine_uk") / "parameters" / "gov" / "dwp"


class Incomplete(Exception):
    """A run-16 variant is not (yet) complete and current."""


def complete_variant(draws: pd.DataFrame, variant: str, build_id: str, fingerprint: str,
                     seeds=SEEDS) -> pd.DataFrame:
    d = draws[draws["variant"].astype(str) == variant]
    have = set(d["seed"].astype(int))
    if have != set(seeds) or len(d) != len(seeds):
        raise Incomplete(f"run 16 variant {variant}: {len(have & set(seeds))} of {len(seeds)} seeds")
    if set(d["certified_data_build_id"].astype(str)) != {build_id}:
        raise Incomplete(f"run 16 variant {variant}: other data build")
    if set(d["code_fingerprint"].astype(str)) != {fingerprint}:
        raise Incomplete(f"run 16 variant {variant}: written by other next_runs.py code")
    return d.sort_values("seed").reset_index(drop=True)


def public_paybill(variants: dict[str, pd.DataFrame]) -> dict:
    """Mean, SD and draws of the displaced public-sector pay bill (pounds bn)
    across seeds; all complete variants must agree seed by seed."""
    if not variants:
        raise Incomplete("no complete run-16 variant")
    cols = [v.set_index("seed")["public_paybill_displaced_bn"] for v in variants.values()]
    ref = cols[0]
    for c in cols[1:]:
        if not np.allclose(c.reindex(ref.index).to_numpy(), ref.to_numpy(), rtol=1e-9):
            raise ValueError("run-16 variants disagree on the displaced public pay bill")
    x = ref.astype(float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "draws": int(len(x))}


def capital_breakeven(central: pd.DataFrame, cap60: pd.DataFrame,
                      x0: float = CENTRAL_UPLIFT_PCT, x1: float = 60.0) -> dict:
    """Uplift (per cent of capital income, and points of return) at which the
    mean central cost is zero, linear between the paired means at x0 and x1.
    Raises unless the cost changes sign between the two points."""
    a = central.set_index("seed")["exchequer_cost_bn"].astype(float)
    b = cap60.set_index("seed")["exchequer_cost_bn"].astype(float)
    if set(a.index) != set(b.index):
        raise ValueError("break-even endpoints are not paired on the same seeds")
    y0, y1 = float(a.mean()), float(b.reindex(a.index).mean())
    if not (y0 > 0 > y1):
        raise ValueError(f"cost does not change sign between +{x0:.1f}% ({y0:.2f}) and +{x1:.0f}% ({y1:.2f})")
    x = x0 + (x1 - x0) * y0 / (y0 - y1)
    return {"uplift_pct": x, "pp": BASELINE_RETURN_PCT * x / 100, "cost_at_x0_bn": y0,
            "cost_at_x1_bn": y1, "draws": int(len(a))}


def obr_lines(draws: pd.DataFrame) -> dict:
    """20-draw means of run 15's balance split (pounds bn, positive = adds
    to borrowing). ``household_benefits`` includes childcare support and
    statutory pay, which are neither OBR 'welfare' nor receipts, so they are
    separate lines; 'welfare' is UC plus the other cash benefits."""
    m = draws.mean(numeric_only=True)
    out = {
        "income_tax": -m.income_tax_change_bn,
        "ni_employee": -m.ni_employee_change_bn,
        "ni_employer": -m.employer_ni_change_bn,
        "uc": m.uc_change_bn,
        "childcare": m.childcare_change_bn,
        "statutory_pay": m.statutory_pay_change_bn,
    }
    out["other_benefits"] = m.benefits_change_bn - m.uc_change_bn - m.childcare_change_bn \
        - m.statutory_pay_change_bn
    out["receipts"] = out["income_tax"] + out["ni_employee"] + out["ni_employer"]
    out["welfare"] = out["uc"] + out["other_benefits"]
    out["total"] = out["receipts"] + out["welfare"] + out["childcare"] + out["statutory_pay"]
    out["cost"] = m.exchequer_cost_bn
    out["residual"] = out["cost"] - out["total"]
    if abs(out["residual"]) > 0.05:
        raise ValueError(f"run 15 lines do not add to the balance: residual {out['residual']:.3f}bn")
    out["household_basis"] = m.household_basis_cost_bn
    out["displaced_m"] = m.displaced_weighted_m
    out["paybill"] = m.public_paybill_displaced_bn
    out["wage_bill"] = m.wage_bill_lost_bn
    out["cost_per_displaced_gbp"] = float((draws.exchequer_cost_bn / draws.displaced_weighted_m).mean()) * 1e3
    return out


def flows_from_arrays(base: dict, shocked: dict) -> pd.DataFrame:
    """Household flows of one draw by baseline decile (pounds)."""
    d = {"market_change": shocked["hh_market"] - base["hh_market"],
         "benefit_change": shocked["hh_benefits"] - base["hh_benefits"],
         "uc_change": shocked["hh_uc"] - base["hh_uc"],
         "tax_change": (shocked["hh_it"] + shocked["hh_ni"]) - (base["hh_it"] + base["hh_ni"]),
         "net_income_change": shocked["hh_net"] - base["hh_net"]}
    w = np.asarray(base["weight"], float)
    df = pd.DataFrame({k: np.asarray(v, float) * w for k, v in d.items()})
    df["decile"] = np.asarray(base["decile"])
    return df.groupby("decile").sum()


def decile_stabilisation(per_draw: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean flows by decile over draws (pounds bn) and the pooled
    coefficients: tax-benefit share (dBenefits - dTax)/-dMarket and
    1 - dNet/dMarket. Adds an 'all' row."""
    mean = sum(per_draw) / len(per_draw) / 1e9
    mean.loc["all"] = mean.sum()
    mean["tax_benefit_share"] = (mean.benefit_change - mean.tax_change) / -mean.market_change
    mean["stabilisation_net"] = 1 - mean.net_income_change / mean.market_change
    mean["draws"] = len(per_draw)
    return mean.reset_index()


def load_arrays(name: str, dirs=(ARRAYS_DIR, BACKUP_ARRAYS_DIR)) -> dict:
    for d in dirs:
        f = Path(d) / f"{name}.npz"
        if f.exists():
            with np.load(f, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
    raise FileNotFoundError(f"{name}.npz not in {[str(d) for d in dirs]}")


def pe_parameter(rel: str, date: str = "2026-04-01", key: str | None = None) -> float:
    """Value in force at ``date`` of a policyengine-uk parameter YAML under
    gov/dwp (``key`` selects a child of a parameter node)."""
    import importlib.util

    import yaml
    spec = importlib.util.find_spec("policyengine_uk")
    path = Path(spec.origin).parent / "parameters" / "gov" / "dwp" / rel
    node = yaml.safe_load(path.read_text())
    if key is not None:
        node = node[key]
    vals = {str(k): v for k, v in node["values"].items()}  # YAML parses dates
    best = max((k for k in vals if k <= date), default=None)
    if best is None:
        raise KeyError(f"{rel}: no value in force at {date}")
    v = vals[best]
    if isinstance(v, dict):
        v = v["value"]
    return float(v)


def main() -> None:  # pragma: no cover - needs the household arrays
    base = load_arrays("baseline")
    per_draw = [flows_from_arrays(base, load_arrays(f"obr_seed{s:02d}")) for s in SEEDS]
    # cross-check against the run's own decile file (same flows, written per draw)
    ref = pd.read_csv(NR / "obr_benchmark_deciles.csv")
    for s, f in zip(SEEDS, per_draw):
        r = ref[ref.seed == s].set_index("decile")
        for col in ("market_change", "benefit_change", "tax_change", "net_income_change"):
            if not np.allclose(f[col].to_numpy(), r[col].to_numpy(), rtol=1e-4, atol=1e3):
                raise AssertionError(f"seed {s} {col}: arrays and obr_benchmark_deciles.csv disagree")
    out = decile_stabilisation(per_draw)
    out.to_csv(NR / "obr_household_deciles.csv", index=False)
    print(f"wrote {NR / 'obr_household_deciles.csv'} ({len(per_draw)} draws, checked against the decile file)")


if __name__ == "__main__":  # pragma: no cover
    main()
