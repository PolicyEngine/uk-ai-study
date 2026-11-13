#!/usr/bin/env python3
"""Wage-cut run on the household basis: written 13 Nov 2026, not yet run.

Why. The paper compares job loss and wage cuts at the same gross earnings
loss (Table 3). For job loss the household-basis absorption (income tax,
employee NI and benefits; the stabilisation coefficient of Dolls et al.,
2012) is available per draw (run 15, ``results/next_runs/``); for wage cuts
only the government balance of one seed-0 run exists. This script records
the wage-cut scenario on the household basis over the 20 seeds of the paired
draws, by baseline income decile and household family type, so that the
coefficient 1 - dYD/dYM is computed in the same way for both margins.

Scenario. Exactly the wage-margin definition of Table 3
(``WAGE_MARGIN_PRESETS["wage_margin_central"]`` in ``uk_ai_study.shocks``):
no one loses a job; every employee's earnings fall by an occupation-level
percentage proportional to the min-shifted C-AIOE, scaled so the aggregate
gross cut equals the baseline earnings of the PAIRED central 7 per cent
displacement draw of the same seed (``apply_wage_margin_shock`` asserts the
identity); the +2.6 per cent complementarity uplift and the capital shock
apply on top.

Variants (variant-major, so a stopped run leaves whole variants complete):
    table3     the Table 3 definition above (uplift and capital shock on);
    cut_only   the same gross cut with no uplift and no capital shock. Job
               loss is measured displacement-only (run 15 and the channel
               split), so this is the like-for-like intensive-margin
               counterpart for the stabilisation coefficient.
``--variants table3`` runs the Table 3 definition alone (20 simulations).

Nothing in ``uk_ai_study/`` or ``analysis/next_runs.py`` is modified: their
helpers are imported. The checkpoint fingerprint covers ``uk_ai_study/*.py``,
this script and ``analysis/next_runs.py``.

Run (one heavy job at a time; respect /tmp/pe-heavy.lock):
    python analysis/wage_cut_household.py --resume

Outputs (results/wage_cut/):
    wage_cut_draws.csv          one row per (variant, seed), stamped
    wage_cut_deciles.csv        flows by baseline decile per draw
    wage_cut_decile_family.csv  flows by decile x family type per draw
    wage_cut.json               means, SDs and SEs per variant, plus the
                                aggregate stabilisation coefficients
Household arrays per draw go to ``../ai-study-wage-cut-arrays/`` (outside
the repository, as for runs 15 and 16).
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import next_runs as nr  # noqa: E402
from uk_ai_study.checkpoint import (  # noqa: E402
    Checkpoint,
    certified_data_build_id,
    code_fingerprint,
)
from uk_ai_study.shocks import WAGE_MARGIN_PRESETS, WageMarginScenario  # noqa: E402

OUT = ROOT / "results" / "wage_cut"
ARRAYS_DIR = ROOT.parent / "ai-study-wage-cut-arrays"
PERIOD = nr.PERIOD
SEEDS = nr.SEEDS  # 0-19: the seeds of the channel split and runs 15-16

TABLE3 = WAGE_MARGIN_PRESETS["wage_margin_central"]
VARIANTS: dict[str, WageMarginScenario] = {
    "table3": TABLE3,
    "cut_only": replace(TABLE3, name="wage_cut_only", wage_uplift=0.0, capital_return_increase=0.0),
}

DRAW_COLS = ("exchequer_cost_bn", "household_basis_cost_bn", "employer_ni_change_bn",
             "income_tax_change_bn", "ni_employee_change_bn", "uc_change_bn",
             "other_benefits_change_bn", "benefits_change_bn", "childcare_change_bn",
             "statutory_pay_change_bn", "market_income_change_bn", "net_income_change_bn",
             "gross_cut_bn", "gross_cut_share_pct", "survivor_gain_bn", "capital_income_added_bn",
             "poverty_change_bhc_pp", "poverty_change_ahc_pp", "relative_poverty_change_bhc_pp",
             "gini_change_pp", "stabilisation_net", "stabilisation_tax_benefit")


def fingerprint() -> str:
    """Code fingerprint of this run: the package, this script and next_runs.py
    (whose helpers it imports), so a change to any of them stales the run."""
    h = hashlib.sha256(code_fingerprint(Path(__file__)).encode())
    h.update((ROOT / "analysis" / "next_runs.py").read_bytes())
    return "c" + h.hexdigest()[:16]


def stamp() -> dict:
    return {"certified_data_build_id": certified_data_build_id(), "code_fingerprint": fingerprint()}


def stabilisation(d_market: float, d_net: float) -> float:
    """Dolls et al. (2012) income stabilisation coefficient 1 - dYD/dYM:
    the share of a market-income change that does not reach disposable
    income. NaN when market income does not change."""
    return float("nan") if d_market == 0 else 1.0 - d_net / d_market


def tax_benefit_stabilisation(d_market: float, d_benefits: float, d_tax: float) -> float:
    """(dBenefits - dTax) / -dYM: the part of the coefficient carried by
    income tax, employee NI and benefits alone. It differs from
    ``stabilisation`` by the items between market and HBAI disposable income
    that are neither (pension contributions, council tax), which job loss
    zeroes and a wage cut does not."""
    return float("nan") if d_market == 0 else (d_benefits - d_tax) / -d_market


def household_flows(group, weights, d: dict) -> pd.DataFrame:
    """Weighted sums of household changes by ``group`` (a decile array, or a
    list of arrays for decile x family), with both coefficients.

    ``d`` maps market, net, income_tax, ni_employee, uc, benefits to
    household-level change arrays (pounds)."""
    keys = list(group) if isinstance(group, (list, tuple)) else [group]
    names = ["decile", "family_type"][: len(keys)]
    df = pd.DataFrame({n: np.asarray(k) for n, k in zip(names, keys)})
    w = np.asarray(weights, float)
    cols = {"market_change": "market", "net_income_change": "net", "income_tax_change": "income_tax",
            "ni_employee_change": "ni_employee", "uc_change": "uc", "benefit_change": "benefits"}
    for c, k in cols.items():
        df[c] = np.asarray(d[k], float) * w
    df["weight"] = w
    g = df.groupby(names)[list(cols) + ["weight"]].sum()
    g["other_benefit_change"] = g.benefit_change - g.uc_change
    g["tax_change"] = g.income_tax_change + g.ni_employee_change
    g["stabilisation_net"] = [stabilisation(m, n) for m, n in zip(g.market_change, g.net_income_change)]
    g["stabilisation_tax_benefit"] = [tax_benefit_stabilisation(m, b, t) for m, b, t in
                                      zip(g.market_change, g.benefit_change, g.tax_change)]
    return g.reset_index()


def changes(b: dict, s: dict) -> dict:
    """Household change arrays from two ``next_runs._sim_metrics`` dicts."""
    return {"market": s["hh_market"] - b["hh_market"], "net": s["hh_net"] - b["hh_net"],
            "income_tax": s["hh_it"] - b["hh_it"], "ni_employee": s["hh_ni"] - b["hh_ni"],
            "uc": s["hh_uc"] - b["hh_uc"], "benefits": s["hh_benefits"] - b["hh_benefits"]}


def draw_row(b: dict, s: dict, persons: pd.DataFrame, table: pd.DataFrame) -> dict:
    """One draw's aggregates: next_runs' columns plus the household-basis
    split and both stabilisation coefficients."""
    row = nr._row(b, s, persons, table)
    d = changes(b, s)
    hw = np.asarray(b["hw"], float)
    agg = {k: float((v * hw).sum()) for k, v in d.items()}
    row.update({
        "other_benefits_change_bn": (agg["benefits"] - agg["uc"]) / 1e9,
        "market_income_change_bn": agg["market"] / 1e9,
        "net_income_change_bn": agg["net"] / 1e9,
        "gross_cut_bn": float(table.attrs.get("gross_cut_realised", np.nan)) / 1e9,
        "gross_cut_share_pct": 100 * float(table.attrs.get("gross_cut_share_target", np.nan)),
        "stabilisation_net": stabilisation(agg["market"], agg["net"]),
        "stabilisation_tax_benefit": tax_benefit_stabilisation(
            agg["market"], agg["benefits"], agg["income_tax"] + agg["ni_employee"]),
    })
    return row


def summarise(draws: pd.DataFrame) -> dict:
    """Per-variant means, SDs and SEs, and coefficients of the summed flows
    (ratio of means, the aggregate counterpart of the per-draw ratios)."""
    out = {}
    for v, d in draws.groupby("variant", sort=False):
        s = nr.summarise(d, [c for c in DRAW_COLS if c in d.columns])
        m = d[["market_income_change_bn", "net_income_change_bn", "benefits_change_bn",
               "income_tax_change_bn", "ni_employee_change_bn"]].mean()
        s["pooled"] = {
            "stabilisation_net": stabilisation(m.market_income_change_bn, m.net_income_change_bn),
            "stabilisation_tax_benefit": tax_benefit_stabilisation(
                m.market_income_change_bn, m.benefits_change_bn,
                m.income_tax_change_bn + m.ni_employee_change_bn),
            "draws": int(len(d)),
        }
        out[str(v)] = s
    return out


def parse_variants(argv) -> list[str]:
    if "--variants" not in argv:
        return list(VARIANTS)
    names = argv[argv.index("--variants") + 1].split(",")
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; choose from {sorted(VARIANTS)}")
    return [v for v in VARIANTS if v in names]  # keep the canonical order


# ---------------------------------------------------------------------------
# Simulation driver (needs Microcosm; not executed when written)
# ---------------------------------------------------------------------------
def _append(path: Path, df: pd.DataFrame, key: dict) -> None:  # pragma: no cover
    nr._append_deciles(path, df, key)


def main(argv=None) -> None:  # pragma: no cover - needs Microcosm
    import gc

    from uk_ai_study.data import microsimulation
    from uk_ai_study.runner import build_person_table, weighted_deciles
    from uk_ai_study.shocks import TRANSITION_ZEROED_VARIABLES, apply_wage_margin_shock, \
        build_shocked_simulation

    argv = sys.argv if argv is None else argv
    variants = parse_variants(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = microsimulation()
    persons = build_person_table(baseline, PERIOD)
    b = nr._sim_metrics(baseline)
    arrays = {v: baseline.calculate(v, period=PERIOD, map_to="person").values.astype(float)
              for v in TRANSITION_ZEROED_VARIABLES}
    arrays["employment_status"] = baseline.calculate(
        "employment_status", period=PERIOD, map_to="person").values.astype(object)
    # the deciles and family types of runs 15-16: baseline equivalised HBAI
    # income, households weighted by members; family type of the first person
    hh_decile = weighted_deciles(b["equiv"], b["hw"] * b["count"])
    hid_p = baseline.calculate("household_id", period=PERIOD, map_to="person").values
    hid_h = baseline.calculate("household_id", period=PERIOD, map_to="household").values
    fam_p = baseline.calculate("family_type", period=PERIOD, map_to="person").values.astype(str)
    family = pd.Series(fam_p).groupby(hid_p, sort=False).first().reindex(hid_h).to_numpy().astype(str)
    np.savez_compressed(ARRAYS_DIR / "baseline.npz", weight=b["hw"], decile=hh_decile, family=family,
                        **{k: b[k] for k in ("hh_market", "hh_net", "hh_it", "hh_ni", "hh_benefits", "hh_uc")})
    del baseline
    gc.collect()

    ck = Checkpoint(OUT / "wage_cut_draws.csv", ("variant", "seed"), stamp=stamp())
    dec_path, fam_path = OUT / "wage_cut_deciles.csv", OUT / "wage_cut_decile_family.csv"
    if not ck.done:  # restart the per-draw side files with the draws
        for f in (dec_path, fam_path):
            f.unlink(missing_ok=True)
    for name in variants:
        scen = VARIANTS[name]
        for seed in SEEDS:
            if ck.is_done(name, seed):
                continue
            table = apply_wage_margin_shock(persons, scen, seed=seed)
            sim = build_shocked_simulation(None, table, PERIOD, base_arrays=arrays)
            s = nr._sim_metrics(sim)
            del sim
            gc.collect()
            d = changes(b, s)
            key = {"variant": name, "seed": seed}
            _append(dec_path, household_flows(hh_decile, b["hw"], d), dict(key))
            _append(fam_path, household_flows([hh_decile, family], b["hw"], d), dict(key))
            np.savez_compressed(ARRAYS_DIR / f"{name}_seed{seed:02d}.npz", **{k: s[k] for k in (
                "hh_market", "hh_net", "hh_it", "hh_ni", "hh_benefits", "hh_uc")})
            ck.append({**key, **draw_row(b, s, persons, table)})
            print(f"wage_cut {name} seed {seed} done", flush=True)
    draws = ck.read()
    (OUT / "wage_cut.json").write_text(json.dumps(
        {"variants": {k: VARIANTS[k].__dict__ for k in variants}, "summary": summarise(draws)},
        indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
