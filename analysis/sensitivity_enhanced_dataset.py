"""Referee M7: decompose the +3.9pp (enhanced FRS 2023-24, period 2025, geo
pipeline) vs +1.87pp (plain FRS 2024-25, period 2026, main pipeline) BHC
poverty-change gap into dataset and period components.

Runs the central scenario (7% displacement, +2.6% wages, +0.4pp capital),
seed 0, with geo_impact.py's SOC imputation on the enhanced dataset, at:
  (enhanced, 2026)  -- same period as the main analysis (dataset component)
  (enhanced, 2025)  -- the geography section's setting (period component)
  (plain,    2025)  -- closes the 2x2 (interaction check)
The (plain, 2026) cell is the paper's main result (results/central.json).

Output: results/robustness/enhanced_dataset_central.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group  # noqa: E402
from uk_ai_study.shocks import PRESETS, apply_shocks, build_shocked_simulation  # noqa: E402
from geo_impact import age_band, person_frame, weighted_decile_edges  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "robustness"
PLAIN_FRS = DATA / "frs_2024_25.h5"
ENHANCED_FRS = Path(
    "/Users/janansadeqian/policyengine-uk-data/policyengine_uk_data/storage/enhanced_frs_2023_24.h5"
)
ADULT_TAB = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
SEED = 0
SCENARIO = PRESETS["central"]
CKPT = OUT / "enhanced_dataset_cells.csv"


def poverty_bhc(sim, period):
    pw = sim.calculate("person_weight", period=period, map_to="person").values
    pov = sim.calculate("in_poverty_bhc", period=period, map_to="person").values
    return float(np.average(pov, weights=pw))


def donor_distributions(period):
    """Plain-FRS weighted SOC distribution by cell (geo_impact route B)."""
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    plain = Microsimulation(dataset=UKSingleYearDataset(file_path=str(PLAIN_FRS)))
    donor = person_frame(plain, period)
    donor["soc"] = attach_soc_major_group(donor["person_id"], ADULT_TAB)
    demp = donor[donor["employment_income"] > 0].copy()
    demp = demp[np.isfinite(demp["soc"])]
    edges = weighted_decile_edges(
        demp["employment_income"].to_numpy(float), demp["weight"].to_numpy(float)
    )
    demp["band"] = age_band(demp["age"].to_numpy())
    demp["dec"] = np.digitize(demp["employment_income"], edges)

    def dist(frame):
        g = frame.groupby("soc")["weight"].sum()
        return g.index.to_numpy(float), np.cumsum((g / g.sum()).to_numpy(float))

    full = {k: dist(g) for k, g in demp.groupby(["band", "gender", "region", "dec"])}
    fb1 = {k: dist(g) for k, g in demp.groupby(["band", "gender", "dec"])}
    fb2 = {k: dist(g) for k, g in demp.groupby(["band", "gender"])}
    marginal = dist(demp)
    del plain, donor
    return edges, full, fb1, fb2, marginal


def run_cell(dataset_name, period):
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    if dataset_name == "plain":
        from uk_ai_study.runner import run_scenario

        r = run_scenario(PLAIN_FRS, ADULT_TAB, "central", period=period, seed=SEED)
        return {
            "poverty_change_bhc_pp": 100 * r.poverty_rate_change_bhc,
            "exchequer_cost_bn": r.exchequer_cost / 1e9,
            "displaced_weighted_m": r.displaced_weighted / 1e6,
        }

    edges, full, fb1, fb2, marginal = donor_distributions(period)
    dataset = UKSingleYearDataset(file_path=str(ENHANCED_FRS))
    baseline = Microsimulation(dataset=dataset)
    persons = person_frame(baseline, period)
    emp = persons["employment_income"].to_numpy(float) > 0
    band = age_band(persons["age"].to_numpy())
    dec = np.digitize(persons["employment_income"], edges)
    gender = persons["gender"].to_numpy()
    region = persons["region"].to_numpy()

    rng = np.random.default_rng(SEED)
    soc = np.full(len(persons), np.nan)
    for i in np.flatnonzero(emp):
        for key, table in (
            ((band[i], gender[i], region[i], dec[i]), full),
            ((band[i], gender[i], dec[i]), fb1),
            ((band[i], gender[i]), fb2),
        ):
            if key in table:
                codes, p = table[key]
                break
        else:
            codes, p = marginal
        soc[i] = codes[min(np.searchsorted(p, rng.random()), len(codes) - 1)]

    persons["soc_major_group"] = soc
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
    persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))

    table = apply_shocks(persons, SCENARIO, seed=SEED)
    shocked = build_shocked_simulation(dataset, baseline, table, period)

    pov0 = poverty_bhc(baseline, period)
    pov1 = poverty_bhc(shocked, period)
    hw0 = baseline.calculate("household_weight", period=period, map_to="household").values
    gov0 = float((baseline.calculate("gov_balance", period=period, map_to="household").values * hw0).sum())
    hw1 = shocked.calculate("household_weight", period=period, map_to="household").values
    gov1 = float((shocked.calculate("gov_balance", period=period, map_to="household").values * hw1).sum())
    w = persons["weight"].to_numpy()
    displaced = table["displaced"].to_numpy()
    return {
        "poverty_change_bhc_pp": 100 * (pov1 - pov0),
        "baseline_poverty_bhc_pct": 100 * pov0,
        "exchequer_cost_bn": (gov0 - gov1) / 1e9,
        "displaced_weighted_m": float(w[displaced].sum() / 1e6),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not ENHANCED_FRS.exists():
        out = {"status": "skipped", "reason": f"enhanced dataset not found at {ENHANCED_FRS}"}
        (OUT / "enhanced_dataset_central.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out))
        return

    done = {}
    if CKPT.exists():
        for _, r in pd.read_csv(CKPT).iterrows():
            done[r["cell"]] = json.loads(r["payload"])

    cells = [("enhanced", 2026), ("enhanced", 2025), ("plain", 2025)]
    for name, period in cells:
        key = f"{name}_{period}"
        if key in done:
            continue
        res = run_cell(name, period)
        done[key] = res
        pd.DataFrame([{"cell": key, "payload": json.dumps(res)}]).to_csv(
            CKPT, mode="a", header=not CKPT.exists(), index=False
        )
        print(key, res, flush=True)

    plain_2026 = json.loads((ROOT / "results" / "central.json").read_text())
    done["plain_2026"] = {
        "poverty_change_bhc_pp": 100 * plain_2026["poverty_rate_change_bhc"],
        "exchequer_cost_bn": plain_2026["exchequer_cost"] / 1e9,
        "displaced_weighted_m": plain_2026["displaced_weighted"] / 1e6,
        "source": "results/central.json (main analysis, seed 0)",
    }

    p = {k: done[k]["poverty_change_bhc_pp"] for k in
         ("plain_2026", "enhanced_2026", "enhanced_2025", "plain_2025")}
    out = {
        "description": "Central scenario, seed 0; enhanced FRS 2023-24 with "
                       "geo_impact.py's SOC imputation (route B). Decomposes "
                       "the geography-section (+enhanced, 2025) vs main "
                       "(+plain, 2026) BHC poverty-change gap.",
        "enhanced_dataset_path": str(ENHANCED_FRS),
        "cells": done,
        "decomposition_bhc_poverty_pp": {
            "total_gap_enhanced2025_minus_plain2026": p["enhanced_2025"] - p["plain_2026"],
            "dataset_component_at_2026": p["enhanced_2026"] - p["plain_2026"],
            "period_component_on_enhanced": p["enhanced_2025"] - p["enhanced_2026"],
            "period_component_on_plain": p["plain_2025"] - p["plain_2026"],
            "interaction": (p["enhanced_2025"] - p["enhanced_2026"])
                           - (p["plain_2025"] - p["plain_2026"]),
            "note": "single seed-0 run per cell; imputation noise not "
                    "separately varied (imputation seed fixed at 0).",
        },
    }
    (OUT / "enhanced_dataset_central.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
