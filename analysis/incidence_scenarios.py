"""Incidence as a first-class scenario axis (issue #1, Finding 0).

Same aggregate shock (central: 7% displacement, +2.6% wages, +0.4pp capital),
allocated under four incidence families spanning the live debate on who bears
AI displacement:

  exposure      — exposure-proportional (JR16/IMF school; the original
                  central scenario)
  junior        — junior-concentrated (Klein Teeselink / Hosseini-Lichtinger /
                  Canaries school): draws tilted toward ages 16-24 by the
                  KT junior/total ratio 5.8/4.5
  compression   — expertise-compression (Autor school, stylised): AI erodes
                  elite expertise rents, so displacement draws are tilted
                  toward the top weighted-earnings tertile of workers in
                  above-median-exposure occupations (multiplier 2), and the
                  wage uplift is allocated by INVERSE complementarity rank
                  (mid-skill capability gains). An author-designed stress
                  test, not a calibrated implementation of Autor (2024).
  uniform       — equal exposure and complementarity for everyone

Outputs results/incidence/<family>.json plus a combined summary CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.runner import gini
from uk_ai_study.shocks import (
    BASELINE_CAPITAL_RETURN,
    PRESETS,
    apply_shocks,
    build_shocked_simulation,
    draw_displaced,
)

DATA = Path("data")
OUT = Path("results/incidence")
PERIOD = 2026
SEED = 0
COMPRESSION_TOP_TERTILE_MULTIPLIER = 2.0


def _weighted_tertile_threshold(values, weights, q):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return values[order][np.searchsorted(cw, q * cw[-1])]


def shocked_table_for(family: str, persons: pd.DataFrame) -> pd.DataFrame:
    scenario = PRESETS["central"]
    if family == "exposure":
        return apply_shocks(persons, scenario, seed=SEED)
    if family == "junior":
        return apply_shocks(persons, PRESETS["central_youth_tilted"], seed=SEED)
    if family == "uniform":
        uniform = persons.copy()
        uniform["exposure"] = 1.0
        uniform["complementarity"] = 1.0
        return apply_shocks(uniform, scenario, seed=SEED)
    if family == "compression":
        return compression_table(persons, scenario)
    raise ValueError(family)


def compression_table(persons: pd.DataFrame, scenario) -> pd.DataFrame:
    """Stylised Autor-school variant: top-rent erosion + mid-skill gains."""
    shocked = persons.copy()
    w = persons["weight"].to_numpy()
    earnings = persons["employment_income"].to_numpy(dtype=float)
    exposure = persons["exposure"].to_numpy()
    employed = earnings > 0

    med_exp = _weighted_tertile_threshold(exposure[employed], w[employed], 0.5)
    top_earn = _weighted_tertile_threshold(earnings[employed], w[employed], 2 / 3)
    elite = employed & (exposure > med_exp) & (earnings >= top_earn)

    # reuse the eq 3.4 quota machinery by tilting draws toward the elite
    # group, exactly as the youth multiplier tilts toward the young: encode
    # the tilt in a synthetic "age" channel is NOT possible, so draw here
    # with the same quota-fill logic at the all-employee level.
    rng = np.random.default_rng(SEED)
    quota = scenario.displacement_rate * float(w[employed].sum())
    members = np.flatnonzero(employed)
    p = np.where(elite[members], COMPRESSION_TOP_TERTILE_MULTIPLIER, 1.0)
    p = p / p.sum()
    displaced = np.zeros(len(persons), dtype=bool)
    chosen = rng.choice(members, size=len(members), replace=False, p=p)
    cum = np.cumsum(w[chosen])
    displaced[chosen[cum <= quota]] = True
    crossing = np.searchsorted(cum, quota)
    if crossing < len(chosen) and cum[crossing] > quota:
        shortfall = quota - (cum[crossing - 1] if crossing else 0.0)
        if rng.random() < shortfall / w[chosen[crossing]]:
            displaced[chosen[crossing]] = True
    shocked["displaced"] = displaced

    # wage uplift by INVERSE complementarity: theta' = max(theta) + min(theta)
    # - theta flips the gradient (mid-skill gains), keeping the same
    # employment-weighted normalisation as eq 3.5
    theta = persons["complementarity"].to_numpy(dtype=float)
    theta_inv = theta[employed].max() + theta[employed].min() - theta
    survivors = employed & ~displaced
    theta_bar = float((theta_inv * w)[employed].sum() / w[employed].sum())
    uplift = np.zeros_like(earnings)
    if theta_bar > 0:
        uplift[survivors] = scenario.wage_uplift * (theta_inv[survivors] / theta_bar) * earnings[survivors]
    shocked["employment_income"] = np.where(displaced, 0.0, earnings + uplift)

    factor = (BASELINE_CAPITAL_RETURN + scenario.capital_return_increase) / BASELINE_CAPITAL_RETURN
    for col in ("savings_interest_income", "dividend_income"):
        shocked[col] = shocked[col].to_numpy(dtype=float) * factor
    return shocked


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    baseline = Microsimulation(dataset=ds)
    calc = lambda v: baseline.calculate(v, period=PERIOD, map_to="person").values
    persons = pd.DataFrame(
        {
            "person_id": calc("person_id"),
            "age": calc("age"),
            "employment_income": calc("employment_income"),
            "savings_interest_income": calc("savings_interest_income"),
            "dividend_income": calc("dividend_income"),
            "weight": calc("person_weight"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(
        persons["person_id"], DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
    )
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
    persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))

    equiv = calc("equiv_hbai_household_net_income")
    w = persons["weight"].to_numpy()
    order = np.argsort(equiv)
    cw = np.cumsum(w[order])
    ranks = np.empty(len(equiv))
    ranks[order] = cw / cw[-1]
    dec = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)

    def metrics(s):
        pw = s.calculate("person_weight", period=PERIOD, map_to="person").values
        hw = s.calculate("household_weight", period=PERIOD, map_to="household").values
        eq = s.calculate("equiv_hbai_household_net_income", period=PERIOD, map_to="household").values
        n = s.calculate("household_count_people", period=PERIOD, map_to="household").values
        return {
            "gov": float((s.calculate("gov_balance", period=PERIOD, map_to="household").values * hw).sum()),
            "pov_bhc": float(np.average(s.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values, weights=pw)),
            "pov_ahc": float(np.average(s.calculate("in_poverty_ahc", period=PERIOD, map_to="person").values, weights=pw)),
            "gini": gini(eq, hw * n),
            "hni": s.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values,
        }

    b = metrics(baseline)
    summary = []
    for family in ("exposure", "junior", "compression", "uniform"):
        table = shocked_table_for(family, persons)
        sim = build_shocked_simulation(ds, baseline, table, PERIOD)
        m = metrics(sim)
        displaced = table["displaced"].to_numpy()
        delta = m["hni"] - b["hni"]
        rec = {
            "family": family,
            "displaced_weighted_m": float(w[displaced].sum() / 1e6),
            "exchequer_cost_bn": (b["gov"] - m["gov"]) / 1e9,
            "poverty_change_bhc_pp": 100 * (m["pov_bhc"] - b["pov_bhc"]),
            "poverty_change_ahc_pp": 100 * (m["pov_ahc"] - b["pov_ahc"]),
            "gini_change_pp": 100 * (m["gini"] - b["gini"]),
            "decile_transition_share_pct": {
                int(d): float(100 * w[(dec == d) & displaced].sum() / w[dec == d].sum())
                for d in range(1, 11)
            },
            "decile_income_change_pct": {
                int(d): float(100 * (delta[dec == d] * w[dec == d]).sum()
                              / (b["hni"][dec == d] * w[dec == d]).sum())
                for d in range(1, 11)
            },
        }
        (OUT / f"{family}.json").write_text(json.dumps(rec, indent=2))
        summary.append({k: v for k, v in rec.items() if not isinstance(v, dict)})
        print(family, {k: round(v, 2) for k, v in summary[-1].items() if k != "family"}, flush=True)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
