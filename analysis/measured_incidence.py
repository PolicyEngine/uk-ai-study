"""Klein-anchored top-loaded incidence stress test.

"Generative AI and Labor Market Outcomes: Evidence from the United Kingdom"
(SSRN 5516798, KCL). Diff-in-diff on ~75m UK employment spells / job postings,
2021-2025, by firm & occupation LLM exposure. Estimates used here:

  PAPER ESTIMATES (December 2025 revision, per standard deviation):
    - total employment about -0.3%; junior employment about -0.4%
    - high-compensation-firm employment about -0.7%; low-compensation effect
      close to zero; London effect minimal and statistically insignificant
  PRESS-RELEASE / MAXIMUM-EXPOSURE SCALINGS:
    - 4.5% total, 5.8% junior and 9.6% high-paying-firm employment effects
  AUTHOR-IMPOSED (documented, not from the paper):
    - LOW_WAGE_MULT floor = 0.6 (relative to 9.6 for above-median earners):
      the paper reports "almost no change", not zero; a strict 0 would
      remove below-median earners from the risk set entirely
    - the composite tilt multiplies exposure, junior and wage-tier channels
      (independence imposed). No geographic tilt is applied: the December
      2025 paper reports minimal, statistically insignificant London effects.
    - the same central 7% aggregate quota, +2.6% wage uplift and +0.4pp
      capital shock as every other family (KT estimates SHAPE, not SIZE)

Displacement draw: single all-employee quota (7% of weighted employees),
probability proportional to shifted C-AIOE exposure x junior mult x wage-tier
mult. The retained ``london_mult`` argument must equal 1.0 and exists only to
keep older analysis callers source-compatible.
Wage and capital shocks are the standard eq 3.5 / capital mechanics.

Outputs results/incidence/klein_top_loaded.json (same schema as the other four
families + provenance and London-sensitivity blocks) and
results/incidence/summary_five.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_ai_study.runner import build_person_table, gini
from uk_ai_study.shocks import (
    BASELINE_CAPITAL_RETURN,
    PRESETS,
    build_shocked_simulation,
    prescribed_systematic_sample,
)

DATA = Path("data")
OUT = Path("results/incidence")
PERIOD = 2026
SEED = 0

JUNIOR_MULT = 5.8 / 4.5          # press-release maximum-exposure scaling
HIGH_WAGE_MULT = 9.6             # press-release maximum-exposure scaling
LOW_WAGE_MULT = 0.6              # author floor for KT's "almost no change"
LONDON_MULT_CENTRAL = 1.0        # neutral: revised paper contradicts a London tilt


def composite_tilt(
    persons: pd.DataFrame,
    london_mult: float,
    high_wage_mult: float = HIGH_WAGE_MULT,
    low_wage_mult: float = LOW_WAGE_MULT,
) -> np.ndarray:
    """Unnormalised draw weights over employees per the KT gradient."""
    if london_mult != 1.0:
        raise ValueError("London tilt removed: london_mult must equal 1.0")
    earnings = persons["employment_income"].to_numpy(dtype=float)
    w = persons["weight"].to_numpy()
    employed = earnings > 0
    exposure = persons["exposure"].to_numpy()
    exposure = exposure - exposure[employed].min()  # JR16 shift: least-exposed -> 0

    order = np.argsort(earnings[employed])
    cw = np.cumsum(w[employed][order])
    med = earnings[employed][order][np.searchsorted(cw, 0.5 * cw[-1])]

    tilt = np.where(exposure > 0, exposure, 1e-9)
    tilt = tilt * np.where(persons["age"].to_numpy() < 25, JUNIOR_MULT, 1.0)
    tilt = tilt * np.where(earnings >= med, high_wage_mult, low_wage_mult)
    return tilt


def draw_measured(
    persons: pd.DataFrame,
    scenario,
    london_mult: float,
    seed=SEED,
    high_wage_mult: float = HIGH_WAGE_MULT,
    low_wage_mult: float = LOW_WAGE_MULT,
):
    """All-employee weighted quota draw, p proportional to the composite tilt."""
    w = persons["weight"].to_numpy()
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    quota = scenario.displacement_rate * float(w[employed].sum())
    members = np.flatnonzero(employed)
    displaced = np.zeros(len(persons), dtype=bool)
    displaced[members] = prescribed_systematic_sample(
        w[members],
        quota,
        composite_tilt(persons, london_mult, high_wage_mult, low_wage_mult)[members],
        np.random.default_rng(seed),
    )
    return displaced


def measured_table(
    persons: pd.DataFrame,
    scenario,
    london_mult: float,
    seed: int = SEED,
    high_wage_mult: float = HIGH_WAGE_MULT,
    low_wage_mult: float = LOW_WAGE_MULT,
) -> pd.DataFrame:
    shocked = persons.copy()
    displaced = draw_measured(
        persons, scenario, london_mult, seed=seed,
        high_wage_mult=high_wage_mult, low_wage_mult=low_wage_mult,
    )
    shocked["displaced"] = displaced
    earnings = persons["employment_income"].to_numpy(dtype=float)
    w = persons["weight"].to_numpy()
    employed = earnings > 0
    survivors = employed & ~displaced
    theta = persons["complementarity"].to_numpy(dtype=float)
    theta_bar = float((theta * w)[employed].sum() / w[employed].sum())
    uplift = np.zeros_like(earnings)
    if theta_bar > 0:
        uplift[survivors] = scenario.wage_uplift * (theta[survivors] / theta_bar) * earnings[survivors]
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
    persons = build_person_table(
        baseline,
        PERIOD,
        DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab",
        extra_variables=("region",),
    )

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
            # household-level %, identical for all members; broadcast
            # intentional — used only in the delta/base ratio below (issue #6)
            "hni": s.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values,
        }

    b = metrics(baseline)
    scenario = PRESETS["central"]

    # Report London composition descriptively; geography does not enter risk.
    london = persons["region"].to_numpy() == "LONDON"
    employed = persons["employment_income"].to_numpy() > 0
    d_neutral = draw_measured(persons, scenario, LONDON_MULT_CENTRAL)
    london_composition = {
        "london_share_of_displaced_pct": float(100 * w[d_neutral & london].sum() / w[d_neutral].sum()),
        "london_displacement_rate_pct": float(100 * w[d_neutral & london].sum() / w[employed & london].sum()),
    }

    table = measured_table(persons, scenario, LONDON_MULT_CENTRAL)
    sim = build_shocked_simulation(ds, baseline, table, PERIOD)
    m = metrics(sim)
    displaced = table["displaced"].to_numpy()
    delta = m["hni"] - b["hni"]
    rec = {
        "family": "klein_top_loaded",
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
        "london_composition_no_tilt": london_composition,
        "provenance": {
            "source": "Klein Teeselink (2025), 'Generative AI and Labor Market "
                      "Outcomes: Evidence from the United Kingdom', SSRN 5516798; "
                      "KCL press release 2025-09.",
            "paper_per_standard_deviation_estimates_pct": {
                "total_employment": -0.3,
                "junior_employment": -0.4,
                "high_compensation_firm_employment": -0.7,
                "low_compensation_firm_employment": "close to zero",
                "london": "minimal and statistically insignificant effects",
            },
            "press_release_maximum_exposure_scalings_pct": {
                "total_employment": -4.5,
                "junior_employment": -5.8,
                "high_paying_firm_employment": -9.6,
                "exposed_occupations_postings_pct": -23.4,
                "high_salary_occupations_postings_pct": -34.2,
            },
            "author_imposed": {
                "junior_multiplier": JUNIOR_MULT,
                "geographic_multiplier": "none (London multiplier fixed at 1.0)",
                "low_wage_multiplier_floor": LOW_WAGE_MULT,
                "high_wage_multiplier": HIGH_WAGE_MULT,
                "junior_proxy": "junior positions proxied by age < 25 (KT's concept "
                                "is seniority, not age; under-25s are ~2% of the "
                                "displaced given the wage-tier tilt)",
                "wage_tier_split": "weighted median of employee earnings (FRS person "
                                   "earnings proxy for KT's firm pay tiers)",
                "composite": "multiplicative combination of exposure x junior x "
                             "wage-tier channels (independence imposed)",
                "aggregate_shock": "central preset (7% displacement, +2.6% wages, "
                                   "+0.4pp capital) — KT shapes incidence, not size",
            },
        },
    }
    (OUT / "klein_top_loaded.json").write_text(json.dumps(rec, indent=2))
    print("klein_top_loaded", {k: round(v, 2) for k, v in rec.items() if isinstance(v, float)}, flush=True)
    print("London composition (no tilt)", london_composition, flush=True)

    # combined five-family summary
    prev = pd.read_csv(OUT / "summary.csv")
    row = {k: v for k, v in rec.items() if not isinstance(v, dict)}
    pd.concat([prev[~prev.family.isin(["measured", "klein_top_loaded"])], pd.DataFrame([row])], ignore_index=True).to_csv(
        OUT / "summary_five.csv", index=False
    )


if __name__ == "__main__":
    main()
