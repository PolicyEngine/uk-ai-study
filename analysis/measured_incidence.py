"""Fifth incidence family: MEASURED, from Klein Teeselink (2025).

"Generative AI and Labor Market Outcomes: Evidence from the United Kingdom"
(SSRN 5516798, KCL). Diff-in-diff on ~75m UK employment spells / job postings,
2021-2025, by firm & occupation LLM exposure. Estimates used here:

  MEASURED (paper):
    - firms maximally exposed to LLMs cut total employment by 4.5%
      (18 months post-ChatGPT); highly exposed occupations: -23.4% postings
    - junior positions fell 5.8% vs 4.5% total -> junior multiplier 5.8/4.5
    - high-paying firms: -9.6% employment; low-paying firms: "almost no
      change" -> high/low relative displacement rates 9.6 : ~0
    - effects "worst in London" (qualitative only; no point estimate
      recoverable from the paper/press materials)

  AUTHOR-IMPOSED (documented, not from the paper):
    - LONDON_MULT central = 1.5 with sensitivity {1.0, 2.0}: the London
      result is qualitative, so the multiplier is a parameter, not an estimate
    - LOW_WAGE_MULT floor = 0.6 (relative to 9.6 for above-median earners):
      the paper reports "almost no change", not zero; a strict 0 would
      remove below-median earners from the risk set entirely
    - the composite tilt multiplies the four channels (independence imposed)
    - the same central 7% aggregate quota, +2.6% wage uplift and +0.4pp
      capital shock as every other family (KT estimates SHAPE, not SIZE)

Displacement draw: single all-employee quota (7% of weighted employees),
probability proportional to
    shifted C-AIOE exposure x junior mult x London mult x wage-tier mult.
Wage and capital shocks are the standard eq 3.5 / capital mechanics.

Outputs results/incidence/measured.json (same schema as the other four
families + provenance and London-sensitivity blocks) and
results/incidence/summary_five.csv.
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
    build_shocked_simulation,
)

DATA = Path("data")
OUT = Path("results/incidence")
PERIOD = 2026
SEED = 0

JUNIOR_MULT = 5.8 / 4.5          # KT: junior -5.8% vs total -4.5%
HIGH_WAGE_MULT = 9.6             # KT: high-paying firms -9.6%
LOW_WAGE_MULT = 0.6              # author floor for KT's "almost no change"
LONDON_MULT_CENTRAL = 1.5        # author-imposed (KT London result qualitative)
LONDON_MULT_SENS = (1.0, 2.0)


def composite_tilt(persons: pd.DataFrame, london_mult: float) -> np.ndarray:
    """Unnormalised draw weights over employees per the KT gradient."""
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
    tilt = tilt * np.where(persons["region"].to_numpy() == "LONDON", london_mult, 1.0)
    tilt = tilt * np.where(earnings >= med, HIGH_WAGE_MULT, LOW_WAGE_MULT)
    return tilt


def draw_measured(persons: pd.DataFrame, scenario, london_mult: float, seed=SEED):
    """All-employee weighted quota draw, p proportional to the composite tilt."""
    rng = np.random.default_rng(seed)
    w = persons["weight"].to_numpy()
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    quota = scenario.displacement_rate * float(w[employed].sum())
    members = np.flatnonzero(employed)
    p = composite_tilt(persons, london_mult)[members]
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
    return displaced


def measured_table(persons: pd.DataFrame, scenario, london_mult: float) -> pd.DataFrame:
    shocked = persons.copy()
    displaced = draw_measured(persons, scenario, london_mult)
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
    persons = pd.DataFrame(
        {
            "person_id": calc("person_id"),
            "age": calc("age"),
            "employment_income": calc("employment_income"),
            "savings_interest_income": calc("savings_interest_income"),
            "dividend_income": calc("dividend_income"),
            "weight": calc("person_weight"),
            "region": baseline.calculate("region", period=PERIOD, map_to="person").values,
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
    scenario = PRESETS["central"]

    # London-multiplier sensitivity on displacement composition (draws only)
    london = persons["region"].to_numpy() == "LONDON"
    employed = persons["employment_income"].to_numpy() > 0
    sens = {}
    for lm in sorted(set(LONDON_MULT_SENS) | {LONDON_MULT_CENTRAL}):
        d = draw_measured(persons, scenario, lm)
        sens[lm] = {
            "london_share_of_displaced_pct": float(100 * w[d & london].sum() / w[d].sum()),
            "london_displacement_rate_pct": float(100 * w[d & london].sum() / w[employed & london].sum()),
        }

    table = measured_table(persons, scenario, LONDON_MULT_CENTRAL)
    sim = build_shocked_simulation(ds, baseline, table, PERIOD)
    m = metrics(sim)
    displaced = table["displaced"].to_numpy()
    delta = m["hni"] - b["hni"]
    rec = {
        "family": "measured",
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
        "london_sensitivity": {str(k): v for k, v in sens.items()},
        "provenance": {
            "source": "Klein Teeselink (2025), 'Generative AI and Labor Market "
                      "Outcomes: Evidence from the United Kingdom', SSRN 5516798; "
                      "KCL press release 2025-09.",
            "from_paper": {
                "total_employment_effect_exposed_firms_pct": -4.5,
                "junior_employment_effect_pct": -5.8,
                "junior_multiplier": JUNIOR_MULT,
                "high_paying_firms_employment_effect_pct": -9.6,
                "low_paying_firms_employment_effect": "almost no change (qualitative)",
                "london": "effects 'worst in London' (qualitative only)",
                "exposed_occupations_postings_pct": -23.4,
                "high_salary_occupations_postings_pct": -34.2,
            },
            "author_imposed": {
                "london_multiplier_central": LONDON_MULT_CENTRAL,
                "london_multiplier_sensitivity": list(LONDON_MULT_SENS),
                "low_wage_multiplier_floor": LOW_WAGE_MULT,
                "high_wage_multiplier": HIGH_WAGE_MULT,
                "junior_proxy": "junior positions proxied by age < 25 (KT's concept "
                                "is seniority, not age; under-25s are ~2% of the "
                                "displaced given the wage-tier tilt)",
                "wage_tier_split": "weighted median of employee earnings (FRS person "
                                   "earnings proxy for KT's firm pay tiers)",
                "composite": "multiplicative combination of exposure x junior x "
                             "London x wage-tier channels (independence imposed)",
                "aggregate_shock": "central preset (7% displacement, +2.6% wages, "
                                   "+0.4pp capital) — KT shapes incidence, not size",
            },
        },
    }
    (OUT / "measured.json").write_text(json.dumps(rec, indent=2))
    print("measured", {k: round(v, 2) for k, v in rec.items() if isinstance(v, float)}, flush=True)
    print("london sensitivity", sens, flush=True)

    # combined five-family summary
    prev = pd.read_csv(OUT / "summary.csv")
    row = {k: v for k, v in rec.items() if not isinstance(v, dict)}
    pd.concat([prev[prev.family != "measured"], pd.DataFrame([row])], ignore_index=True).to_csv(
        OUT / "summary_five.csv", index=False
    )


if __name__ == "__main__":
    main()
