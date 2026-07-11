"""UK analogues of the three anchor papers' employment findings.

Each paper's headline employment effect is implemented as a displacement
rule on the FRS person table, run through PolicyEngine UK, and summarised
by age band (fiscal cost, poverty, Gini, displaced shares).

1. Klein Teeselink (2025, SSRN 5516798) — GenAI-exposed UK firms cut
   employment by 4.5%, concentrated in junior roles (-5.8%). Implemented as
   a 4.5% aggregate employee displacement, allocated by exposure (eq 3.4)
   with the junior draw probability multiplied by 5.8/4.5, no wage uplift.

2. Hosseini & Lichtinger (2026, SSRN 5425555) — junior employment at
   AI-adopting firms falls ~9% within six quarters; senior employment is
   unaffected. Implemented as displacement of 9% of employees aged under 30
   in above-median-exposure occupations; no one 30+ is displaced.

3. Brynjolfsson, Chandar & Chen (2025, "Canaries in the Coal Mine?") —
   ~13% relative employment decline for workers aged 22-25 in the
   most-exposed occupations. Implemented as displacement of 13% of
   employees aged 22-25 in the top exposure quintile of occupations.

All three are employment-only scenarios (no wage or capital shock), so
differences across them are driven purely by who is displaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.runner import AGE_BANDS, gini
from uk_ai_study.shocks import ShockScenario, draw_displaced

DATA = Path("data")
OUT = Path("results/paper_scenarios")
PERIOD = 2026
SEED = 0


def displaced_klein_teeselink(persons: pd.DataFrame) -> np.ndarray:
    scenario = ShockScenario(
        "klein_teeselink", 0.045, 0.0, youth_displacement_multiplier=5.8 / 4.5
    )
    return draw_displaced(persons, scenario, seed=SEED)


def _matched_employed(persons: pd.DataFrame) -> np.ndarray:
    """Employees with an observed SOC group (the analysable population)."""
    return (persons["employment_income"].to_numpy() > 0) & np.isfinite(
        persons["soc_major_group"].to_numpy()
    )


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return values[order][np.searchsorted(cw, q * cw[-1])]


def _rate_within(persons, mask, rate, seed=SEED):
    """Displace `rate` of weighted employees inside `mask`, ∝ exposure."""
    rng = np.random.default_rng(seed)
    weight = persons["weight"].to_numpy()
    eligible = np.flatnonzero(mask)
    quota = rate * float(weight[eligible].sum())
    exposure = persons["exposure"].to_numpy()
    exposure = exposure - exposure[mask].min() + 1e-9
    p = exposure[eligible] * weight[eligible]
    p = p / p.sum()
    chosen = rng.choice(eligible, size=len(eligible), replace=False, p=p)
    displaced = np.zeros(len(persons), dtype=bool)
    cum = np.cumsum(weight[chosen])
    displaced[chosen[cum <= quota]] = True
    crossing = np.searchsorted(cum, quota)
    if crossing < len(chosen) and cum[crossing] > quota:
        shortfall = quota - (cum[crossing - 1] if crossing else 0.0)
        if rng.random() < shortfall / weight[chosen[crossing]]:
            displaced[chosen[crossing]] = True
    return displaced


#: Hosseini & Lichtinger: adopting firms cover ~16% of US employment; the
#: unscaled 9% is therefore an at-adopter effect. The scaled variant
#: (0.09 x 0.16) approximates the economy-wide junior effect.
HL_ADOPTER_EMPLOYMENT_SHARE = 0.16


def displaced_hosseini_lichtinger(persons: pd.DataFrame, scale: float = 1.0) -> np.ndarray:
    matched = _matched_employed(persons)
    exposure = persons["exposure"].to_numpy()
    weight = persons["weight"].to_numpy()
    median_exp = _weighted_quantile(exposure[matched], weight[matched], 0.5)
    mask = matched & (persons["age"].to_numpy() < 30) & (exposure > median_exp)
    return _rate_within(persons, mask, 0.09 * scale)


def displaced_brynjolfsson(persons: pd.DataFrame) -> np.ndarray:
    # Canaries fact 4: ~16% RELATIVE decline (Nov 2025 version; 13% in
    # earlier drafts) for ages 22-25 in the top exposure quintiles. Treating
    # it as an absolute displacement rate makes this an upper bound.
    matched = _matched_employed(persons)
    exposure = persons["exposure"].to_numpy()
    weight = persons["weight"].to_numpy()
    q80 = _weighted_quantile(exposure[matched], weight[matched], 0.8)
    age = persons["age"].to_numpy()
    mask = matched & (age >= 22) & (age <= 25) & (exposure >= q80)
    return _rate_within(persons, mask, 0.16)


def run(name: str, displaced: np.ndarray, dataset, baseline_sim, persons):
    from policyengine_uk import Microsimulation

    employment = persons["employment_income"].to_numpy(dtype=float)
    shocked_emp = np.where(displaced, 0.0, employment)

    sim = Microsimulation(dataset=dataset)
    sim.set_input("employment_income", PERIOD, shocked_emp)

    def metrics(s):
        pw = s.calculate("person_weight", period=PERIOD, map_to="person").values
        hw = s.calculate("household_weight", period=PERIOD, map_to="household").values
        eq = s.calculate("equiv_household_net_income", period=PERIOD, map_to="household").values
        npeople = s.calculate("household_count_people", period=PERIOD, map_to="household").values
        return {
            "gov_balance": float(
                (s.calculate("gov_balance", period=PERIOD, map_to="household").values * hw).sum()
            ),
            "poverty_bhc": float(np.average(
                s.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values, weights=pw)),
            "gini": gini(eq, hw * npeople),
            "hni": s.calculate("household_net_income", period=PERIOD, map_to="person").values,
        }

    base, shock = metrics(baseline_sim), metrics(sim)

    weight = persons["weight"].to_numpy()
    age = persons["age"].to_numpy()
    employed = employment > 0
    delta = shock["hni"] - base["hni"]
    bands = {}
    for lo, hi in AGE_BANDS:
        m = (age >= lo) & (age <= hi)
        emp_w = float(weight[m & employed].sum())
        bands[f"{lo}-{hi if hi < 200 else '+'}"] = {
            "displaced_weighted": float(weight[m & displaced].sum()),
            "displacement_rate_of_employed": float(weight[m & displaced].sum() / emp_w) if emp_w else 0.0,
            "mean_income_change": float(np.average(delta[m], weights=weight[m])) if m.any() else 0.0,
        }

    result = {
        "scenario": name,
        "displaced_weighted_total": float(weight[displaced].sum()),
        "share_of_employed_displaced": float(weight[displaced].sum() / weight[employed].sum()),
        "exchequer_cost_bn": (base["gov_balance"] - shock["gov_balance"]) / 1e9,
        "poverty_change_bhc_pp": 100 * (shock["poverty_bhc"] - base["poverty_bhc"]),
        "gini_change_pp": 100 * (shock["gini"] - base["gini"]),
        "age_bands": bands,
    }
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    baseline = Microsimulation(dataset=dataset)

    def calc(v):
        return baseline.calculate(v, period=PERIOD, map_to="person").values

    persons = pd.DataFrame(
        {
            "person_id": calc("person_id"),
            "age": calc("age"),
            "employment_income": calc("employment_income"),
            "weight": calc("person_weight"),
        }
    )
    persons["soc_major_group"] = attach_soc_major_group(
        persons["person_id"], DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
    )
    exposure = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    theta = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(exposure), exposure, np.nanmean(exposure))
    persons["complementarity"] = np.where(np.isfinite(theta), theta, np.nanmean(theta))

    for name, fn in [
        ("klein_teeselink_2025", displaced_klein_teeselink),
        ("hosseini_lichtinger_2026", displaced_hosseini_lichtinger),
        (
            "hosseini_lichtinger_2026_adopter_scaled",
            lambda p: displaced_hosseini_lichtinger(p, scale=HL_ADOPTER_EMPLOYMENT_SHARE),
        ),
        ("brynjolfsson_canaries_2025", displaced_brynjolfsson),
    ]:
        r = run(name, fn(persons), dataset, baseline, persons)
        print(
            f"{name}: displaced {r['displaced_weighted_total']/1e6:.2f}m "
            f"({100*r['share_of_employed_displaced']:.1f}% of employees), "
            f"exchequer £{r['exchequer_cost_bn']:.1f}bn, "
            f"poverty {r['poverty_change_bhc_pp']:+.2f}pp, "
            f"gini {r['gini_change_pp']:+.2f}pp"
        )


if __name__ == "__main__":
    main()
