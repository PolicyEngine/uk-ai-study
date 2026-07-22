"""Incidence as a first-class scenario axis (issue #1, Findings 0, R2-3).

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
                  above-median-exposure occupations (multiplier
                  ``compression_multiplier``, default 2). An author-designed
                  stress test, not a calibrated implementation of Autor
                  (2024).
  uniform       — equal displacement exposure for everyone (exposure = 1 in
                  the eq 3.4 draw)

Factorial design (round-2 finding R2-3): the DEFAULT families vary ONLY the
displacement mask. The person-level survivor wage-uplift channel (eq 3.5
with the observed complementarity theta) and the capital channel are
IDENTICAL across families, so the headline spread is attributable to who
loses work. The earlier compound behaviour — where "uniform" also flattened
the wage uplift (theta = 1) and "compression" also inverted it (mid-skill
gains) — lives on a separate, explicitly named axis: pass
``wage_axis="family"`` or use the family names ``uniform_compound`` /
``compression_compound``.

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
    build_shocked_simulation,
    draw_displaced,
)

DATA = Path("data")
OUT = Path("results/incidence")
PERIOD = 2026
SEED = 0
#: Default draw tilt toward the elite (top-earnings x high-exposure) group in
#: the compression family. Parameterised for sensitivity runs (R2-6a):
#: 1.5 / 2.0 / 3.0.
COMPRESSION_TOP_TERTILE_MULTIPLIER = 2.0

DEFAULT_FAMILIES = ("exposure", "junior", "compression", "uniform")


def _weighted_tertile_threshold(values, weights, q):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return values[order][np.searchsorted(cw, q * cw[-1])]


def compression_mask(
    persons: pd.DataFrame,
    scenario,
    seed: int = SEED,
    multiplier: float = COMPRESSION_TOP_TERTILE_MULTIPLIER,
) -> np.ndarray:
    """Boolean displaced mask for the expertise-compression family.

    Draws are tilted toward the elite group — top weighted-earnings tertile
    within above-median-exposure occupations — by ``multiplier`` (>= 1),
    with the eq 3.4 quota filled at the all-employee level.
    """
    w = persons["weight"].to_numpy()
    earnings = persons["employment_income"].to_numpy(dtype=float)
    exposure = persons["exposure"].to_numpy()
    employed = earnings > 0

    med_exp = _weighted_tertile_threshold(exposure[employed], w[employed], 0.5)
    top_earn = _weighted_tertile_threshold(earnings[employed], w[employed], 2 / 3)
    elite = employed & (exposure > med_exp) & (earnings >= top_earn)

    rng = np.random.default_rng(seed)
    quota = scenario.displacement_rate * float(w[employed].sum())
    members = np.flatnonzero(employed)
    p = np.where(elite[members], float(multiplier), 1.0)
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


def displacement_mask_for(
    family: str,
    persons: pd.DataFrame,
    seed: int = SEED,
    compression_multiplier: float = COMPRESSION_TOP_TERTILE_MULTIPLIER,
) -> np.ndarray:
    """Family-specific displaced mask; the ONLY axis the default families vary.

    Note: draw_displaced uses systematic sampling with prescribed inclusion
    probabilities — draws are NOT prefix-nested across rates or families;
    each family's mask is an independent realisation for the given seed.
    """
    scenario = PRESETS["central"]
    if family == "exposure":
        return draw_displaced(persons, scenario, seed=seed)
    if family == "junior":
        return draw_displaced(persons, PRESETS["central_youth_tilted"], seed=seed)
    if family == "uniform":
        flat = persons.copy()
        flat["exposure"] = 1.0
        return draw_displaced(flat, scenario, seed=seed)
    if family == "compression":
        return compression_mask(
            persons, scenario, seed=seed, multiplier=compression_multiplier
        )
    raise ValueError(family)


def apply_channels(
    persons: pd.DataFrame,
    displaced: np.ndarray,
    scenario,
    theta: np.ndarray | None = None,
) -> pd.DataFrame:
    """Wage (eq 3.5) and capital channels for a GIVEN displaced mask.

    ``theta`` defaults to the observed person-level complementarity — the
    central wage channel shared by every default family. Passing an explicit
    theta (flat or inverted) produces the compound variants.
    """
    shocked = persons.copy()
    shocked["displaced"] = displaced
    earnings = persons["employment_income"].to_numpy(dtype=float)
    w = persons["weight"].to_numpy(dtype=float)
    employed = earnings > 0
    survivors = employed & ~displaced

    if theta is None:
        theta = persons["complementarity"].to_numpy(dtype=float)
    theta = np.asarray(theta, dtype=float)
    # employment-weighted normalisation over baseline workers, as eq 3.5
    theta_bar = float((theta * w)[employed].sum() / w[employed].sum())
    uplift = np.zeros_like(earnings)
    if theta_bar > 0:
        uplift[survivors] = (
            scenario.wage_uplift * (theta[survivors] / theta_bar) * earnings[survivors]
        )
    shocked["employment_income"] = np.where(displaced, 0.0, earnings + uplift)

    factor = (
        BASELINE_CAPITAL_RETURN + scenario.capital_return_increase
    ) / BASELINE_CAPITAL_RETURN
    for col in ("savings_interest_income", "dividend_income"):
        shocked[col] = shocked[col].to_numpy(dtype=float) * factor
    return shocked


def shocked_table_for(
    family: str,
    persons: pd.DataFrame,
    seed: int = SEED,
    wage_axis: str = "central",
    compression_multiplier: float = COMPRESSION_TOP_TERTILE_MULTIPLIER,
) -> pd.DataFrame:
    """Shocked person table for an incidence family.

    ``wage_axis="central"`` (default, R2-3 factorial design): every family
    shares the identical eq 3.5 wage channel (observed theta) and capital
    channel; only the displacement mask differs.

    ``wage_axis="family"`` restores the compound bundles: "uniform" also
    flattens the wage uplift (theta = 1) and "compression" also inverts it
    (mid-skill gains). Equivalent family-name aliases: "uniform_compound",
    "compression_compound". "exposure" and "junior" are identical under
    either axis.

    ``compression_multiplier`` (R2-6a): the elite draw tilt in the
    compression family, default 2.0; 1.5 / 3.0 for sensitivity runs.
    """
    if family.endswith("_compound"):
        family = family[: -len("_compound")]
        wage_axis = "family"
    if wage_axis not in ("central", "family"):
        raise ValueError(f"unknown wage_axis: {wage_axis!r}")

    scenario = PRESETS["central"]
    displaced = displacement_mask_for(
        family, persons, seed=seed, compression_multiplier=compression_multiplier
    )

    theta = None
    if wage_axis == "family":
        if family == "uniform":
            theta = np.ones(len(persons))
        elif family == "compression":
            th = persons["complementarity"].to_numpy(dtype=float)
            employed = persons["employment_income"].to_numpy(dtype=float) > 0
            # inverse complementarity: theta' = max + min - theta flips the
            # gradient (mid-skill gains), same normalisation as eq 3.5
            theta = th[employed].max() + th[employed].min() - th
    return apply_channels(persons, displaced, scenario, theta=theta)


def compression_table(
    persons: pd.DataFrame,
    scenario,
    seed: int = SEED,
    multiplier: float = COMPRESSION_TOP_TERTILE_MULTIPLIER,
) -> pd.DataFrame:
    """Back-compat wrapper: the COMPOUND compression bundle (tilted draw +
    inverted wage uplift). Prefer shocked_table_for("compression", ...) for
    the factorial variant."""
    displaced = compression_mask(persons, scenario, seed=seed, multiplier=multiplier)
    th = persons["complementarity"].to_numpy(dtype=float)
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    theta = th[employed].max() + th[employed].min() - th
    return apply_channels(persons, displaced, scenario, theta=theta)


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
            # household-level %, identical for all members; broadcast
            # intentional — used only in the delta/base ratio below (issue #6)
            "hni": s.calculate("hbai_household_net_income", period=PERIOD, map_to="person").values,
        }

    b = metrics(baseline)
    summary = []
    # default (factorial) families plus the compound wage-axis variants,
    # reported on their own clearly named axis
    for family in DEFAULT_FAMILIES + ("compression_compound", "uniform_compound"):
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
