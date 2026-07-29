"""Decompose the central scenario's BHC absolute-poverty change (seed 0).

The paper's headline +1.81pp poverty change is a NET figure. For referees,
this script splits it into a gross inflow (persons not in_poverty_bhc at
baseline who are in poverty after the shock) and a gross outflow (persons
leaving poverty, e.g. via the capital return or eq 3.5 wage uplift), and
characterises the newly poor:

1. weighted count of newly-poor persons;
2. their weighted distribution across baseline equivalised HBAI household
   net income deciles (person-weighted deciles, HBAI convention);
3. share living in a household containing a displaced worker;
4. median baseline ratio of household income to the household's BHC poverty
   line (how close they started to the line);
5. median post-shock shortfall below the line, as % of the line;
6. share in displaced-worker households where the displaced worker(s) were
   NOT the household's only baseline earner(s).

Outputs results/robustness/poverty_decomposition.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uk_ai_study.runner import build_person_table
from uk_ai_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

DATA = Path("data")
OUT = Path("results/robustness")
PERIOD = 2026
SEED = 0
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, 0.5 * cw[-1])])


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    baseline = Microsimulation(dataset=ds)
    persons = build_person_table(baseline, PERIOD, ADULT)

    table = apply_shocks(persons, PRESETS["central"], seed=SEED)
    shocked = build_shocked_simulation(ds, baseline, table, PERIOD)

    pcalc = lambda s, v: s.calculate(v, period=PERIOD, map_to="person").values
    pw = pcalc(baseline, "person_weight")
    hh_id = pcalc(baseline, "household_id")

    pov_b = pcalc(baseline, "in_poverty_bhc").astype(bool)
    pov_s = pcalc(shocked, "in_poverty_bhc").astype(bool)
    # household income relative to the household-specific BHC poverty line
    line = pcalc(baseline, "poverty_line_bhc")
    ratio_b = pcalc(baseline, "hbai_household_net_income") / line
    ratio_s = pcalc(shocked, "hbai_household_net_income") / line
    eq_b = pcalc(baseline, "equiv_hbai_household_net_income")

    inflow = ~pov_b & pov_s
    outflow = pov_b & ~pov_s
    total_w = float(pw.sum())
    inflow_w = float(pw[inflow].sum())
    outflow_w = float(pw[outflow].sum())
    net_pp = 100 * (inflow_w - outflow_w) / total_w
    headline_pp = 100 * (float((pov_s * pw).sum()) - float((pov_b * pw).sum())) / total_w
    assert np.isclose(net_pp, headline_pp)

    # baseline equivalised HBAI income deciles (person-weighted, HBAI style)
    order = np.argsort(eq_b)
    cw = np.cumsum(pw[order])
    edges = [eq_b[order][np.searchsorted(cw, q * cw[-1])] for q in np.arange(0.1, 1.0, 0.1)]
    decile = 1 + np.searchsorted(edges, eq_b, side="right")
    decile_shares = {
        int(d): float(pw[inflow & (decile == d)].sum() / inflow_w) for d in range(1, 11)
    }

    # household displaced-worker / other-earner composition
    displaced = table["displaced"].to_numpy()
    employed_b = persons["employment_income"].to_numpy(dtype=float) > 0
    hh_displaced = np.isin(hh_id, np.unique(hh_id[displaced]))
    hh_other_earner = np.isin(hh_id, np.unique(hh_id[employed_b & ~displaced]))

    in_disp_hh = inflow & hh_displaced
    in_disp_hh_w = float(pw[in_disp_hh].sum())
    result = {
        "scenario": "central",
        "seed": SEED,
        "period": PERIOD,
        "poverty_rate_bhc_baseline": float((pov_b * pw).sum() / total_w),
        "poverty_rate_bhc_shocked": float((pov_s * pw).sum() / total_w),
        "net_poverty_change_pp": net_pp,
        "gross_inflow_persons": inflow_w,
        "gross_outflow_persons": outflow_w,
        "gross_inflow_pp": 100 * inflow_w / total_w,
        "gross_outflow_pp": 100 * outflow_w / total_w,
        "newly_poor_decile_shares_baseline_equiv_income": decile_shares,
        "newly_poor_share_in_displaced_worker_household": in_disp_hh_w / inflow_w,
        "newly_poor_median_baseline_income_to_line_ratio": weighted_median(
            ratio_b[inflow], pw[inflow]
        ),
        "newly_poor_median_postshock_shortfall_pct_of_line": 100 * weighted_median(
            (1.0 - ratio_s)[inflow], pw[inflow]
        ),
        "newly_poor_share_displaced_hh_with_other_earner": float(
            pw[in_disp_hh & hh_other_earner].sum() / in_disp_hh_w
        ),
    }
    (OUT / "poverty_decomposition.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
