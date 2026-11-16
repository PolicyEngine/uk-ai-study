"""Phase 2 prototype: household state runs for the new central case.

docs/referee/phase2-design.md and phase2-calibration.md. Outside analysis/ and
uk_ai_study/ so the attested build is untouched.

For each displacement state b of a household's (eligible) employees and each
take-up mode m in {flag, claim}, seven shocked runs:

  out_jsa  displaced earn 0, JSA at the 52-week rate (monthly value x 12)
  out      displaced earn 0, no JSA
  re       displaced re-employed all year at (1 - penalty) x pay
  tax_s    s in {3, 6, 9, 12}: displaced earn (12-s)/12 x (1-penalty) x pay,
           JSA received for min(s, 6) months; read taxes and HBAI income

and two baselines (survey flags; every employee benefit unit flagged). The
monthly benefit mix for a spell of s months is
  (min(s,6)/12) B(out_jsa) + ((s - min(s,6))/12) B(out) + ((12-s)/12) B(re)
and replaces the annual benefits of the tax_s run in both the government
balance and HBAI income. Survivor uplift and capital follow the scenario.

Usage: python prototypes/phase2_states.py OUT_DIR CONFIG_JSON
CONFIG keys: K, universe ("all"|"private"), modes (["flag","claim"]),
  spells ([3,6,9,12]), penalty (0.1), wage_uplift (0.026),
  dividend_factor, interest_factor (capital multipliers on baseline income),
  jsa (bool), run_types (subset of out_jsa,out,re,tax).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from uk_ai_study import shocks  # noqa: E402
from uk_ai_study.data import PERIOD, microsimulation  # noqa: E402
from uk_ai_study.runner import build_person_table  # noqa: E402

HH_VARS = ("gov_balance", "household_benefits", "hbai_household_net_income",
           "equiv_hbai_household_net_income", "equiv_hbai_household_net_income_ahc",
           "hbai_household_net_income_ahc", "poverty_threshold_bhc", "poverty_threshold_ahc")
PERSON_TAX = ("income_tax", "national_insurance", "ni_employer")


def outcomes(sim, hh_index, n_hh):
    c = lambda v, m: np.asarray(sim.calculate(v, period=PERIOD, map_to=m).values, float)  # noqa: E731
    out = {v: c(v, "household") for v in HH_VARS}
    for v in PERSON_TAX:
        out[v] = np.bincount(hh_index, weights=c(v, "person"), minlength=n_hh)
    return out


def main(out_dir: str, cfg: dict) -> None:
    import displaced_takeup_runs as tk

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    K = cfg.get("K", 3)
    modes = cfg.get("modes", ["flag", "claim"])
    spells = cfg.get("spells", [3, 6, 9, 12])
    penalty = cfg.get("penalty", 0.1)
    run_types = cfg.get("run_types", ["out_jsa", "out", "re", "tax"])
    t0 = time.time()

    base_sim = microsimulation()
    persons = build_person_table(base_sim, PERIOD)
    c = lambda v, m: base_sim.calculate(v, period=PERIOD, map_to=m).values  # noqa: E731
    hh_ids, p_hh = c("household_id", "household"), c("person_household_id", "person")
    pos = {h: i for i, h in enumerate(hh_ids)}
    hh_index = np.array([pos[h] for h in p_hh])
    n_hh = len(hh_ids)
    person_bu, bu_id = c("person_benunit_id", "person"), c("benunit_id", "benunit")
    base_flags = np.asarray(c("would_claim_uc", "benunit"), bool)
    base_jsa_rep = np.asarray(c("jsa_contrib_reported", "person"), float)
    sector = np.asarray(c("employment_sector", "person")).astype(str)
    arrays = {v: base_sim.calculate(v, period=PERIOD, map_to="person").values
              for v in list(shocks.TRANSITION_ZEROED_VARIABLES) + ["employment_status"]} \
        if hasattr(shocks, "TRANSITION_ZEROED_VARIABLES") else None

    employed, weight, _, group = shocks._employee_groups(persons)
    eligible = employed & ((sector != "PUBLIC") if cfg.get("universe", "all") == "private" else True)
    rank = np.full(len(persons), -1)
    seen: dict = {}
    for i in np.flatnonzero(eligible):
        h = hh_index[i]
        rank[i] = seen.get(h, 0)
        seen[h] = rank[i] + 1

    system = base_sim.tax_benefit_system if hasattr(base_sim, "tax_benefit_system") else None
    params = (system or __import__("policyengine_uk").CountryTaxBenefitSystem()).parameters(f"{PERIOD}-01-01")
    jp = params.gov.dwp.JSA.contrib
    lel = float(params.gov.hmrc.national_insurance.class_1.thresholds.lower_earnings_limit)
    age = persons["age"].to_numpy(float)
    earnings = persons["employment_income"].to_numpy(float)
    award52 = tk.jsa_award(age, tk.jsa_eligible(age, earnings, eligible, lel),
                           float(jp.amount_over_25), float(jp.amount_18_24), weeks=52)

    base = outcomes(base_sim, hh_index, n_hh)
    np.savez_compressed(out / "baseline_flag.npz", hh_w=c("household_weight", "household"),
                        hh_n=c("household_count_people", "household"), **base)
    any_emp_bu = np.isin(bu_id, np.unique(person_bu[eligible]))
    if "claim" in modes:
        b2 = microsimulation()
        tk._set_checked(b2, "would_claim_uc", base_flags | any_emp_bu)
        np.savez_compressed(out / "baseline_claim.npz", **outcomes(b2, hh_index, n_hh))
        del b2
        gc.collect()
    np.save(out / "exposure.npy", persons["exposure"].to_numpy(float))
    np.save(out / "employed.npy", employed)
    np.savez_compressed(out / "persons.npz", hh_index=hh_index, rank=rank, eligible=eligible,
                        weight=weight, group=group, age=age, sector=sector,
                        person_bu=person_bu, bu_id=bu_id, base_flags=base_flags)

    scen = shocks.ShockScenario("phase2", shocks.CENTRAL_DISPLACEMENT_RATE, cfg.get("wage_uplift", 0.026),
                                capital_return_increase=0.0)
    div_f, int_f = cfg.get("dividend_factor", 1.0), cfg.get("interest_factor", 1.0)
    original = shocks.draw_displaced
    meta = {"seconds": {}}
    for b in range(2 ** K):
        mask = eligible & (rank >= 0) & (rank < K) & (((b >> np.clip(rank, 0, None)) & 1) == 1)
        shocks.draw_displaced = lambda p_, s_, seed=0, m=mask: m
        try:
            table = shocks.apply_shocks(persons, scen, seed=0)
        finally:
            shocks.draw_displaced = original
        table["dividend_income"] = persons["dividend_income"].to_numpy(float) * div_f
        table["savings_interest_income"] = persons["savings_interest_income"].to_numpy(float) * int_f
        surv_pay = table["employment_income"].to_numpy(float)
        d_bu = np.isin(bu_id, np.unique(person_bu[mask]))
        variants = []
        for rt in run_types:
            if rt == "tax":
                variants += [(f"tax{s}", s) for s in spells]
            else:
                variants.append((rt, None))
        for m in modes:
            flags = base_flags | d_bu if m == "claim" else base_flags
            for name, s in variants:
                key = f"s{b:02d}_{m}_{name}"
                if (out / f"{key}.npz").exists():
                    continue
                t = time.time()
                tab = table.copy()
                pay = np.where(mask, earnings, surv_pay)
                if name == "re":
                    pay = np.where(mask, (1 - penalty) * earnings, surv_pay)
                elif name.startswith("tax"):
                    pay = np.where(mask, (12 - s) / 12 * (1 - penalty) * earnings, surv_pay)
                elif name in ("out_jsa", "out"):
                    pay = np.where(mask, 0.0, surv_pay)
                tab["employment_income"] = pay
                work_share = None
                if name == "re":
                    work_share = 1.0
                elif name.startswith("tax"):
                    work_share = (12 - s) / 12
                if work_share is not None:
                    tab["displaced"] = False  # in work for part or all of the year
                sim = shocks.build_shocked_simulation(None, tab, PERIOD, base_arrays=arrays)
                if work_share is not None and work_share < 1.0:
                    # employment-linked inputs scale with the share of the year worked
                    for v in shocks.TRANSITION_ZEROED_VARIABLES:
                        vals = np.asarray(arrays[v], float).copy()
                        vals[mask] = vals[mask] * work_share
                        sim.set_input(v, PERIOD, vals)
                if m == "claim":
                    tk._set_checked(sim, "would_claim_uc", flags)
                jsa_months = 12 if name == "out_jsa" else (min(s, 6) if name.startswith("tax") else 0)
                if cfg.get("jsa", True) and jsa_months:
                    sim.set_input("jsa_contrib_reported", PERIOD,
                                  np.where(mask, award52 * jsa_months / 12, base_jsa_rep))
                np.savez_compressed(out / f"{key}.npz", **outcomes(sim, hh_index, n_hh))
                del sim, tab
                gc.collect()
                meta["seconds"][key] = round(time.time() - t, 1)
                print(f"{key} {meta['seconds'][key]}s", flush=True)
    meta["total_seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()))
