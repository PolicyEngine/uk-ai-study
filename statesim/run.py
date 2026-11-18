"""Household state runs for every setting of the round-4 revision.

A port of prototypes/phase2_states.py (validated: options off reproduce the
Phase 1 exact central case; 12-month mixing matches the annual run). Writes one
compressed file per (state, take-up mode, run type) under OUT_ROOT/<setting>/,
skipping files that already exist, so a killed run resumes where it stopped.

Usage: python -m statesim.run SETTING [SETTING ...]   (or "all")
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from statesim import engine  # noqa: E402

OUT_ROOT = Path.home() / "uk-ai-study-rescue" / "state-arrays"
HH_VARS = ("gov_balance", "household_benefits", "hbai_household_net_income",
           "equiv_hbai_household_net_income", "equiv_hbai_household_net_income_ahc",
           "hbai_household_net_income_ahc", "poverty_threshold_bhc", "poverty_threshold_ahc",
           "universal_credit", "household_market_income")
PERSON_TAX = ("income_tax", "national_insurance", "ni_employer")

_CENTRAL = {"universe": "private", "uplift_universe": "private", "modes": ["flag", "claim"],
            "penalty": engine.REEMPLOYMENT_PENALTY, "wage_uplift": 0.026, "jsa": True,
            "interest_factor": 1.0}
#: settings, in run order (docs/referee/phase2-stage-inventory.md)
SETTINGS = {
    "central_f100": {**_CENTRAL, "K": 4, "spells": [3, 6, 9, 12], "dividend_factor": 1.0,
                     "run_types": ["out_jsa", "out", "re", "tax"]},
    "central_f115": {**_CENTRAL, "K": 4, "spells": [3, 6, 9, 12], "dividend_factor": 1.15,
                     "run_types": ["out_jsa", "out", "re", "tax"]},
    "nochannels": {**_CENTRAL, "K": 4, "spells": [12], "wage_uplift": 0.0, "dividend_factor": 1.0,
                   "run_types": ["tax"]},
    "opencap": {**_CENTRAL, "K": 4, "spells": [12], "dividend_factor": 1.0 + 0.004 / 0.01005,
                "interest_factor": 1.0 + 0.004 / 0.01005, "run_types": ["tax"]},
    **{f"grid_u{u}": {**_CENTRAL, "K": 4, "spells": [12], "wage_uplift": u / 100, "dividend_factor": 1.0,
                      "run_types": ["tax"]} for u in range(6)},
}


def outcomes(sim, hh_index, n_hh, period):
    c = lambda v, m: np.asarray(sim.calculate(v, period=period, map_to=m).values, float)  # noqa: E731
    out = {v: c(v, "household") for v in HH_VARS}
    for v in PERSON_TAX:
        out[v] = np.bincount(hh_index, weights=c(v, "person"), minlength=n_hh)
    return out


def run_setting(name: str, cfg: dict) -> None:  # pragma: no cover - needs Microcosm
    import displaced_takeup_runs as tk
    from uk_ai_study import shocks
    from uk_ai_study.data import PERIOD, microsimulation
    from uk_ai_study.runner import build_person_table

    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    K, modes, spells, penalty = cfg["K"], cfg["modes"], cfg["spells"], cfg["penalty"]
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
              for v in list(shocks.TRANSITION_ZEROED_VARIABLES) + ["employment_status"]}
    employed, weight, _, group = shocks._employee_groups(persons)
    eligible = employed & (sector != "PUBLIC") if cfg["universe"] == "private" else employed
    rank = engine.within_household_rank(hh_index, eligible)
    params = __import__("policyengine_uk").CountryTaxBenefitSystem().parameters(f"{PERIOD}-01-01")
    jp = params.gov.dwp.JSA.contrib
    lel = float(params.gov.hmrc.national_insurance.class_1.thresholds.lower_earnings_limit)
    age = persons["age"].to_numpy(float)
    earnings = persons["employment_income"].to_numpy(float)
    award52 = tk.jsa_award(age, tk.jsa_eligible(age, earnings, eligible, lel),
                           float(jp.amount_over_25), float(jp.amount_18_24), weeks=52)
    if not (out / "baseline_flag.npz").exists():
        np.savez_compressed(out / "baseline_flag.npz", hh_w=c("household_weight", "household"),
                            hh_n=c("household_count_people", "household"),
                            **outcomes(base_sim, hh_index, n_hh, PERIOD))
    if "claim" in modes and not (out / "baseline_claim.npz").exists():
        b2 = microsimulation()
        tk._set_checked(b2, "would_claim_uc", base_flags | np.isin(bu_id, np.unique(person_bu[eligible])))
        np.savez_compressed(out / "baseline_claim.npz", **outcomes(b2, hh_index, n_hh, PERIOD))
        del b2
        gc.collect()
    np.save(out / "exposure.npy", persons["exposure"].to_numpy(float))
    np.save(out / "employed.npy", employed)
    np.savez_compressed(out / "persons.npz", hh_index=hh_index, rank=rank, eligible=eligible,
                        weight=weight, group=group, age=age, sector=sector, earnings=earnings,
                        complementarity=persons["complementarity"].to_numpy(float),
                        dividends=persons["dividend_income"].to_numpy(float),
                        employer_ni=np.asarray(c("ni_employer", "person"), float),
                        employer_pension=np.asarray(c("employer_pension_contributions", "person"), float),
                        person_bu=person_bu, bu_id=bu_id, base_flags=base_flags)

    scen = shocks.ShockScenario(name, shocks.CENTRAL_DISPLACEMENT_RATE, cfg["wage_uplift"],
                                capital_return_increase=0.0)
    th = persons["complementarity"].to_numpy(float)
    ref = eligible if cfg.get("uplift_universe") == "private" else employed
    th_bar = float((th * weight)[ref].sum() / weight[ref].sum())
    upl = cfg["wage_uplift"] * th / th_bar * earnings
    meta = {"seconds": {}}
    original = shocks.draw_displaced
    for b in range(2 ** K):
        mask = engine.state_mask(rank, b, K)
        shocks.draw_displaced = lambda p_, s_, seed=0, m=mask: m
        try:
            table = shocks.apply_shocks(persons, scen, seed=0)
        finally:
            shocks.draw_displaced = original
        if cfg.get("uplift_universe") == "private":
            pay = np.where(ref & ~mask, earnings + upl, np.where(mask, 0.0, earnings))
            table["employment_income"] = np.where(employed, pay, table["employment_income"].to_numpy(float))
        table["dividend_income"] = persons["dividend_income"].to_numpy(float) * cfg["dividend_factor"]
        table["savings_interest_income"] = persons["savings_interest_income"].to_numpy(float) * cfg["interest_factor"]
        surv_pay = table["employment_income"].to_numpy(float)
        d_bu = np.isin(bu_id, np.unique(person_bu[mask]))
        variants = []
        for rt in cfg["run_types"]:
            variants += [(f"tax{s}", s) for s in spells] if rt == "tax" else [(rt, None)]
        for m in modes:
            for vname, s in variants:
                key = f"s{b:02d}_{m}_{vname}"
                if (out / f"{key}.npz").exists():
                    continue
                t = time.time()
                tab = table.copy()
                if vname == "re":
                    pay, work = np.where(mask, (1 - penalty) * earnings, surv_pay), 1.0
                elif vname.startswith("tax"):
                    pay, work = np.where(mask, (12 - s) / 12 * (1 - penalty) * earnings, surv_pay), (12 - s) / 12
                else:
                    pay, work = np.where(mask, 0.0, surv_pay), None
                tab["employment_income"] = pay
                if work is not None and work > 0:
                    tab["displaced"] = False
                sim = shocks.build_shocked_simulation(None, tab, PERIOD, base_arrays=arrays)
                if work is not None and 0 < work < 1:
                    for v in shocks.TRANSITION_ZEROED_VARIABLES:
                        vals = np.asarray(arrays[v], float).copy()
                        vals[mask] *= work
                        sim.set_input(v, PERIOD, vals)
                if m == "claim":
                    tk._set_checked(sim, "would_claim_uc", base_flags | d_bu)
                jsa_months = 12 if vname == "out_jsa" else (min(s, 6) if vname.startswith("tax") else 0)
                if cfg["jsa"] and jsa_months:
                    sim.set_input("jsa_contrib_reported", PERIOD,
                                  np.where(mask, award52 * jsa_months / 12, base_jsa_rep))
                np.savez_compressed(out / f"{key}.npz", **outcomes(sim, hh_index, n_hh, PERIOD))
                del sim, tab
                gc.collect()
                meta["seconds"][key] = round(time.time() - t, 1)
                print(f"[{name}] {key} {meta['seconds'][key]}s", flush=True)
    meta["total_seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "DONE").write_text(time.strftime("%F %T"))


def main(argv: list[str]) -> None:  # pragma: no cover
    names = list(SETTINGS) if argv == ["all"] else argv
    for n in names:
        if (OUT_ROOT / n / "DONE").exists():
            print(f"[{n}] already done", flush=True)
            continue
        run_setting(n, SETTINGS[n])


if __name__ == "__main__":
    main(sys.argv[1:])
