#!/usr/bin/env python3
"""UC take-up and new-style JSA for the displaced: runs reported beside the
central assumption (specified 13 Nov 2026; not yet run).

The central assumption is unchanged: every benefit unit keeps the UC take-up
flag (``would_claim_uc``) drawn for it in the baseline data, and newly
displaced workers receive no new-style (contributory) Jobseeker's Allowance,
because the model pays it only where the survey reports an award. Two
variants are reported beside it:

``takeup1``
    ``would_claim_uc`` is True for every benefit unit that contains a
    displaced person, in BOTH a paired baseline and the shocked simulation,
    so the change measures the shock and not the flag change.
``jsa``
    ``takeup1`` plus ``jsa_contrib_reported`` = 26 x the model's new-style
    JSA weekly rate (the 25-and-over or the 18-24 rate) for displaced people
    aged 18-65 whose baseline annual earnings are at least 50 x the weekly
    lower earnings limit (the class 1 contribution condition, approximated
    on one year's earnings). It shares the ``takeup1`` paired baseline.
    UC counts the JSA as unearned income; whether the model taxes it is
    read from the tax-benefit system and recorded in the summary JSON.

Scenarios, each paired with an existing run seed by seed (the displaced set
is asserted identical through ``displaced_weighted_m``):

``disp_only``  7 per cent exposure-proportional displacement, no survivor
               uplift and no capital shock, seed 0: the run behind the
               job-loss absorption rates (tax_composition.py, u = 7).
``obr``        run 15's sizing (results/next_runs/obr_benchmark.json),
               seeds 0-19, paired with obr_benchmark_draws.csv.
``central``    the central preset, seeds 0-19, paired with the ``central``
               rows of results/robustness/dividend_coverage_draws.csv.

Each (scenario, seed) runs three simulations: the paired baseline with the
take-up flags, the ``takeup1`` shock and the ``jsa`` shock, so the design is
3 + 60 + 60 = 123 simulations plus one unmodified baseline at start-up
(deciles, flags and the existing-unemployed reference).

Outputs (results/takeup/):
    takeup_draws.csv    one row per (scenario, variant, seed), stamped with
                        this script's own code fingerprint; continued only
                        with --resume
    takeup_deciles.csv  household flows by baseline decile per draw
    takeup_runs.json    summaries, JSA facts and the take-up diagnostics

Usage:
    python analysis/displaced_takeup_runs.py [--resume] [--only SCENARIO]

The shock mechanics (build_person_table, apply_shocks,
build_shocked_simulation) are imported unchanged; the pure helpers below
are unit-tested in tests/test_displaced_takeup_runs.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "takeup"
DRAWS = OUT / "takeup_draws.csv"
DECILES = OUT / "takeup_deciles.csv"
SUMMARY = OUT / "takeup_runs.json"
PERIOD = 2026
SEEDS = tuple(range(20))
VARIANTS = ("takeup1", "jsa")
#: execution order: the three-simulation absorption case first
SCENARIOS = ("disp_only", "obr", "central")
SCENARIO_SEEDS = {"disp_only": (0,), "obr": SEEDS, "central": SEEDS}
DISP_ONLY_RATE = 0.07
JSA_WEEKS = 26
JSA_MIN_AGE, JSA_MAX_AGE = 18, 66  # eligible if 18 <= age < 66
LEL_WEEKS = 50
CHILDCARE_VARS = ("tax_free_childcare", "extended_childcare_entitlement", "universal_childcare_entitlement",
                  "targeted_childcare_entitlement", "wales_childcare_offer")
STATUTORY_PAY_VARS = ("statutory_sick_pay", "statutory_maternity_pay", "statutory_paternity_pay")
OWNER_TENURES = ("OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE")
EXPECTED_ROWS = sum(len(s) for s in SCENARIO_SEEDS.values()) * len(VARIANTS)  # 82


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def displaced_benunits(person_benunit_id, benunit_id, displaced) -> np.ndarray:
    """Boolean over benefit units: the unit contains a displaced person."""
    hit = np.unique(np.asarray(person_benunit_id)[np.asarray(displaced, bool)])
    return np.isin(np.asarray(benunit_id), hit)


def takeup_flags(base_flags, displaced_bu) -> np.ndarray:
    """Baseline take-up flags with every displaced benefit unit claiming."""
    return np.asarray(base_flags, bool) | np.asarray(displaced_bu, bool)


def jsa_eligible(age, baseline_earnings, displaced, lel_week: float) -> np.ndarray:
    """Displaced people aged 18-65 whose baseline earnings meet 50 x LEL."""
    age = np.asarray(age, float)
    return (np.asarray(displaced, bool) & (age >= JSA_MIN_AGE) & (age < JSA_MAX_AGE)
            & (np.asarray(baseline_earnings, float) >= LEL_WEEKS * lel_week))


def jsa_award(age, eligible, rate_25: float, rate_18_24: float, weeks: int = JSA_WEEKS) -> np.ndarray:
    """Annual new-style JSA of each eligible person (pounds)."""
    rate = np.where(np.asarray(age, float) >= 25, rate_25, rate_18_24)
    return np.where(np.asarray(eligible, bool), weeks * rate, 0.0)


def jsa_reported_input(base_reported, award, eligible) -> np.ndarray:
    """``jsa_contrib_reported`` for the jsa variant: the award for eligible
    displaced people, the baseline value for everyone else."""
    return np.where(np.asarray(eligible, bool), np.asarray(award, float),
                    np.asarray(base_reported, float))


def jsa_taxed(variables: dict) -> bool:
    """True when the model's income tax base includes new-style JSA:
    ``jsa_contrib_reported`` in ``social_security_income`` and that in
    ``taxable_social_security_income``. ``variables`` maps a variable name to
    an object with an ``adds`` list (the tax-benefit system's variables)."""
    def adds(name):
        return list(getattr(variables[name], "adds", None) or [])
    return ("jsa_contrib_reported" in adds("social_security_income")
            and "social_security_income" in adds("taxable_social_security_income"))


def obr_table(persons: pd.DataFrame, shocked: pd.DataFrame) -> pd.DataFrame:
    """Displacement only (run 15's table): survivors and capital at baseline."""
    t = persons.copy()
    disp = shocked["displaced"].to_numpy(bool)
    t["displaced"] = disp
    t["employment_income"] = np.where(disp, 0.0, persons["employment_income"].to_numpy(float))
    return t


def check_pairing(got_m: float, ref, seed: int, label: str, rtol: float = 1e-9) -> None:
    """Assert this draw displaces the same weighted count as the reference
    run's draw of the same seed (``ref`` maps seed -> displaced millions)."""
    if seed not in ref:
        raise AssertionError(f"{label}: no reference draw for seed {seed}")
    if not np.isclose(got_m, float(ref[seed]), rtol=rtol):
        raise AssertionError(f"{label} seed {seed}: displaced {got_m} != reference {float(ref[seed])}")


#: scalar metrics whose shocked-minus-baseline change is reported (pounds
#: -> bn, or rates -> percentage points)
BN_CHANGES = {
    "gov_balance_change_bn": "gov", "income_tax_change_bn": "income_tax",
    "ni_employee_change_bn": "ni_employee", "national_insurance_change_bn": "national_insurance",
    "employer_ni_change_bn": "ni_employer", "uc_change_bn": "uc", "benefits_change_bn": "benefits",
    "cash_benefits_change_bn": "cash_benefits", "childcare_change_bn": "childcare",
    "statutory_pay_change_bn": "statutory_pay", "jsa_change_bn": "jsa",
    "household_net_income_change_bn": "hh_net_total",
}
PP_CHANGES = {
    "poverty_change_bhc_pp": "pov_bhc", "poverty_change_ahc_pp": "pov_ahc",
    "relative_poverty_change_bhc_pp": "rel_bhc", "relative_poverty_change_ahc_pp": "rel_ahc",
    "gini_change_pp": "gini",
}


def draw_changes(base: dict, shocked: dict) -> dict:
    """Changes of one draw from scalar metric dicts. ``exchequer_cost_bn`` is
    the fall in the government balance; ``household_basis_cost_bn`` is income
    tax + employee NI lost plus benefits added (as in run 15) and
    ``household_basis_cash_cost_bn`` the same with cash benefits only."""
    out = {k: (shocked[v] - base[v]) / 1e9 for k, v in BN_CHANGES.items()}
    out.update({k: 100 * (shocked[v] - base[v]) for k, v in PP_CHANGES.items()})
    out["exchequer_cost_bn"] = -out["gov_balance_change_bn"]
    tax = out["income_tax_change_bn"] + out["ni_employee_change_bn"]
    out["household_basis_cost_bn"] = out["benefits_change_bn"] - tax
    out["household_basis_cash_cost_bn"] = out["cash_benefits_change_bn"] - tax
    # the absorption-rate basis of tax_composition.py (IT + NI - benefits)
    out["narrow_cash_change_bn"] = (out["income_tax_change_bn"] + out["national_insurance_change_bn"]
                                    - out["benefits_change_bn"])
    return out


def decile_flows(deciles, weights, flows: dict) -> pd.DataFrame:
    """Weighted household flows (pounds) summed by baseline decile."""
    w = np.asarray(weights, float)
    df = pd.DataFrame({k: np.asarray(v, float) * w for k, v in flows.items()})
    df["decile"] = np.asarray(deciles)
    return df.groupby("decile").sum().reset_index()


def _wshare(w, mask, within=None) -> float:
    w = np.asarray(w, float)
    base = np.ones(len(w), bool) if within is None else np.asarray(within, bool)
    den = w[base].sum()
    return float(w[base & np.asarray(mask, bool)].sum() / den) if den > 0 else float("nan")


def displaced_diagnostics(*, bu_weight, displaced_bu, flag_base, entitled_before, entitled_after,
                          capital, capital_limit, other_earner, owner,
                          person_weight, displaced, baseline_earnings) -> dict:
    """Who the displaced are and whether they claim (benefit units weighted
    by household weight; persons by person weight).

    ``entitled_*``: positive UC award before the benefit cap if claimed, in
    the paired baseline and in the take-up = 1 shock. ``flag_base``: the
    unit's baseline take-up flag (the central assumption)."""
    d = np.asarray(displaced_bu, bool)
    after = np.asarray(entitled_after, bool)
    newly = after & ~np.asarray(entitled_before, bool)
    pw = np.asarray(person_weight, float)
    pd_ = np.asarray(displaced, bool)
    return {
        "displaced_benunits_m": float(np.asarray(bu_weight, float)[d].sum() / 1e6),
        "flag_true_share": _wshare(bu_weight, flag_base, d),
        "entitled_after_share": _wshare(bu_weight, after, d),
        "flag_true_share_entitled_after": _wshare(bu_weight, flag_base, d & after),
        "flag_true_share_newly_entitled": _wshare(bu_weight, flag_base, d & newly),
        "over_capital_limit_share": _wshare(bu_weight, np.asarray(capital, float) > capital_limit, d),
        "other_earner_share": _wshare(bu_weight, other_earner, d),
        "owner_share": _wshare(bu_weight, owner, d),
        "mean_lost_earnings_gbp": float((np.asarray(baseline_earnings, float) * pw)[pd_].sum() / pw[pd_].sum()),
    }


def existing_unemployed_reference(*, bu_weight, n_unemployed, flag, entitled, uc) -> dict:
    """Benefit units with an unemployed claimant or partner aged 18-65 in the
    unshocked baseline: their UC per unemployed person and the take-up flag
    among those entitled (``n_unemployed`` counts such people per unit)."""
    n = np.asarray(n_unemployed, float)
    has = n > 0
    w = np.asarray(bu_weight, float)
    return {
        "persons_m": float((w * n)[has].sum() / 1e6),
        "benunits_m": float(w[has].sum() / 1e6),
        "uc_per_person_gbp": float((w * np.asarray(uc, float))[has].sum() / (w * n)[has].sum()),
        "entitled_share": _wshare(w, entitled, has),
        "flag_true_share_entitled": _wshare(w, flag, has & np.asarray(entitled, bool)),
    }


DRAW_COLS = tuple(BN_CHANGES) + tuple(PP_CHANGES) + (
    "exchequer_cost_bn", "household_basis_cost_bn", "household_basis_cash_cost_bn", "narrow_cash_change_bn",
    "displaced_weighted_m", "wage_bill_lost_bn", "jsa_eligible_m", "jsa_award_input_bn")


def summarise(draws: pd.DataFrame, cols=DRAW_COLS) -> dict:
    out = {}
    for c in cols:
        if c not in draws:
            continue
        x = draws[c].astype(float)
        n = len(x)
        out[c] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if n > 1 else float("nan"),
                  "se": float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"), "n": int(n)}
    return out


# ---------------------------------------------------------------------------
# Simulation driver (needs Microcosm; not run in the revision of 13 Nov 2026)
# ---------------------------------------------------------------------------
def _sim_metrics(sim) -> dict:  # pragma: no cover - needs Microcosm
    from uk_ai_study.runner import gini

    def p(v):
        return np.asarray(sim.calculate(v, period=PERIOD, map_to="person").values)

    def h(v):
        return np.asarray(sim.calculate(v, period=PERIOD, map_to="household").values, float)

    pw, hw = p("person_weight").astype(float), h("household_weight")
    hh = {
        "hh_market": h("household_market_income"), "hh_benefits": h("household_benefits"),
        "hh_uc": h("universal_credit"), "hh_jsa": h("jsa_contrib"),
        "hh_childcare": sum(h(v) for v in CHILDCARE_VARS),
        "hh_statutory_pay": sum(h(v) for v in STATUTORY_PAY_VARS),
        "hh_tax": h("income_tax") + h("ni_employee"), "hh_net": h("hbai_household_net_income"),
    }
    hh["hh_cash_benefits"] = hh["hh_benefits"] - hh["hh_childcare"] - hh["hh_statutory_pay"]
    tot = {k: float((x * hw).sum()) for k, x in hh.items()}
    return {
        **hh, "hw": hw,
        "gov": float((h("gov_balance") * hw).sum()),
        "income_tax": float((p("income_tax") * pw).sum()),
        "ni_employee": float((p("ni_employee") * pw).sum()),
        "national_insurance": float((p("national_insurance") * pw).sum()),
        "ni_employer": float((p("ni_employer") * pw).sum()),
        "uc": tot["hh_uc"], "benefits": tot["hh_benefits"], "cash_benefits": tot["hh_cash_benefits"],
        "childcare": tot["hh_childcare"], "statutory_pay": tot["hh_statutory_pay"], "jsa": tot["hh_jsa"],
        "hh_net_total": tot["hh_net"],
        "pov_bhc": float(np.average(p("in_poverty_bhc"), weights=pw)),
        "pov_ahc": float(np.average(p("in_poverty_ahc"), weights=pw)),
        "rel_bhc": float(np.average(p("in_relative_poverty_bhc"), weights=pw)),
        "rel_ahc": float(np.average(p("in_relative_poverty_ahc"), weights=pw)),
        "gini": gini(h("equiv_hbai_household_net_income"), hw * h("household_count_people")),
    }


def _entitled_if_claimed(sim) -> np.ndarray:  # pragma: no cover
    """Benefit units with a positive UC award before the benefit cap were
    they to claim: eligible, maximum amount above the income reduction, and
    not taking Tax-Free Childcare instead. None of these depends on the
    take-up flag."""
    def b(v):
        return np.asarray(sim.calculate(v, period=PERIOD).values)
    return (b("is_uc_eligible").astype(bool)
            & (b("uc_maximum_amount").astype(float) - b("uc_income_reduction").astype(float) > 0)
            & ~b("forgoes_universal_credit_for_tax_free_childcare").astype(bool))


def _set_checked(sim, var: str, values) -> None:  # pragma: no cover
    """set_input that fails hard if the simulation does not take the values."""
    sim.set_input(var, PERIOD, values)
    got = np.asarray(sim.calculate(var, period=PERIOD).values)
    if not np.array_equal(got.astype(np.asarray(values).dtype), np.asarray(values)):
        raise RuntimeError(f"{var}: set_input not applied")


def _append_deciles(dec: pd.DataFrame, key: dict) -> None:  # pragma: no cover
    for k, val in reversed(list(key.items())):
        dec.insert(0, k, val)
    if DECILES.exists():
        old = pd.read_csv(DECILES)
        mask = np.ones(len(old), bool)
        for k, val in key.items():
            mask &= old[k].astype(str).to_numpy() == str(val)
        if mask.any():
            old[~mask].to_csv(DECILES, index=False)
    dec.to_csv(DECILES, mode="a", header=not DECILES.exists(), index=False)


def main(only: str | None = None) -> None:  # pragma: no cover - needs Microcosm
    import gc

    from uk_ai_study.checkpoint import Checkpoint
    from uk_ai_study.data import microsimulation
    from uk_ai_study.runner import build_person_table, weighted_deciles
    from uk_ai_study.shocks import PRESETS, TRANSITION_ZEROED_VARIABLES, ShockScenario, apply_shocks, \
        build_shocked_simulation

    OUT.mkdir(parents=True, exist_ok=True)
    R = ROOT / "results"
    obr_meta = json.loads((R / "next_runs" / "obr_benchmark.json").read_text())
    obr_rate = float(obr_meta["displacement_rate"])
    obr_ref = pd.read_csv(R / "next_runs" / "obr_benchmark_draws.csv").set_index("seed")
    if not np.allclose(obr_ref["displacement_rate"].astype(float), obr_rate, rtol=1e-12):
        raise AssertionError("obr_benchmark_draws.csv and obr_benchmark.json disagree on the rate")
    dc = pd.read_csv(R / "robustness" / "dividend_coverage_draws.csv")
    central_ref = dc[dc["run"] == "central"].set_index("seed")["displaced_weighted_m"]
    grid = pd.read_csv(R / "tax_composition" / "composition_grid.csv")
    u7_wage_bill_bn = float(grid[grid.unemployment_pct == 7]["w_lost_bn"].iloc[0])
    refs = {"obr": obr_ref["displaced_weighted_m"], "central": central_ref, "disp_only": central_ref}

    baseline = microsimulation()
    system = baseline.tax_benefit_system
    persons = build_person_table(baseline, PERIOD)

    def bp(v, sim=baseline):
        return np.asarray(sim.calculate(v, period=PERIOD, map_to="person").values)

    def bb(v, sim=baseline):
        return np.asarray(sim.calculate(v, period=PERIOD).values)

    arrays = {v: bp(v).astype(float) for v in TRANSITION_ZEROED_VARIABLES}
    arrays["employment_status"] = bp("employment_status").astype(object)
    b0 = _sim_metrics(baseline)
    hh_decile = weighted_deciles(np.asarray(baseline.calculate(
        "equiv_hbai_household_net_income", period=PERIOD, map_to="household").values), b0["hw"] * np.asarray(
        baseline.calculate("household_count_people", period=PERIOD, map_to="household").values, float))
    person_bu = bp("benunit_id")
    bu_id = bb("benunit_id")
    bu_pos = pd.Series(np.arange(len(bu_id)), index=bu_id).loc[person_bu].to_numpy()
    bu_weight = bb("benunit_weight").astype(float)
    base_flags = bb("would_claim_uc").astype(bool)
    base_jsa_reported = bp("jsa_contrib_reported").astype(float)
    age = persons["age"].to_numpy(float)
    earnings = persons["employment_income"].to_numpy(float)
    cp = bp("is_claimant_or_partner").astype(bool)
    earned_other = earnings + bp("self_employment_income").astype(float)
    # household tenure of each benefit unit, through its members
    owner_bu = np.bincount(bu_pos, weights=np.isin(bp("tenure_type").astype(str), OWNER_TENURES).astype(float),
                           minlength=len(bu_id)) > 0

    params = system.parameters(f"{PERIOD}-01-01")
    jsa_p = params.gov.dwp.JSA.contrib
    rate_25, rate_18_24 = float(jsa_p.amount_over_25), float(jsa_p.amount_18_24)
    lel_week = float(params.gov.hmrc.national_insurance.class_1.thresholds.lower_earnings_limit)
    capital_limit = float(params.gov.dwp.universal_credit.means_test.capital.limit)
    jsa_facts = {"rate_25_plus_week": rate_25, "rate_18_24_week": rate_18_24, "weeks": JSA_WEEKS,
                 "lower_earnings_limit_week": lel_week, "earnings_condition_gbp": LEL_WEEKS * lel_week,
                 "ages": [JSA_MIN_AGE, JSA_MAX_AGE - 1], "taxed_by_model": jsa_taxed(system.variables),
                 "counts_as_uc_unearned_income": True}

    # existing unemployed (claimant or partner, 18-65) in the unshocked baseline
    unemployed = (arrays["employment_status"].astype(str) == "UNEMPLOYED") & cp & (age >= 18) & (age < 66)
    unemployed_ref = existing_unemployed_reference(
        bu_weight=bu_weight, n_unemployed=np.bincount(bu_pos, weights=unemployed.astype(float),
                                                      minlength=len(bu_id)),
        flag=base_flags, entitled=_entitled_if_claimed(baseline), uc=bb("universal_credit").astype(float))
    print(f"baseline ready; JSA {jsa_facts}; existing unemployed {unemployed_ref}", flush=True)
    del baseline
    gc.collect()

    ck = Checkpoint(DRAWS, ("scenario", "variant", "seed"), script=__file__)
    if not ck.done:
        DECILES.unlink(missing_ok=True)

    for scen_name in SCENARIOS:
        if only is not None and scen_name != only:
            continue
        for seed in SCENARIO_SEEDS[scen_name]:
            if all(ck.is_done(scen_name, v, seed) for v in VARIANTS):
                continue
            if scen_name == "central":
                table = apply_shocks(persons, PRESETS["central"], seed=seed)
            elif scen_name == "obr":
                table = obr_table(persons, apply_shocks(
                    persons, ShockScenario("obr_benchmark", obr_rate, 0.0, capital_return_increase=0.0), seed=seed))
            else:
                table = apply_shocks(persons, ShockScenario("disp_only_u7", DISP_ONLY_RATE, 0.0,
                                                            capital_return_increase=0.0), seed=seed)
            disp = table["displaced"].to_numpy(bool)
            w = persons["weight"].to_numpy(float)
            disp_m = float(w[disp].sum() / 1e6)
            check_pairing(disp_m, refs[scen_name], seed, scen_name)
            wage_bill_bn = float((earnings * w)[disp].sum() / 1e9)
            if scen_name == "disp_only" and not np.isclose(wage_bill_bn, u7_wage_bill_bn, rtol=1e-6):
                raise AssertionError(f"disp_only wage bill {wage_bill_bn} != composition grid {u7_wage_bill_bn}")
            if scen_name == "obr":
                check_pairing(wage_bill_bn, obr_ref["wage_bill_lost_bn"], seed, "obr wage bill", rtol=1e-6)

            d_bu = displaced_benunits(person_bu, bu_id, disp)
            flags = takeup_flags(base_flags, d_bu)
            elig = jsa_eligible(age, earnings, disp, lel_week)
            award = jsa_award(age, elig, rate_25, rate_18_24)
            other_earner = np.bincount(bu_pos, weights=((earned_other > 0) & ~disp & cp).astype(float),
                                       minlength=len(bu_id)) > 0

            base = microsimulation()
            _set_checked(base, "would_claim_uc", flags)
            bm = _sim_metrics(base)
            ent_before = _entitled_if_claimed(base)
            del base
            gc.collect()

            for variant in VARIANTS:
                if ck.is_done(scen_name, variant, seed):
                    continue
                sim = build_shocked_simulation(None, table, PERIOD, base_arrays=arrays)
                _set_checked(sim, "would_claim_uc", flags)
                if variant == "jsa":
                    sim.set_input("jsa_contrib_reported", PERIOD, jsa_reported_input(base_jsa_reported, award, elig))
                    got = np.asarray(sim.calculate("jsa_contrib", period=PERIOD, map_to="person").values, float)
                    if not np.allclose(got[elig], award[elig]):
                        raise RuntimeError("jsa_contrib_reported not applied to the eligible displaced")
                sm = _sim_metrics(sim)
                diag = {}
                if variant == "takeup1":
                    diag = displaced_diagnostics(
                        bu_weight=bu_weight, displaced_bu=d_bu, flag_base=base_flags,
                        entitled_before=ent_before, entitled_after=_entitled_if_claimed(sim),
                        capital=np.asarray(sim.calculate("uc_assessable_capital", period=PERIOD).values, float),
                        capital_limit=capital_limit, other_earner=other_earner, owner=owner_bu,
                        person_weight=w, displaced=disp, baseline_earnings=earnings)
                del sim
                gc.collect()
                flows = {k: sm[k] - bm[k] for k in ("hh_market", "hh_benefits", "hh_cash_benefits", "hh_uc",
                                                    "hh_jsa", "hh_childcare", "hh_statutory_pay", "hh_tax",
                                                    "hh_net")}
                _append_deciles(decile_flows(hh_decile, b0["hw"], flows),
                                {"scenario": scen_name, "variant": variant, "seed": seed})
                ck.append({"scenario": scen_name, "variant": variant, "seed": seed,
                           **draw_changes(bm, sm), "displaced_weighted_m": disp_m, "wage_bill_lost_bn": wage_bill_bn,
                           "jsa_eligible_m": float(w[elig].sum() / 1e6) if variant == "jsa" else 0.0,
                           "jsa_award_input_bn": float((award * w).sum() / 1e9) if variant == "jsa" else 0.0,
                           **{f"diag_{k}": v for k, v in diag.items()}})
                print(f"{scen_name} {variant} seed {seed} done", flush=True)

    draws = ck.read()
    summary = {"jsa": jsa_facts, "existing_unemployed": unemployed_ref, "capital_limit_gbp": capital_limit,
               "obr_displacement_rate": obr_rate, "disp_only_rate": DISP_ONLY_RATE,
               "complete": bool(len(draws) == EXPECTED_ROWS), "results": {}, "diagnostics": {}}
    for (s, v), d in draws.groupby(["scenario", "variant"]):
        summary["results"].setdefault(s, {})[v] = summarise(d)
        if v == "takeup1":
            dcols = [c for c in d.columns if c.startswith("diag_")]
            summary["diagnostics"][s] = {c[5:]: float(d[c].astype(float).mean()) for c in dcols}
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"wrote {SUMMARY} ({len(draws)} of {EXPECTED_ROWS} draws)", flush=True)


if __name__ == "__main__":  # pragma: no cover
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    if only is not None and only not in SCENARIOS:
        raise SystemExit(f"unknown scenario {only}; choose from {SCENARIOS}")
    main(only)
