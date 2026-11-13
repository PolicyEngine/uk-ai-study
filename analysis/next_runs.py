#!/usr/bin/env python3
"""Runs 15 and 16 of the Fiscal Studies revision: specified, not yet run.

Specification: the run spec written on 13 Nov 2026 (copied into the PR #18
description). Nothing here has been executed against Microcosm; the table
builders are unit-tested on fake person tables in
``tests/test_next_runs.py``. Run one stage at a time:

    python analysis/next_runs.py obr        # run 15: 20 paired draws
    python analysis/next_runs.py capital    # run 16: 5 variants x 20 draws

Run 15 (``obr``): the OBR March 2026 technological-displacement benchmark.
    Displacement sized so that the unemployment rate rises by
    ``OBR_UNEMPLOYMENT_RISE_PP`` points of the labour force (4.1 -> 5.5 per
    cent), exposure-proportional incidence, no survivor-wage uplift and no
    capital-income uplift (flat earnings, no capital windfall). Seeds 0-19.

Run 16 (``capital``): capital-channel variants on the central 7 per cent
    exposure-proportional draw with the +2.6 per cent survivor uplift, seeds
    0-19 (the seeds of the existing channel split, so every variant pairs
    with ``dividend_coverage_draws.csv`` draw by draw):
      cap10, cap20,     interest and dividends x1.10, x1.20 and x1.60;
      cap60
      div_only          dividends x(1.405/1.005), interest unchanged;
      closed_k*         no exogenous capital uplift; instead household
                        dividends rise by kappa x (W - S), where W is the
                        draw's displaced wage bill and S the survivors' gain
                        from the uplift, pro rata to baseline dividend
                        holders. kappa = (1 - CT) x phi x payout x UK share:
                        0.375, 0.1875, 0.09375 are phi = 1, 0.5, 0.25 at the
                        50 per cent payout of Table 6; 0.75 is the
                        full-payout upper bound. Corporation tax on phi x
                        (W - S) is added in post-processing.
Variants run in the order of ``CAPITAL_VARIANTS``, 20 seeds each.

Every draw row also records the displaced wage bill, the survivor gain, the
capital income added and the household-basis cost (income tax + employee NI
- benefits) next to the government balance, and run 15 writes the
market-income, benefit and tax changes by baseline income decile, so the
stabilisation coefficients of the paper can be computed per draw.

Outputs (results/next_runs/):
    obr_benchmark_draws.csv, obr_benchmark_deciles.csv, obr_benchmark.json
    capital_variants_draws.csv, capital_variants.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "next_runs"
PERIOD = 2026
SEEDS = tuple(range(20))
CT_RATE = 0.25
BASELINE_CAPITAL_RETURN = 0.01005
CAPITAL_RETURN_INCREASE = 0.004

#: OBR, Economic and fiscal outlook, March 2026, Box 2.2: equilibrium
#: unemployment 5.5 per cent against 4.1 per cent in the central forecast.
OBR_UNEMPLOYMENT_BASE_PCT = 4.1
OBR_UNEMPLOYMENT_SCENARIO_PCT = 5.5
OBR_UNEMPLOYMENT_RISE_PP = OBR_UNEMPLOYMENT_SCENARIO_PCT - OBR_UNEMPLOYMENT_BASE_PCT
#: OBR fiscal effect, para 6.18: borrowing ~£9bn/yr higher on average,
#: receipts ~£6bn lower, welfare ~£3bn higher (the comparators).
OBR_BORROWING_BN, OBR_RECEIPTS_BN, OBR_WELFARE_BN = 9.0, 6.0, 3.0


def obr_displacement_rate(labour_force_m: float, employees_m: float,
                          rise_pp: float = OBR_UNEMPLOYMENT_RISE_PP) -> float:
    """Employee displacement rate that adds ``rise_pp`` points to the
    unemployment rate of a labour force of ``labour_force_m`` million.

    The OBR's rate is a share of the labour force (employees, the
    self-employed and the unemployed); the study displaces employees only,
    so the head-count target is rise_pp% of the labour force, expressed as a
    share of Microcosm's weighted employees.
    """
    if labour_force_m <= employees_m:
        raise ValueError("the labour force must exceed the employee count")
    return rise_pp / 100 * labour_force_m / employees_m


LABOUR_FORCE_STATUSES = ("FT_EMPLOYED", "PT_EMPLOYED", "FT_SELF_EMPLOYED", "PT_SELF_EMPLOYED", "UNEMPLOYED")


def labour_force_m(status, age, weights) -> float:
    """Weighted labour force (millions): people aged 16+ who are employed,
    self-employed or unemployed in the baseline. Run 15 uses Microcosm's own
    labour force by default, so the head-count target and the employee base
    come from the same data; ``--labour-force-m`` overrides it."""
    status = np.asarray(status).astype(str)
    active = np.isin(status, LABOUR_FORCE_STATUSES) & (np.asarray(age) >= 16)
    return float(np.asarray(weights, float)[active].sum() / 1e6)


@dataclass(frozen=True)
class CapitalVariant:
    name: str
    interest_factor: float = 1.0
    dividend_factor: float = 1.0
    #: closed-economy pass-through to household dividends:
    #: kappa = (1 - CT) x phi x payout x UK share, applied to (W - S).
    #: Table 6 (50 per cent payout, UK share 1): phi = 1 -> 0.375,
    #: phi = 0.5 -> 0.1875, phi = 0.25 -> 0.09375; 0.75 is the full-payout
    #: upper bound. Corporation tax stays in post-processing.
    closed_kappa: float | None = None


JR16_FACTOR = (BASELINE_CAPITAL_RETURN + CAPITAL_RETURN_INCREASE) / BASELINE_CAPITAL_RETURN
CAPITAL_VARIANTS = (
    CapitalVariant("cap10", 1.10, 1.10),
    CapitalVariant("cap20", 1.20, 1.20),
    # decision (c), 13 Nov 2026: +60 per cent brackets the extrapolated
    # +57 per cent capital break-even
    CapitalVariant("cap60", 1.60, 1.60),
    CapitalVariant("div_only", 1.0, JR16_FACTOR),
    CapitalVariant("closed_k0375", closed_kappa=0.375),
    CapitalVariant("closed_k01875", closed_kappa=0.1875),
    CapitalVariant("closed_k009375", closed_kappa=0.09375),
    CapitalVariant("closed_k075", closed_kappa=0.75),
)
#: The tuple order is the execution order of run 16 (variant by variant, 20
#: seeds each), so a stopped run leaves whole variants complete.


def wage_flows(persons: pd.DataFrame, shocked: pd.DataFrame) -> dict:
    """Displaced wage bill W and survivor gain S (pounds, weighted)."""
    w = persons["weight"].to_numpy(float)
    base = persons["employment_income"].to_numpy(float)
    new = shocked["employment_income"].to_numpy(float)
    disp = shocked["displaced"].to_numpy(bool)
    surv = (base > 0) & ~disp
    return {"wage_bill_lost": float((base * w)[disp].sum()),
            "survivor_gain": float(((new - base) * w)[surv].sum())}


def capital_variant_table(persons: pd.DataFrame, shocked: pd.DataFrame, v: CapitalVariant) -> pd.DataFrame:
    """Replace the shocked table's capital incomes with variant ``v``.

    ``shocked`` is the central draw (displacement + uplift + JR16 capital);
    its labour columns are kept and its capital columns are rebuilt from the
    baseline person table.
    """
    t = shocked.copy()
    w = persons["weight"].to_numpy(float)
    interest = persons["savings_interest_income"].to_numpy(float)
    dividends = persons["dividend_income"].to_numpy(float)
    if v.closed_kappa is None:
        t["savings_interest_income"] = interest * v.interest_factor
        t["dividend_income"] = dividends * v.dividend_factor
        return t
    f = wage_flows(persons, shocked)
    target = v.closed_kappa * (f["wage_bill_lost"] - f["survivor_gain"])
    total = float((dividends * w).sum())
    if total <= 0:
        raise ValueError("no baseline dividends to scale")
    t["savings_interest_income"] = interest
    t["dividend_income"] = dividends * (1 + max(target, 0.0) / total)
    t.attrs["closed_wedge_bn"] = (f["wage_bill_lost"] - f["survivor_gain"]) / 1e9
    return t


def obr_table(persons: pd.DataFrame, shocked: pd.DataFrame) -> pd.DataFrame:
    """Displacement only: survivors and capital incomes at baseline."""
    t = persons.copy()
    disp = shocked["displaced"].to_numpy(bool)
    t["displaced"] = disp
    t["employment_income"] = np.where(disp, 0.0, persons["employment_income"].to_numpy(float))
    return t


def decile_flows(deciles, weights, d_market, d_benefits, d_tax, d_net=None) -> pd.DataFrame:
    """Weighted sums of household changes by baseline decile, and the
    stabilisation coefficient (share of the market-income change offset by
    benefits and lower taxes). Arrays are household-level."""
    df = pd.DataFrame({"decile": deciles, "w": weights, "m": d_market, "b": d_benefits, "t": d_tax})
    g = df.assign(m=df.m * df.w, b=df.b * df.w, t=df.t * df.w).groupby("decile")[["m", "b", "t"]].sum()
    g.columns = ["market_change", "benefit_change", "tax_change"]
    g["stabilisation"] = np.where(g.market_change < 0,
                                  (g.benefit_change - g.tax_change) / -g.market_change, np.nan)
    if d_net is not None:
        g["net_income_change"] = (pd.Series(np.asarray(d_net, float) * np.asarray(weights, float))
                                  .groupby(np.asarray(deciles)).sum())
    return g.reset_index()


def decile_family_flows(deciles, family, weights, d_market, d_benefits, d_tax, d_net) -> pd.DataFrame:
    """As ``decile_flows`` by baseline decile x family type (households)."""
    df = pd.DataFrame({"decile": deciles, "family_type": family, "w": weights})
    for k, x in (("market_change", d_market), ("benefit_change", d_benefits),
                 ("tax_change", d_tax), ("net_income_change", d_net)):
        df[k] = np.asarray(x, float) * df.w
    g = df.groupby(["decile", "family_type"])[["market_change", "benefit_change", "tax_change",
                                                "net_income_change", "w"]].sum()
    g = g.rename(columns={"w": "weight"})
    g["stabilisation"] = np.where(g.market_change < 0,
                                  (g.benefit_change - g.tax_change) / -g.market_change, np.nan)
    return g.reset_index()


def lf_components(status, age, weights) -> dict:
    """Weighted labour-force components (millions) and the implied
    unemployment rate (per cent) of the baseline."""
    status = np.asarray(status).astype(str)
    w = np.asarray(weights, float)
    adult = np.asarray(age) >= 16
    c = {k: float(w[adult & np.isin(status, sts)].sum() / 1e6) for k, sts in (
        ("employed_m", ("FT_EMPLOYED", "PT_EMPLOYED")),
        ("self_employed_m", ("FT_SELF_EMPLOYED", "PT_SELF_EMPLOYED")),
        ("unemployed_m", ("UNEMPLOYED",)))}
    c["labour_force_m"] = c["employed_m"] + c["self_employed_m"] + c["unemployed_m"]
    c["unemployment_rate_pct"] = 100 * c["unemployed_m"] / c["labour_force_m"]
    return c


def summarise(draws: pd.DataFrame, cols) -> dict:
    out = {}
    for c in cols:
        x = draws[c].astype(float)
        out[c] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                  "se": float(x.std(ddof=1) / np.sqrt(len(x))), "n": int(len(x))}
    return out


DRAW_COLS = ("exchequer_cost_bn", "household_basis_cost_bn", "employer_ni_change_bn",
             "uc_change_bn", "benefits_change_bn", "poverty_change_bhc_pp", "poverty_change_ahc_pp",
             "relative_poverty_change_bhc_pp", "gini_change_pp", "displaced_weighted_m",
             "wage_bill_lost_bn", "survivor_gain_bn", "capital_income_added_bn",
             "public_paybill_displaced_bn", "income_tax_change_bn", "ni_employee_change_bn",
             "childcare_change_bn", "statutory_pay_change_bn", "displaced_employer_pension_bn")
CHILDCARE_VARS = ("tax_free_childcare", "extended_childcare_entitlement", "universal_childcare_entitlement",
                  "targeted_childcare_entitlement", "wales_childcare_offer")  # = runner.CHILDCARE_VARIABLES
STATUTORY_PAY_VARS = ("statutory_sick_pay", "statutory_maternity_pay", "statutory_paternity_pay")
#: household arrays per draw (compressed npz) live outside results/
ARRAYS_DIR = ROOT.parent / "ai-study-household-arrays"


def public_paybill(persons: pd.DataFrame, table: pd.DataFrame, public, employer_costs) -> float:
    """Pay bill saved by the public sector (pounds, weighted): baseline
    earnings plus employer NI and employer pension contributions of displaced
    public-sector employees. Output only (decision (b), 13 Nov 2026):
    gov_balance has no pay-bill item, so this is reported beside it and not
    netted from the Exchequer cost."""
    w = persons["weight"].to_numpy(float)
    disp = table["displaced"].to_numpy(bool) & np.asarray(public, bool)
    cost = persons["employment_income"].to_numpy(float) + np.asarray(employer_costs, float)
    return float((cost * w)[disp].sum())


# ---------------------------------------------------------------------------
# Simulation driver (not executed in the revision of 13 Nov 2026)
# ---------------------------------------------------------------------------
def _sim_metrics(sim) -> dict:  # pragma: no cover - needs Microcosm
    from uk_ai_study.runner import gini

    def p(v):
        return sim.calculate(v, period=PERIOD, map_to="person").values

    def h(v):
        return sim.calculate(v, period=PERIOD, map_to="household").values
    pw, hw = p("person_weight"), h("household_weight")
    return {
        "gov": float((h("gov_balance") * hw).sum()),
        "hh_tax": float(((p("income_tax") + p("ni_employee")) * pw).sum()),
        "income_tax": float((p("income_tax") * pw).sum()),
        "ni_employee": float((p("ni_employee") * pw).sum()),
        "childcare": float((sum(h(v) for v in CHILDCARE_VARS) * hw).sum()),
        "statutory_pay": float((sum(p(v) for v in STATUTORY_PAY_VARS) * pw).sum()),
        "hh_net": h("hbai_household_net_income"),
        "hh_uc": h("universal_credit"),
        "hh_it": sim.calculate("income_tax", period=PERIOD, map_to="household").values,
        "hh_ni": sim.calculate("ni_employee", period=PERIOD, map_to="household").values,
        "employer_ni": float((p("ni_employer") * pw).sum()),
        "benefits": float((h("household_benefits") * hw).sum()),
        "uc": float((h("universal_credit") * hw).sum()),
        "pov_bhc": float(np.average(p("in_poverty_bhc"), weights=pw)),
        "pov_ahc": float(np.average(p("in_poverty_ahc"), weights=pw)),
        "rel_bhc": float(np.average(p("in_relative_poverty_bhc"), weights=pw)),
        "gini": gini(h("equiv_hbai_household_net_income"), hw * h("household_count_people")),
        "hh_market": h("household_market_income"), "hh_benefits": h("household_benefits"),
        "hh_tax_arr": sim.calculate("income_tax", period=PERIOD, map_to="household").values
        + sim.calculate("ni_employee", period=PERIOD, map_to="household").values,
        "hw": hw,
        "equiv": h("equiv_hbai_household_net_income"),
        "count": h("household_count_people"),
    }


def _row(base, shocked, persons, table, public=None, employer_costs=None,
         employer_pension=None) -> dict:  # pragma: no cover
    f = wage_flows(persons, table)
    w = persons["weight"].to_numpy(float)
    cap = sum(((table[c].to_numpy(float) - persons[c].to_numpy(float)) * w).sum()
              for c in ("savings_interest_income", "dividend_income"))
    return {
        "exchequer_cost_bn": (base["gov"] - shocked["gov"]) / 1e9,
        "household_basis_cost_bn": ((base["hh_tax"] - shocked["hh_tax"])
                                    + (shocked["benefits"] - base["benefits"])) / 1e9,
        "employer_ni_change_bn": (shocked["employer_ni"] - base["employer_ni"]) / 1e9,
        "uc_change_bn": (shocked["uc"] - base["uc"]) / 1e9,
        "benefits_change_bn": (shocked["benefits"] - base["benefits"]) / 1e9,
        "poverty_change_bhc_pp": 100 * (shocked["pov_bhc"] - base["pov_bhc"]),
        "poverty_change_ahc_pp": 100 * (shocked["pov_ahc"] - base["pov_ahc"]),
        "relative_poverty_change_bhc_pp": 100 * (shocked["rel_bhc"] - base["rel_bhc"]),
        "gini_change_pp": 100 * (shocked["gini"] - base["gini"]),
        "displaced_weighted_m": float(w[table["displaced"].to_numpy(bool)].sum() / 1e6),
        "wage_bill_lost_bn": f["wage_bill_lost"] / 1e9,
        "survivor_gain_bn": f["survivor_gain"] / 1e9,
        "capital_income_added_bn": float(cap) / 1e9,
        "public_paybill_displaced_bn": (np.nan if public is None
                                        else public_paybill(persons, table, public, employer_costs) / 1e9),
        "income_tax_change_bn": (shocked["income_tax"] - base["income_tax"]) / 1e9,
        "ni_employee_change_bn": (shocked["ni_employee"] - base["ni_employee"]) / 1e9,
        "childcare_change_bn": (shocked["childcare"] - base["childcare"]) / 1e9,
        "statutory_pay_change_bn": (shocked["statutory_pay"] - base["statutory_pay"]) / 1e9,
        "displaced_employer_pension_bn": (np.nan if employer_pension is None else float(
            (np.asarray(employer_pension, float) * w)[table["displaced"].to_numpy(bool)].sum()) / 1e9),
    }


def _draw_outputs(b, s, hh_decile, family, dec_path, fam_path, key, tag) -> None:  # pragma: no cover
    """Decile, decile x family and household-array outputs of one draw,
    written before the draw's checkpoint row."""
    d = {k: s[k] - b[k] for k in ("hh_market", "hh_benefits", "hh_tax_arr", "hh_net")}
    _append_deciles(dec_path, decile_flows(hh_decile, b["hw"], d["hh_market"], d["hh_benefits"],
                                           d["hh_tax_arr"], d["hh_net"]), dict(key))
    if family is not None:
        _append_deciles(fam_path, decile_family_flows(hh_decile, family, b["hw"], d["hh_market"],
                                                      d["hh_benefits"], d["hh_tax_arr"], d["hh_net"]), dict(key))
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAYS_DIR / f"{tag}.npz", **{k: s[k] for k in (
        "hh_market", "hh_net", "hh_it", "hh_ni", "hh_benefits", "hh_uc")})


def _append_deciles(path: Path, dec: pd.DataFrame, key: dict) -> None:  # pragma: no cover
    """Write one draw's decile rows, replacing any rows of the same draw left
    by an interrupted run (deciles are written before the draw checkpoint)."""
    for k, val in reversed(list(key.items())):
        dec.insert(0, k, val)
    if path.exists():
        old = pd.read_csv(path)
        mask = np.ones(len(old), bool)
        for k, val in key.items():
            mask &= old[k].astype(str).to_numpy() == str(val)
        if mask.any():
            old[~mask].to_csv(path, index=False)
    dec.to_csv(path, mode="a", header=not path.exists(), index=False)


def main(stage: str, lf_override: float | None = None) -> None:  # pragma: no cover
    import gc

    from uk_ai_study.checkpoint import Checkpoint
    from uk_ai_study.data import microsimulation
    from uk_ai_study.runner import build_person_table, weighted_deciles
    from uk_ai_study.shocks import PRESETS, TRANSITION_ZEROED_VARIABLES, ShockScenario, apply_shocks, \
        build_shocked_simulation

    OUT.mkdir(parents=True, exist_ok=True)
    baseline = microsimulation()
    persons = build_person_table(baseline, PERIOD)
    b = _sim_metrics(baseline)
    arrays = {v: baseline.calculate(v, period=PERIOD, map_to="person").values.astype(float)
              for v in TRANSITION_ZEROED_VARIABLES}
    arrays["employment_status"] = baseline.calculate(
        "employment_status", period=PERIOD, map_to="person").values.astype(object)
    # baseline-fixed deciles of equivalised HBAI income, households weighted
    # by their number of members (the 20-draw decile profile's basis)
    hh_decile = weighted_deciles(b["equiv"], b["hw"] * b["count"])
    # public-sector pay bill inputs (decision (b)): FRS employment sector and
    # the employer's NI and pension cost of each baseline job
    try:
        public = baseline.calculate("employment_sector", period=PERIOD, map_to="person").values.astype(str) == "PUBLIC"
        employer_costs = (baseline.calculate("ni_employer", period=PERIOD, map_to="person").values.astype(float)
                          + arrays["employer_pension_contributions"])
    except Exception as exc:  # recorded as NaN rather than stopping the run
        print(f"public pay bill unavailable: {exc}", flush=True)
        public = employer_costs = None
    try:  # household family type: the family type of the household's first person
        hid_p = baseline.calculate("household_id", period=PERIOD, map_to="person").values
        hid_h = baseline.calculate("household_id", period=PERIOD, map_to="household").values
        fam_p = baseline.calculate("family_type", period=PERIOD, map_to="person").values.astype(str)
        family = pd.Series(fam_p).groupby(hid_p, sort=False).first().reindex(hid_h).to_numpy().astype(str)
    except Exception as exc:
        print(f"family type unavailable: {exc}", flush=True)
        family = None
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAYS_DIR / "baseline.npz", weight=b["hw"], decile=hh_decile,
                        family=family if family is not None else np.array([]),
                        **{k: b[k] for k in ("hh_market", "hh_net", "hh_it", "hh_ni", "hh_benefits", "hh_uc")})
    lfc = lf_components(arrays["employment_status"], persons["age"], persons["weight"])
    print(f"baseline ready; public employees {0 if public is None else int(public.sum())}; "
          f"family types {None if family is None else sorted(set(family))}; lf {lfc}", flush=True)
    assert 2.0 <= lfc["unemployment_rate_pct"] <= 9.0, f"implausible baseline unemployment rate {lfc}"
    employer_pension = arrays["employer_pension_contributions"]
    del baseline
    gc.collect()

    if stage == "obr":
        if lf_override is None:
            lf = labour_force_m(arrays["employment_status"], persons["age"], persons["weight"])
        else:
            lf = lf_override
        employees_m = float(persons["weight"][persons["employment_income"] > 0].sum() / 1e6)
        rate = obr_displacement_rate(lf, employees_m)
        scen = ShockScenario("obr_benchmark", rate, 0.0, capital_return_increase=0.0)
        ck = Checkpoint(OUT / "obr_benchmark_draws.csv", ("seed",), script=__file__)
        if not ck.done:  # restart the per-draw side files with the draws
            for f in ("obr_benchmark_deciles.csv", "obr_benchmark_decile_family.csv"):
                (OUT / f).unlink(missing_ok=True)
        for seed in SEEDS:
            if ck.is_done(seed):
                continue
            table = obr_table(persons, apply_shocks(persons, scen, seed=seed))
            sim = build_shocked_simulation(None, table, PERIOD, base_arrays=arrays)
            s = _sim_metrics(sim)
            del sim
            gc.collect()
            _draw_outputs(b, s, hh_decile, family, OUT / "obr_benchmark_deciles.csv",
                          OUT / "obr_benchmark_decile_family.csv", {"seed": seed}, f"obr_seed{seed:02d}")
            ck.append({"seed": seed, "displacement_rate": rate,
                       **_row(b, s, persons, table, public, employer_costs, employer_pension)})
            print(f"obr seed {seed} done", flush=True)
        draws = ck.read()
        out = {"displacement_rate": rate, "labour_force_m": lf, "employees_m": employees_m,
               "labour_force_components": lfc, "labour_force_override": lf_override,
               "obr": {"borrowing_bn": OBR_BORROWING_BN, "receipts_bn": OBR_RECEIPTS_BN,
                       "welfare_bn": OBR_WELFARE_BN}, "summary": summarise(draws, DRAW_COLS)}
        (OUT / "obr_benchmark.json").write_text(json.dumps(out, indent=2))
    elif stage == "capital":
        scen = PRESETS["central"]
        ck = Checkpoint(OUT / "capital_variants_draws.csv", ("variant", "seed"), script=__file__)
        if not ck.done:
            for f in ("capital_variants_deciles.csv", "capital_variants_decile_family.csv"):
                (OUT / f).unlink(missing_ok=True)
        ref = pd.read_csv(ROOT / "results" / "robustness" / "dividend_coverage_draws.csv")
        ref = ref[ref["run"] == "central"].set_index("seed")["displaced_weighted_m"]
        fulls: dict = {}
        for v in CAPITAL_VARIANTS:  # variant-major: whole variants finish first
            for seed in SEEDS:
                if ck.is_done(v.name, seed):
                    continue
                if seed not in fulls:
                    fulls[seed] = apply_shocks(persons, scen, seed=seed)
                    got = float(persons["weight"].to_numpy(float)[fulls[seed]["displaced"].to_numpy(bool)].sum() / 1e6)
                    assert np.isclose(got, ref.loc[seed], rtol=1e-9), \
                        f"seed {seed}: displaced {got} != central {ref.loc[seed]}"
                table = capital_variant_table(persons, fulls[seed], v)
                sim = build_shocked_simulation(None, table, PERIOD, base_arrays=arrays)
                s = _sim_metrics(sim)
                del sim
                gc.collect()
                _draw_outputs(b, s, hh_decile, family, OUT / "capital_variants_deciles.csv",
                              OUT / "capital_variants_decile_family.csv", {"variant": v.name, "seed": seed},
                              f"capital_{v.name}_seed{seed:02d}")
                ck.append({"variant": v.name, "seed": seed, "closed_kappa": v.closed_kappa,
                           "closed_wedge_bn": table.attrs.get("closed_wedge_bn", np.nan),
                           **_row(b, s, persons, table, public, employer_costs, employer_pension)})
                print(f"capital {v.name} seed {seed} done", flush=True)
        draws = ck.read()
        out = {v.name: summarise(draws[draws.variant == v.name], DRAW_COLS) for v in CAPITAL_VARIANTS
               if (draws.variant == v.name).any()}
        (OUT / "capital_variants.json").write_text(json.dumps(out, indent=2))
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":  # pragma: no cover
    lf = None
    if "--labour-force-m" in sys.argv:
        lf = float(sys.argv[sys.argv.index("--labour-force-m") + 1])
    main(sys.argv[1], lf)
