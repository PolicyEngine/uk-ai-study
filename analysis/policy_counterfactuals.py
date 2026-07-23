"""Workstream 1 — AI-contingent policy counterfactuals.

Reforms are applied ONLY in the shocked world: the estimand for each reform R
and shock scenario S is  M(S + R) - M(S), i.e. what the reform changes about
the shock's damage, holding the shock draw (seed 0) fixed. The unshocked
baseline enters only to define deciles, lost earnings and poverty lines'
validation; no reform is ever evaluated against the no-shock world.

Reforms
  R1 wage insurance — displaced workers receive 50% of their lost (baseline)
     employment income for the year, capped at £15,000. Implemented as a
     POST-SIMULATION transfer added to household HBAI disposable income
     (BHC and AHC alike; housing costs unchanged). Choice documented: the
     transfer is treated as non-taxable and disregarded by means tests, so
     the gross cost equals the net exchequer cost and there is no UC
     clawback. This brackets the generous end; an in-model taxable variant
     would cost less gross but deliver less net. Poverty is recomputed by
     comparing the augmented household income to the model's own
     poverty_line_bhc / poverty_line_ahc; the reconstruction is validated
     against the model's in_poverty_* flags in the no-reform shocked world
     (agreement recorded in the JSON).
  R2 UC circuit breaker — 20% uplift to all four UC standard-allowance
     rates (COVID-style), via PolicyEngine parameter reform, applied to the
     shocked simulation for the shock year only.
  R3 benefit-cap suspension + UC taper cut — the benefit cap is disapplied
     (thresholds x1000) and the UC taper falls 55% -> 45% for the shock
     year. ("Taper freeze" is not a well-defined 2.89.2 parameter change —
     the taper is a rate, not an uprated amount — so the combo implemented
     is suspension of the cap plus a taper cut, the standard automatic-
     stabiliser strengthening; documented per task instruction.)

Scenario families (incidence_scenarios.py machinery, central aggregate
shock): the headline reform R1 runs under exposure (central), junior and
uniform incidence; R2/R3 under the central/exposure family.

Outputs: results/policy/*.json, results/policy/summary.csv,
results/policy/policy_reforms.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from uk_ai_study.runner import build_person_table, gini
from uk_ai_study.shocks import (
    SHOCKED_INCOME_VARIABLES,
    TRANSITION_ZEROED_VARIABLES,
    build_shocked_simulation,
)
from incidence_scenarios import shocked_table_for  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "policy"
PERIOD = 2026
SEED = 0
YEAR_SPAN = f"{PERIOD}-01-01.{PERIOD}-12-31"

WAGE_INSURANCE_RATE = 0.5
WAGE_INSURANCE_CAP = 15_000.0
UC_UPLIFT = 1.20
TAPER_CUT_TO = 0.45
CAP_MULTIPLIER = 1000.0  # de-facto suspension

FAMILIES_R1 = ("exposure", "junior", "uniform")

def household_decile_benefit_shares(
    household_gain, household_weight, household_decile
) -> dict[int, float]:
    """Strict reform-pound shares, counting each household exactly once."""
    gain = np.asarray(household_gain, dtype=float)
    weight = np.asarray(household_weight, dtype=float)
    decile = np.asarray(household_decile, dtype=int)
    if not (gain.shape == weight.shape == decile.shape):
        raise ValueError("household gain, weight and decile shapes must match")
    total = float((gain * weight).sum())
    if total <= 0:
        return {d: 0.0 for d in range(1, 11)}
    return {
        d: float(100 * (gain[decile == d] * weight[decile == d]).sum() / total)
        for d in range(1, 11)
    }


def person_calc(sim, var):
    return sim.calculate(var, period=PERIOD, map_to="person").values


def hh_calc(sim, var):
    return sim.calculate(var, period=PERIOD, map_to="household").values


def build_sim(dataset, base_arrays, table, reform=None):
    """Shared fail-fast constructor, with an optional parameter reform (R2-7)."""
    return build_shocked_simulation(
        dataset, None, table, PERIOD, reform=reform, base_arrays=base_arrays
    )


def hh_state(sim):
    """Household-level state needed for metrics and post-hoc R1."""
    return {
        "gov": float((hh_calc(sim, "gov_balance") * hh_calc(sim, "household_weight")).sum()),
        "hw": hh_calc(sim, "household_weight"),
        "n": hh_calc(sim, "household_count_people"),
        "hbai_bhc": hh_calc(sim, "hbai_household_net_income"),
        "hbai_ahc": hh_calc(sim, "hbai_household_net_income_ahc"),
        "eq_bhc": hh_calc(sim, "equiv_hbai_household_net_income"),
        "eq_ahc": hh_calc(sim, "equiv_hbai_household_net_income_ahc"),
        "line_bhc": hh_calc(sim, "poverty_line_bhc"),
        "line_ahc": hh_calc(sim, "poverty_line_ahc"),
        "equiv_factor_bhc": hh_calc(sim, "household_equivalisation_bhc"),
        "equiv_factor_ahc": hh_calc(sim, "household_equivalisation_ahc"),
        "pov_bhc_model": hh_calc(sim, "in_poverty_bhc"),
        "pov_ahc_model": hh_calc(sim, "in_poverty_ahc"),
        # NOTE: person-broadcast household TOTAL (not per-capita) — currently
        # unused downstream; divide by household size before any per-person
        # cash aggregation (issue #6)
        "hni_person": person_calc(sim, "hbai_household_net_income"),
    }


def poverty_rule(state, hbai_key, line_key):
    """Model rule: absolute HBAI income below the household poverty line."""
    return state[hbai_key] < state[line_key]


def metrics_from_state(state, pw_by_hh, extra_hh_income=None):
    """Population metrics, optionally with a post-hoc household transfer."""
    bhc = state["hbai_bhc"] + (extra_hh_income if extra_hh_income is not None else 0.0)
    ahc = state["hbai_ahc"] + (extra_hh_income if extra_hh_income is not None else 0.0)
    pov_bhc = bhc < state["line_bhc"]
    pov_ahc = ahc < state["line_ahc"]
    tot_pw = pw_by_hh.sum()
    eq_bhc = bhc / state["equiv_factor_bhc"]
    return {
        "poverty_bhc": float((pov_bhc * pw_by_hh).sum() / tot_pw),
        "poverty_ahc": float((pov_ahc * pw_by_hh).sum() / tot_pw),
        "gini": gini(eq_bhc, state["hw"] * state["n"]),
    }


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk.system import system

    OUT.mkdir(parents=True, exist_ok=True)
    ds = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    baseline = Microsimulation(dataset=ds)

    persons = build_person_table(
        baseline,
        PERIOD,
        DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab",
    )

    w = persons["weight"].to_numpy()
    base_emp = persons["employment_income"].to_numpy(dtype=float)

    # baseline arrays needed to rebuild shocked simulations without keeping
    # the baseline Microsimulation alive (memory discipline)
    base_arrays = {
        var: person_calc(baseline, var).astype(float) for var in TRANSITION_ZEROED_VARIABLES
    }
    base_arrays["employment_status"] = person_calc(baseline, "employment_status").astype(object)

    # person -> household mapping and baseline deciles (JR16 convention)
    person_hh = person_calc(baseline, "household_id")
    hh_ids = hh_calc(baseline, "household_id")
    hh_index = pd.Series(np.arange(len(hh_ids)), index=hh_ids)
    p2h = hh_index.loc[person_hh].to_numpy()
    pw_by_hh = np.bincount(p2h, weights=w, minlength=len(hh_ids))
    hh_weight = hh_calc(baseline, "household_weight")

    equiv0 = person_calc(baseline, "equiv_hbai_household_net_income")
    order = np.argsort(equiv0)
    cw = np.cumsum(w[order])
    ranks = np.empty(len(equiv0))
    ranks[order] = cw / cw[-1]
    dec = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)
    hh_dec = np.zeros(len(hh_ids), dtype=int)
    hh_dec[p2h] = dec

    del baseline

    # UC standard-allowance reform values from the live parameter tree
    sa = system.parameters.gov.dwp.universal_credit.standard_allowance.amount
    r2_reform = {
        f"gov.dwp.universal_credit.standard_allowance.amount.{k}": {
            YEAR_SPAN: round(float(sa.children[k](f"{PERIOD}-01-01")) * UC_UPLIFT, 2)
        }
        for k in ("SINGLE_YOUNG", "SINGLE_OLD", "COUPLE_YOUNG", "COUPLE_OLD")
    }
    bc = system.parameters.gov.dwp.benefit_cap
    r3_reform = {
        "gov.dwp.universal_credit.means_test.reduction_rate": {YEAR_SPAN: TAPER_CUT_TO},
        **{
            f"gov.dwp.benefit_cap.{grp}.{loc}": {
                YEAR_SPAN: float(bc.children[grp].children[loc](f"{PERIOD}-01-01"))
                * CAP_MULTIPLIER
            }
            for grp in ("single", "non_single")
            for loc in ("in_london", "outside_london")
        },
    }

    results = []

    def decile_shares(hh_gain):
        """Strict pound shares: count each household gain exactly once."""
        return household_decile_benefit_shares(hh_gain, hh_weight, hh_dec)

    def record(name, family, gross_cost, s0, m0, m1, gain_hh, extra=None):
        gain_hh = np.asarray(gain_hh, dtype=float)
        gain_person = gain_hh[p2h]
        pov_bhc_pp = 100 * (m1["poverty_bhc"] - m0["poverty_bhc"])
        pov_ahc_pp = 100 * (m1["poverty_ahc"] - m0["poverty_ahc"])
        rec = {
            "reform": name,
            "family": family,
            "gross_cost_bn": gross_cost / 1e9,
            "poverty_change_bhc_pp": pov_bhc_pp,
            "poverty_change_ahc_pp": pov_ahc_pp,
            "gini_change_pp": 100 * (m1["gini"] - m0["gini"]),
            "cost_per_pp_bhc_bn": (gross_cost / 1e9 / -pov_bhc_pp) if pov_bhc_pp < 0 else None,
            "cost_per_pp_ahc_bn": (gross_cost / 1e9 / -pov_ahc_pp) if pov_ahc_pp < 0 else None,
            "decile_benefit_share_pct": decile_shares(gain_hh),
            "decile_mean_gain_gbp": {
                int(d): float(np.average(gain_person[dec == d], weights=w[dec == d]))
                for d in range(1, 11)
            },
            "shocked_no_reform": {k: m0[k] for k in ("poverty_bhc", "poverty_ahc", "gini")},
            "shocked_with_reform": {k: m1[k] for k in ("poverty_bhc", "poverty_ahc", "gini")},
        }
        if extra:
            rec.update(extra)
        (OUT / f"{name}_{family}.json").write_text(json.dumps(rec, indent=2))
        results.append(rec)
        print(
            name, family,
            f"cost £{rec['gross_cost_bn']:.2f}bn",
            f"dpov_ahc {pov_ahc_pp:+.3f}pp",
            f"dgini {rec['gini_change_pp']:+.3f}pp",
            flush=True,
        )

    for family in FAMILIES_R1:
        table = shocked_table_for(family, persons)
        displaced = table["displaced"].to_numpy()
        sim0 = build_sim(ds, base_arrays, table)
        s0 = hh_state(sim0)
        del sim0
        m0 = metrics_from_state(s0, pw_by_hh)

        # validation: post-hoc poverty rule reproduces the model's flags
        agree_bhc = float(np.mean((s0["hbai_bhc"] < s0["line_bhc"]) == s0["pov_bhc_model"]))
        agree_ahc = float(np.mean((s0["hbai_ahc"] < s0["line_ahc"]) == s0["pov_ahc_model"]))

        # R1 wage insurance (post-simulation transfer, disregarded)
        transfer_p = np.where(
            displaced, np.minimum(WAGE_INSURANCE_RATE * base_emp, WAGE_INSURANCE_CAP), 0.0
        )
        gross_r1 = float((transfer_p * w).sum())
        transfer_hh = np.bincount(p2h, weights=transfer_p, minlength=len(s0["hw"]))
        m1 = metrics_from_state(s0, pw_by_hh, extra_hh_income=transfer_hh)
        gain_person = transfer_hh[p2h]
        record(
            "R1_wage_insurance", family, gross_r1, s0, m0, m1, transfer_hh,
            extra={
                "implementation": "post-simulation transfer, non-taxable, means-test disregarded",
                "poverty_rule_agreement_bhc": agree_bhc,
                "poverty_rule_agreement_ahc": agree_ahc,
                "displaced_weighted_m": float(w[displaced].sum() / 1e6),
                "mean_transfer_per_displaced_gbp": gross_r1 / float(w[displaced].sum()),
            },
        )

        if family == "exposure":
            for name, reform in (("R2_uc_circuit_breaker", r2_reform),
                                 ("R3_cap_suspension_taper_cut", r3_reform)):
                simr = build_sim(ds, base_arrays, table, reform=reform)
                sr = hh_state(simr)
                del simr
                mr = metrics_from_state(sr, pw_by_hh)
                gross = s0["gov"] - sr["gov"]  # net-of-clawback exchequer cost
                gain_hh = np.clip(sr["hbai_bhc"] - s0["hbai_bhc"], 0, None)
                record(name, family, gross, s0, m0, mr, gain_hh,
                       extra={"implementation": "PolicyEngine parameter reform on shocked sim",
                              "reform_parameters": reform})
        del s0

    flat = []
    for r in results:
        flat.append({k: v for k, v in r.items() if not isinstance(v, dict)})
    pd.DataFrame(flat).to_csv(OUT / "summary.csv", index=False)

    # ---- figure: decile incidence + cost per point of poverty averted ----
    import figstyle as fs

    fs.apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=fs.TWOPANEL)
    ax = axes[0]
    central = [r for r in results if r["family"] == "exposure"]
    colors = {"R1_wage_insurance": fs.BLUE, "R2_uc_circuit_breaker": fs.AQUA,
              "R3_cap_suspension_taper_cut": fs.YELLOW}
    labels = {"R1_wage_insurance": "R1 wage insurance",
              "R2_uc_circuit_breaker": "R2 UC +20% standard allowance",
              "R3_cap_suspension_taper_cut": "R3 cap suspension + taper 45%"}
    width = 0.27
    for i, r in enumerate(central):
        shares = r["decile_benefit_share_pct"]
        ax.bar(np.arange(1, 11) + (i - 1) * width, [shares[d] for d in range(1, 11)],
               width=width, color=colors[r["reform"]], label=labels[r["reform"]])
    fs.decile_ax(ax, "Share of reform benefit (%)")
    ax.set_title("Who receives the reform (central shock)")
    shared_handles, shared_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    bars, names, cols = [], [], []
    for r in results:
        v = r["cost_per_pp_ahc_bn"]
        if v is None:
            continue
        tag = labels[r["reform"]].split()[0]
        names.append(f"{tag}\n{r['family']}")
        bars.append(v)
        cols.append(colors[r["reform"]])
    # order bars by cost-effectiveness (cheapest £/pp first) so it reads as a ranking
    order = sorted(range(len(bars)), key=lambda i: bars[i])
    bars = [bars[i] for i in order]
    names = [names[i] for i in order]
    cols = [cols[i] for i in order]
    ax.bar(range(len(bars)), bars, color=cols)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(names)
    ax.set_ylabel("£bn per pp of AHC poverty averted")
    ax.set_title("Cost-effectiveness (reform vs no-reform shocked world)")
    ax.grid(axis="x", visible=False)
    # One shared legend box serving both panels.
    fig.legend(shared_handles, shared_labels, loc="lower center",
               ncol=len(shared_labels), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT / "policy_reforms.png", dpi=fs.DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
