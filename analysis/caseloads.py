"""Workstream 5A: benefit caseload and spending responses to AI shocks.

For central / high / low and the four incidence families (seed 0), compute:
newly UC-entitled benefit units / households / persons (universal_credit > 0
shocked vs baseline), change in aggregate UC spending, the UC housing-costs
element, and cheap passported gains (free school meals, council tax benefit)
where the model exposes them.

Outputs results/caseloads/<scenario>.json + summary.csv + caseloads.png.
Context (cited, not computed): DWP total welfare spending is ~£334bn in
2025/26 (DWP benefit expenditure and caseload tables, Spring 2025).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uk_ai_study.runner import build_person_table
from uk_ai_study.shocks import PRESETS, apply_shocks, build_shocked_simulation

from incidence_scenarios import shocked_table_for

DATA = Path("data")
OUT = Path("results/caseloads")
PERIOD = 2026
SEED = 0
ADULT = DATA / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"

PASSPORTED = ("free_school_meals", "council_tax_benefit")


def uc_metrics(sim):
    """UC receipt and spending aggregates at benunit level + entity maps."""
    bw = sim.calculate("benunit_weight", period=PERIOD, map_to="benunit").values
    uc_b = sim.calculate("universal_credit", period=PERIOD, map_to="benunit").values
    out = {
        "uc_benunit": uc_b,
        "benunit_weight": bw,
        "uc_spend": float((uc_b * bw).sum()),
        "uc_person": sim.calculate("universal_credit", period=PERIOD, map_to="person").values,
        "uc_household": sim.calculate("universal_credit", period=PERIOD, map_to="household").values,
    }
    try:
        h = sim.calculate("uc_housing_costs_element", period=PERIOD, map_to="benunit").values
        # condition on actual UC receipt: the element is a formula component
        # computed for every benunit, so the aggregate is the housing element
        # inside paid UC awards. NOTE: uc_housing_costs_element is the GROSS
        # entitlement (pre-taper); outputs are labelled accordingly (R2-10).
        out["uc_housing_spend"] = float((h * bw * (uc_b > 0)).sum())
    except Exception:
        out["uc_housing_spend"] = None
    out["passported"] = {}
    for var in PASSPORTED:
        try:
            v = sim.calculate(var, period=PERIOD, map_to="benunit").values
            out["passported"][var] = float((v * bw).sum())
        except Exception:
            pass
    return out


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds = UKSingleYearDataset(file_path=str(DATA / "frs_2024_25.h5"))
    baseline = Microsimulation(dataset=ds)
    calc = lambda v: baseline.calculate(v, period=PERIOD, map_to="person").values
    persons = build_person_table(baseline, PERIOD, ADULT)

    pw = persons["weight"].to_numpy()
    hw = baseline.calculate("household_weight", period=PERIOD, map_to="household").values

    b = uc_metrics(baseline)

    scenarios = {
        "central": lambda: apply_shocks(persons, PRESETS["central"], seed=SEED),
        "high": lambda: apply_shocks(persons, PRESETS["high"], seed=SEED),
        "low": lambda: apply_shocks(persons, PRESETS["low"], seed=SEED),
        "incidence_exposure": lambda: shocked_table_for("exposure", persons),
        "incidence_junior": lambda: shocked_table_for("junior", persons),
        "incidence_compression": lambda: shocked_table_for("compression", persons),
        "incidence_uniform": lambda: shocked_table_for("uniform", persons),
    }

    summary = []
    for name, make in scenarios.items():
        table = make()
        sim = build_shocked_simulation(ds, baseline, table, PERIOD)
        s = uc_metrics(sim)

        new_bu = (s["uc_benunit"] > 0) & (b["uc_benunit"] <= 0)
        lost_bu = (s["uc_benunit"] <= 0) & (b["uc_benunit"] > 0)
        new_hh = (s["uc_household"] > 0) & (b["uc_household"] <= 0)
        new_p = (s["uc_person"] > 0) & (b["uc_person"] <= 0)

        rec = {
            "scenario": name,
            "displaced_weighted_m": float(pw[table["displaced"].to_numpy()].sum() / 1e6),
            "new_uc_benunits_thousands": float(b["benunit_weight"][new_bu].sum() / 1e3),
            "exiting_uc_benunits_thousands": float(b["benunit_weight"][lost_bu].sum() / 1e3),
            "net_new_uc_benunits_thousands": float(
                (b["benunit_weight"][new_bu].sum() - b["benunit_weight"][lost_bu].sum()) / 1e3
            ),
            "new_uc_households_thousands": float(hw[new_hh].sum() / 1e3),
            "new_uc_persons_thousands": float(pw[new_p].sum() / 1e3),
            "uc_spend_baseline_bn": b["uc_spend"] / 1e9,
            "uc_spend_shocked_bn": s["uc_spend"] / 1e9,
            "uc_spend_change_bn": (s["uc_spend"] - b["uc_spend"]) / 1e9,
            # gross entitlement (pre-taper) within paid UC awards
            "uc_housing_element_gross_entitlement_change_bn": (
                (s["uc_housing_spend"] - b["uc_housing_spend"]) / 1e9
                if s["uc_housing_spend"] is not None and b["uc_housing_spend"] is not None
                else None
            ),
            "passported_change_bn": {
                k: (s["passported"][k] - b["passported"][k]) / 1e9
                for k in s["passported"]
                if k in b["passported"]
            },
            "context_dwp_welfare_spend_2025_26_bn": 334.0,
        }
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2))
        summary.append({k: v for k, v in rec.items() if not isinstance(v, dict)})
        print(name, {k: round(v, 2) for k, v in summary[-1].items() if isinstance(v, float)}, flush=True)
        del sim, s, table

    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    figure(df)


def figure(df):
    import figstyle as fs

    fs.apply_style()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fs.TWOPANEL)
    labels = [s.replace("incidence_", "inc:\n") for s in df["scenario"]]
    x = np.arange(len(df))
    ax1.bar(x, df["net_new_uc_benunits_thousands"], color=fs.BLUE)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Net newly UC-entitled benefit units (thousands)")
    ax1.grid(axis="x", visible=False)
    ax2.bar(x - 0.2, df["uc_spend_change_bn"], width=0.4, color=fs.RED, label="Total UC")
    ax2.bar(x + 0.2, df["uc_housing_element_gross_entitlement_change_bn"], width=0.4,
            color=fs.YELLOW, label="Housing-costs element (gross entitlement)")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Change in annual UC spending (£bn/yr)")
    ax2.grid(axis="x", visible=False)
    fs.legend_below(ax2, 2)
    fig.suptitle("Universal Credit caseload and spending response, 2026", fontsize=11)
    fs.save(fig, OUT / "caseloads.png")


if __name__ == "__main__":
    main()
