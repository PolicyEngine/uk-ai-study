"""Gender dimension: displacement incidence and income effects by sex.

Writes results/appendix/gender_incidence.json (paper §5.6). Recovered from
the original generating session — see uk-ai-study#1, finding 8.

Usage: python analysis/gender.py [--data-dir DATA] [--period 2026]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from policyengine_uk import Microsimulation
from policyengine_uk.data import UKSingleYearDataset

from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group
from uk_ai_study.shocks import PRESETS, apply_shocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--draws", type=int, default=20)
    args = parser.parse_args()
    data = Path(args.data_dir)
    P = args.period

    ds = UKSingleYearDataset(file_path=str(data / "frs_2024_25.h5"))
    base = Microsimulation(dataset=ds)
    calc = lambda v: base.calculate(v, period=P, map_to="person").values
    persons = pd.DataFrame(
        {
            "person_id": calc("person_id"),
            "age": calc("age"),
            "employment_income": calc("employment_income"),
            "savings_interest_income": calc("savings_interest_income"),
            "dividend_income": calc("dividend_income"),
            "weight": calc("person_weight"),
            "gender": calc("gender"),
        }
    )
    adult_tab = data / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
    persons["soc_major_group"] = attach_soc_major_group(persons["person_id"], adult_tab)
    e = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    th = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(e), e, np.nanmean(e))
    persons["complementarity"] = np.where(np.isfinite(th), th, np.nanmean(th))

    w = persons["weight"].to_numpy()
    emp = persons["employment_income"].to_numpy() > 0
    female = np.char.upper(persons["gender"].to_numpy().astype(str)) == "FEMALE"
    matched = emp & np.isfinite(e)

    res = {}
    for label, m in [("female", matched & female), ("male", matched & ~female)]:
        res[f"{label}_mean_exposure"] = float(np.average(persons["exposure"][m], weights=w[m]))

    shares_f, rates_f, rates_m = [], [], []
    for s in range(args.draws):
        d = apply_shocks(persons, PRESETS["central"], seed=s)["displaced"].to_numpy()
        shares_f.append(w[d & female].sum() / w[d].sum())
        rates_f.append(w[d & female].sum() / w[emp & female].sum())
        rates_m.append(w[d & ~female].sum() / w[emp & ~female].sum())
    res["female_share_of_displaced_mean"] = float(np.mean(shares_f))
    res["female_displacement_rate_mean"] = float(np.mean(rates_f))
    res["male_displacement_rate_mean"] = float(np.mean(rates_m))
    res["female_share_of_employment"] = float(w[emp & female].sum() / w[emp].sum())

    st = apply_shocks(persons, PRESETS["central"], seed=0)
    sim = Microsimulation(dataset=ds)
    for col in ("employment_income", "savings_interest_income", "dividend_income"):
        sim.set_input(col, P, st[col].to_numpy(dtype=float))
    hni_b = base.calculate("household_net_income", period=P, map_to="person").values
    hni_s = sim.calculate("household_net_income", period=P, map_to="person").values
    for label, m in [("female", female), ("male", ~female)]:
        res[f"{label}_income_change_pct"] = float(
            100 * ((hni_s - hni_b)[m] * w[m]).sum() / (hni_b[m] * w[m]).sum()
        )

    out = Path("results/appendix/gender_incidence.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
