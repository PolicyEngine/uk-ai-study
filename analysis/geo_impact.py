"""Workstream 4: constituency-level geographic impact of the central AI shock.

Route B (documented): the constituency weight matrix
(parliamentary_constituency_weights.h5, key "2025", 650x53508) indexes the
households of enhanced_frs_2023_24.h5 (53,508 households; verified against
policyengine_uk_data/datasets/local_areas/constituencies/calibrate.py, whose
row order is constituencies_2024.csv — 2024 boundaries, 650 seats, with the
PolicyEngine hex-cartogram x,y layout). Our SOC join
(person_id = SERNUM*1000 + PERSON against the 2024-25 UKDA adult.tab) is
only valid for the plain FRS 2024-25: on the enhanced 2023-24 dataset the ID
overlap is spurious (23.7% of adults, different survey year). We therefore
IMPUTE SOC major group for enhanced-FRS employees by drawing from the
plain-FRS weighted SOC distribution within (age band x gender x region x
employee earnings decile) cells, with a documented fallback ladder for empty
cells. Match rates are written to results/geo/imputation_notes.json.

Central scenario (7% displacement, +2.6% wages, +0.4pp capital return),
seed 0, period 2025 (the weights' calibration year).

Outputs (results/geo/):
  constituency_impacts.csv   code, name, region, metrics
  region_summary.csv         ITL1-style region aggregates
  hexmap_income_change.png   diverging cartogram, mean HBAI income change %
  hexmap_displacement.png    sequential cartogram, displaced per 1,000 workers
  imputation_notes.json      match rates + method
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import figstyle  # noqa: E402
from uk_ai_study.exposure import attach_soc_major_group, exposure_for_major_group  # noqa: E402
from uk_ai_study.shocks import PRESETS, apply_shocks, build_shocked_simulation  # noqa: E402

import policyengine_uk_data  # noqa: E402

STORAGE = Path(policyengine_uk_data.__file__).parent / "storage"
PLAIN_FRS = ROOT / "data" / "frs_2024_25.h5"
ENHANCED_FRS = STORAGE / "enhanced_frs_2023_24.h5"
WEIGHTS_H5 = STORAGE / "parliamentary_constituency_weights.h5"
CONSTITUENCIES = STORAGE / "constituencies_2024.csv"
ADULT_TAB = ROOT / "data" / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab"
OUT = ROOT / "results" / "geo"
OUT.mkdir(parents=True, exist_ok=True)

PERIOD = 2025  # the weights' calibration year (h5 key "2025")
SEED = 0
SCENARIO = PRESETS["central"]
AGE_BANDS = ((16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200))


def age_band(age: np.ndarray) -> np.ndarray:
    out = np.full(len(age), -1)
    for i, (lo, hi) in enumerate(AGE_BANDS):
        out[(age >= lo) & (age <= hi)] = i
    return out


def person_frame(sim, period):
    cols = {
        "person_id": "person_id",
        "age": "age",
        "gender": "gender",
        "employment_income": "employment_income",
        "savings_interest_income": "savings_interest_income",
        "dividend_income": "dividend_income",
    }
    df = pd.DataFrame(
        {k: sim.calculate(v, period=period, map_to="person").values for k, v in cols.items()}
    )
    df["weight"] = sim.calculate("person_weight", period=period, map_to="person").values
    df["region"] = sim.calculate("region", period=period, map_to="person").values
    return df


def weighted_decile_edges(values, weights, n=10):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    cw = cw / cw[-1]
    return np.interp(np.arange(1, n) / n, cw, values[order])


def main():
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    notes = {"method": "route B: SOC imputation on enhanced FRS 2023-24", "period": PERIOD,
             "scenario": "central", "seed": SEED}

    # ---- Step 1: plain FRS donor SOC distribution -------------------------
    plain = Microsimulation(dataset=UKSingleYearDataset(file_path=str(PLAIN_FRS)))
    donor = person_frame(plain, PERIOD)
    donor["soc"] = attach_soc_major_group(donor["person_id"], ADULT_TAB)
    demp = donor[donor["employment_income"] > 0].copy()
    notes["plain_frs_employees"] = int(len(demp))
    notes["plain_frs_employee_soc_observed_share"] = float(
        np.average(np.isfinite(demp["soc"]), weights=demp["weight"])
    )
    demp = demp[np.isfinite(demp["soc"])]
    edges = weighted_decile_edges(
        demp["employment_income"].to_numpy(float), demp["weight"].to_numpy(float)
    )
    demp["band"] = age_band(demp["age"].to_numpy())
    demp["dec"] = np.digitize(demp["employment_income"], edges)

    def dist(frame):
        g = frame.groupby("soc")["weight"].sum()
        return g.index.to_numpy(float), np.cumsum((g / g.sum()).to_numpy(float))

    full = {k: dist(g) for k, g in demp.groupby(["band", "gender", "region", "dec"])}
    fb1 = {k: dist(g) for k, g in demp.groupby(["band", "gender", "dec"])}
    fb2 = {k: dist(g) for k, g in demp.groupby(["band", "gender"])}
    marginal = dist(demp)
    del plain, donor

    # ---- Step 2: enhanced FRS + imputation ---------------------------------
    dataset = UKSingleYearDataset(file_path=str(ENHANCED_FRS))
    baseline = Microsimulation(dataset=dataset)
    persons = person_frame(baseline, PERIOD)
    emp = persons["employment_income"].to_numpy(float) > 0
    band = age_band(persons["age"].to_numpy())
    dec = np.digitize(persons["employment_income"], edges)
    gender = persons["gender"].to_numpy()
    region = persons["region"].to_numpy()

    rng = np.random.default_rng(SEED)
    soc = np.full(len(persons), np.nan)
    tier_counts = {"full_cell": 0.0, "age_gender_decile": 0.0, "age_gender": 0.0, "marginal": 0.0}
    w = persons["weight"].to_numpy(float)
    for i in np.flatnonzero(emp):
        key4 = (band[i], gender[i], region[i], dec[i])
        key3 = (band[i], gender[i], dec[i])
        key2 = (band[i], gender[i])
        if key4 in full:
            codes, p = full[key4]; tier_counts["full_cell"] += w[i]
        elif key3 in fb1:
            codes, p = fb1[key3]; tier_counts["age_gender_decile"] += w[i]
        elif key2 in fb2:
            codes, p = fb2[key2]; tier_counts["age_gender"] += w[i]
        else:
            codes, p = marginal; tier_counts["marginal"] += w[i]
        soc[i] = codes[min(np.searchsorted(p, rng.random()), len(codes) - 1)]
    tot = sum(tier_counts.values())
    notes["imputation_tier_shares_weighted"] = {k: v / tot for k, v in tier_counts.items()}
    notes["enhanced_frs_employees"] = int(emp.sum())

    persons["soc_major_group"] = soc
    exposure = exposure_for_major_group(persons["soc_major_group"], "c_aioe")
    theta = exposure_for_major_group(persons["soc_major_group"], "complementarity_theta")
    persons["exposure"] = np.where(np.isfinite(exposure), exposure, np.nanmean(exposure))
    persons["complementarity"] = np.where(np.isfinite(theta), theta, np.nanmean(theta))

    shocked_table = apply_shocks(persons, SCENARIO, seed=SEED)
    shocked = build_shocked_simulation(dataset, baseline, shocked_table, PERIOD)
    displaced = shocked_table["displaced"].to_numpy()

    # ---- Step 3: household-level vectors -----------------------------------
    hh_income_base = baseline.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values
    hh_income_shock = shocked.calculate("hbai_household_net_income", period=PERIOD, map_to="household").values
    hh_people = baseline.calculate("household_count_people", period=PERIOD, map_to="household").values

    # person -> household aggregation for poverty headcounts / workers / displaced
    hh_id = baseline.calculate("household_id", period=PERIOD, map_to="household").values
    person_hh = baseline.calculate("household_id", period=PERIOD, map_to="person").values
    pos = pd.Series(np.arange(len(hh_id)), index=hh_id)
    pidx = pos.loc[person_hh].to_numpy()
    hh_workers = np.zeros(len(hh_id))
    np.add.at(hh_workers, pidx, emp.astype(float))
    hh_displaced = np.zeros(len(hh_id))
    np.add.at(hh_displaced, pidx, displaced.astype(float))
    pov_p_base = baseline.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values.astype(float)
    pov_p_shock = shocked.calculate("in_poverty_bhc", period=PERIOD, map_to="person").values.astype(float)
    hh_pov_base = np.zeros(len(hh_id))
    np.add.at(hh_pov_base, pidx, pov_p_base)
    hh_pov_shock = np.zeros(len(hh_id))
    np.add.at(hh_pov_shock, pidx, pov_p_shock)

    del shocked, baseline

    # ---- Step 4: constituency estimates ------------------------------------
    with h5py.File(WEIGHTS_H5) as f:
        W = f["2025"][:]  # 650 x n_households, grossing weights
    assert W.shape[1] == len(hh_id), (W.shape, len(hh_id))
    const = pd.read_csv(CONSTITUENCIES)
    assert len(const) == W.shape[0]

    inc_base = W @ hh_income_base
    inc_delta = W @ (hh_income_shock - hh_income_base)
    workers = W @ hh_workers
    disp = W @ hh_displaced
    pov_delta = W @ (hh_pov_shock - hh_pov_base)
    pov_base = W @ hh_pov_base
    people = W @ hh_people

    df = const[["code", "name", "region", "x", "y"]].copy()
    df["income_change_pct"] = 100 * inc_delta / inc_base
    df["displaced_per_1000_workers"] = 1000 * disp / workers
    df["poverty_headcount_change"] = pov_delta
    df["poverty_rate_change_pp"] = 100 * pov_delta / people
    df["baseline_poverty_headcount"] = pov_base
    df["workers"] = workers
    df["displaced"] = disp
    df.drop(columns=["x", "y"]).to_csv(OUT / "constituency_impacts.csv", index=False)

    # aggregate from numerators/denominators (not means of ratios)
    agg = pd.DataFrame({
        "region": df["region"], "inc_base": inc_base, "inc_delta": inc_delta,
        "workers": workers, "disp": disp, "pov_delta": pov_delta, "people": people,
    }).groupby("region").sum()
    reg = pd.DataFrame({
        "income_change_pct": 100 * agg["inc_delta"] / agg["inc_base"],
        "displaced_per_1000_workers": 1000 * agg["disp"] / agg["workers"],
        "poverty_headcount_change": agg["pov_delta"],
        "poverty_rate_change_pp": 100 * agg["pov_delta"] / agg["people"],
    })
    reg.to_csv(OUT / "region_summary.csv")

    # ---- Step 5: hex cartograms --------------------------------------------
    hexmap(df, "income_change_pct", "Mean HBAI household income change (%)",
           OUT / "hexmap_income_change.png", diverging=True)
    hexmap(df, "displaced_per_1000_workers", "Displaced workers per 1,000 workers",
           OUT / "hexmap_displacement.png", diverging=False)

    (OUT / "imputation_notes.json").write_text(json.dumps(notes, indent=2))

    top = df.nlargest(10, "income_change_pct")[["code", "name", "income_change_pct"]]
    bot = df.nsmallest(10, "income_change_pct")[["code", "name", "income_change_pct"]]
    print(json.dumps({
        "notes": notes,
        "national_income_change_pct": float(100 * inc_delta.sum() / inc_base.sum()),
        "national_displaced_per_1000": float(1000 * disp.sum() / workers.sum()),
        "national_poverty_headcount_change": float(pov_delta.sum()),
        "top10_income_change": top.to_dict("records"),
        "bottom10_income_change": bot.to_dict("records"),
    }, indent=2))


def hexmap(df, col, title, path, diverging):
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap

    figstyle.apply_style()
    vals = df[col].to_numpy()
    if diverging:
        # Robustify the scale: a handful of extreme constituencies (NI seats,
        # inner London) otherwise crush the ~600 English hexes into a single
        # pale tone, making the map read as flat/random. Clip the symmetric
        # limit to the 95th percentile of |value| and flag clipping with
        # extended colourbar caps.
        vmax = float(np.nanpercentile(np.abs(vals), 95)) or float(np.abs(vals).max())
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        cmap = figstyle.DIVERGING
    else:
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        cmap = LinearSegmentedColormap.from_list(
            "pe_seq", [figstyle.BLUE_98, figstyle.BLUE_LIGHT, figstyle.BLUE, figstyle.BLUE_PRESSED]
        )
    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    dy = np.sqrt(3) / 2
    for _, r in df.iterrows():
        cx = r["x"] + 0.5 * (int(r["y"]) % 2 != 0)
        cy = r["y"] * dy
        ax.add_patch(RegularPolygon(
            (cx, cy), numVertices=6, radius=1 / np.sqrt(3) * 0.98,
            orientation=0, facecolor=cmap(norm(r[col])), edgecolor="white", linewidth=0.3,
        ))
    ax.set_xlim(df["x"].min() - 1.5, df["x"].max() + 1.5)
    ax.set_ylim(df["y"].min() * dy - 1.5, df["y"].max() * dy + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title + "\nCentral scenario (7% displacement, +2.6% wages), 2025, seed 0")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
        extend="both" if diverging else "neither",
    )
    cbar.outline.set_visible(False)
    figstyle.save(fig, path)


if __name__ == "__main__":
    main()
