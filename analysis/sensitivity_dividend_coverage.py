"""Capital channel and dividend coverage: channel split and SPI-scaled rerun.

The capital shock scales savings interest and dividend income by
1.405/1.005, so its fiscal size depends on how much dividend income the
microdata carry at the top. Microcosm UK is calibrated to HMRC Survey of
Personal Incomes (SPI) dividend totals by income band, but its 2026 total
runs above the SPI projection for 2026. This script

  (a) splits the central seed-0 shock into its three channels, each run on
      its own with the other two held at baseline (displacement only,
      survivor wages only, capital only), and
  (b) reruns the central exposure-proportional shock over 20 paired draws
      (seeds 0-19) with every person's dividend_income scaled by
      SPI_2026 / Microcosm_2026 in BOTH the baseline and the shocked runs,
      alongside the unscaled run on the same seeds.

SPI benchmark: policyengine-uk-data storage/incomes_projection.csv (commit
fa4ebe9, "regenerate SPI projections from the 2023 outturn"): HMRC SPI
2023-24 outturn projected to 2026, dividend_income_amount summed over all
total-income bands from £12,570 (row 12570-inf), £79.97bn. This is the
projection Microcosm's own SPI calibration targets come from.

Outputs results/robustness/dividend_coverage_draws.csv (checkpointed draws)
and results/robustness/dividend_coverage.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uk_ai_study.data import bundle_provenance, microsimulation  # noqa: E402
from uk_ai_study.runner import build_person_table, gini  # noqa: E402
from uk_ai_study.shocks import (  # noqa: E402
    PRESETS,
    TRANSITION_ZEROED_VARIABLES,
    apply_shocks,
    build_shocked_simulation,
)

OUT = ROOT / "results" / "robustness"
CSV = OUT / "dividend_coverage_draws.csv"
PERIOD = 2026
N_DRAWS = 20
SPI_DIVIDENDS_2026 = 79_969_122_354.0
SPI_SOURCE = (
    "policyengine-uk-data policyengine_uk_data/storage/incomes_projection.csv "
    "@ fa4ebe9: HMRC SPI 2023-24 outturn projected to 2026, "
    "dividend_income_amount, total income band 12570-inf"
)


def person(sim, var):
    return sim.calculate(var, period=PERIOD, map_to="person").values


def household(sim, var):
    return sim.calculate(var, period=PERIOD, map_to="household").values


def metrics(sim) -> dict:
    pw = person(sim, "person_weight")
    hw = household(sim, "household_weight")
    return {
        "gov": float((household(sim, "gov_balance") * hw).sum()),
        "pov_bhc": float(np.average(person(sim, "in_poverty_bhc"), weights=pw)),
        "pov_ahc": float(np.average(person(sim, "in_poverty_ahc"), weights=pw)),
        "gini": gini(
            household(sim, "equiv_hbai_household_net_income"),
            hw * household(sim, "household_count_people"),
        ),
        "dividend_tax": float((person(sim, "dividend_income_tax") * pw).sum()),
    }


def deltas(base: dict, shocked: dict) -> dict:
    return {
        "exchequer_cost_bn": (base["gov"] - shocked["gov"]) / 1e9,
        "poverty_change_bhc_pp": 100 * (shocked["pov_bhc"] - base["pov_bhc"]),
        "poverty_change_ahc_pp": 100 * (shocked["pov_ahc"] - base["pov_ahc"]),
        "gini_change_pp": 100 * (shocked["gini"] - base["gini"]),
        "dividend_tax_change_bn": (shocked["dividend_tax"] - base["dividend_tax"]) / 1e9,
    }


def base_arrays_for(sim) -> dict:
    arrays = {v: person(sim, v).astype(float) for v in TRANSITION_ZEROED_VARIABLES}
    arrays["employment_status"] = person(sim, "employment_status").astype(object)
    return arrays


def channel_table(persons: pd.DataFrame, full: pd.DataFrame, channel: str) -> pd.DataFrame:
    """One channel of the central shock on its own; the others at baseline."""
    t = persons.copy()
    disp = full["displaced"].to_numpy()
    earn = persons["employment_income"].to_numpy(float)
    t["displaced"] = disp if channel == "displacement" else np.zeros(len(t), dtype=bool)
    if channel == "displacement":
        t["employment_income"] = np.where(disp, 0.0, earn)
    elif channel == "wage":
        t["employment_income"] = np.where(disp, earn, full["employment_income"].to_numpy(float))
    elif channel == "capital":
        for col in ("savings_interest_income", "dividend_income"):
            t[col] = full[col].to_numpy(float)
    else:
        raise ValueError(channel)
    return t


def summarise(df: pd.DataFrame, cols) -> dict:
    return {
        c: {
            "mean": float(df[c].mean()),
            "sd": float(df[c].std(ddof=1)),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
        }
        for c in cols
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenario = PRESETS["central"]

    # ---- unscaled baseline -------------------------------------------------
    baseline = microsimulation()
    provenance = bundle_provenance(baseline)
    persons = build_person_table(baseline, PERIOD)
    w = persons["weight"].to_numpy(float)
    dividends = persons["dividend_income"].to_numpy(float)
    microcosm_total = float((dividends * w).sum())
    holders_m = float(w[dividends > 0].sum() / 1e6)
    factor = SPI_DIVIDENDS_2026 / microcosm_total
    b = metrics(baseline)
    arrays = base_arrays_for(baseline)
    del baseline

    done = set()
    if CSV.exists():
        prev = pd.read_csv(CSV)
        done = set(zip(prev["run"], prev["seed"]))

    def record(run: str, seed: int, base: dict, table: pd.DataFrame, shocked_metrics: dict):
        row = {"run": run, "seed": seed,
               "displaced_weighted_m": float(w[table["displaced"].to_numpy()].sum() / 1e6)}
        row.update(deltas(base, shocked_metrics))
        pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
        print(row, flush=True)

    # ---- (a) channel split, seed 0, unscaled -------------------------------
    full0 = apply_shocks(persons, scenario, seed=0)
    for channel in ("displacement", "wage", "capital"):
        run = f"channel_{channel}"
        if (run, 0) in done:
            continue
        table = channel_table(persons, full0, channel)
        sim = build_shocked_simulation(None, table, PERIOD, base_arrays=arrays)
        m = metrics(sim)
        del sim
        record(run, 0, b, table, m)

    # ---- (b) unscaled central draws ----------------------------------------
    for seed in range(N_DRAWS):
        if ("central", seed) in done:
            continue
        table = apply_shocks(persons, scenario, seed=seed)
        sim = build_shocked_simulation(None, table, PERIOD, base_arrays=arrays)
        m = metrics(sim)
        del sim
        record("central", seed, b, table, m)

    # ---- (b) SPI-scaled baseline and draws ---------------------------------
    scaled_persons = persons.copy()
    scaled_persons["dividend_income"] = dividends * factor
    base_scaled = microsimulation()
    base_scaled.set_input("dividend_income", PERIOD, dividends * factor)
    bs = metrics(base_scaled)
    arrays_scaled = base_arrays_for(base_scaled)
    del base_scaled

    def scaled_shocked(table):
        # build_shocked_simulation sets the shocked dividend column from the
        # table, which already carries the SPI-scaled baseline x 1.398
        return build_shocked_simulation(None, table, PERIOD, base_arrays=arrays_scaled)

    if ("spi_scaled_channel_capital", 0) not in done:
        full0s = apply_shocks(scaled_persons, scenario, seed=0)
        table = channel_table(scaled_persons, full0s, "capital")
        sim = scaled_shocked(table)
        m = metrics(sim)
        del sim
        record("spi_scaled_channel_capital", 0, bs, table, m)
    for seed in range(N_DRAWS):
        if ("spi_scaled", seed) in done:
            continue
        table = apply_shocks(scaled_persons, scenario, seed=seed)
        sim = scaled_shocked(table)
        m = metrics(sim)
        del sim
        record("spi_scaled", seed, bs, table, m)

    # ---- summary -----------------------------------------------------------
    draws = pd.read_csv(CSV)
    cols = ["exchequer_cost_bn", "poverty_change_bhc_pp", "poverty_change_ahc_pp",
            "gini_change_pp", "dividend_tax_change_bn", "displaced_weighted_m"]
    central = draws[draws.run == "central"].sort_values("seed")
    scaled = draws[draws.run == "spi_scaled"].sort_values("seed")
    if len(central) != N_DRAWS or len(scaled) != N_DRAWS:
        raise ValueError("dividend coverage checkpoint is incomplete")
    diff = scaled[cols].to_numpy() - central[cols].to_numpy()
    paired = pd.DataFrame(diff, columns=cols)
    channels = {
        r.run.replace("channel_", ""): {c: float(getattr(r, c)) for c in cols}
        for r in draws[draws.run.str.startswith("channel_")].itertuples()
    }
    out = {
        "provenance": provenance,
        "period": PERIOD,
        "draws": N_DRAWS,
        "seeds": [0, N_DRAWS - 1],
        "spi_source": SPI_SOURCE,
        "spi_dividends_2026_bn": SPI_DIVIDENDS_2026 / 1e9,
        "microcosm_dividends_2026_bn": microcosm_total / 1e9,
        "microcosm_dividend_holders_m": holders_m,
        "microcosm_to_spi_ratio": microcosm_total / SPI_DIVIDENDS_2026,
        "scale_factor": factor,
        "baseline_dividend_tax_bn": b["dividend_tax"] / 1e9,
        "baseline_dividend_tax_spi_scaled_bn": bs["dividend_tax"] / 1e9,
        "channels_seed0": channels,
        "channels_seed0_sum_cost_bn": float(sum(v["exchequer_cost_bn"] for v in channels.values())),
        "capital_channel_seed0_spi_scaled": {
            c: float(draws[draws.run == "spi_scaled_channel_capital"][c].iloc[0]) for c in cols
        },
        "central": summarise(central, cols),
        "spi_scaled": summarise(scaled, cols),
        "paired_difference_spi_minus_central": summarise(paired, cols),
    }
    (OUT / "dividend_coverage.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("microcosm_dividends_2026_bn", "scale_factor",
                                          "channels_seed0", "central", "spi_scaled")},
                     indent=1)[:4000])


if __name__ == "__main__":
    main()
