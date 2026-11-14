"""Capital channel and capital-income coverage: channel split and SPI-scaled reruns.

The capital shock scales savings interest and dividend income by
1.405/1.005, so its fiscal size depends on how much capital income the
microdata carry. Microcosm UK's 2026 dividends and savings interest both run
above the HMRC Survey of Personal Incomes (SPI) projection for 2026. This
script runs the central exposure-proportional shock over 20 paired draws
(seeds 0-19; every run uses the same displacement draw for a given seed,
asserted) in these runs:

  central             the unscaled shock;
  channel_displacement, channel_wage
                      each channel of the same draw on its own, the other
                      two held at baseline;
  channel_capital     the capital channel on its own (it does not depend on
                      the draw, so it is run once and applies to every seed);
  div_uniform         dividends scaled by one factor, SPI / Microcosm;
  div_banded          dividends scaled band by band to the SPI totals of
                      each total-income band (the top band, total income
                      above £1m, carries £23.96bn in Microcosm against
                      £12.37bn in SPI);
  savings_uniform     savings interest scaled by one factor, SPI / Microcosm
                      (£27.8bn against £20.4bn).

Basis. SPI covers income tax payers: its projection runs over total-income
bands from £12,570. The Microcosm numerator of every factor is therefore
restricted to people with baseline total income of at least £12,570, and
only their capital income is scaled; income below the threshold has no SPI
counterpart and is left unchanged. Bands are fixed on unscaled baseline
total income.

Each scaled run has its own scaled baseline, so its deltas are against a
baseline with the same capital income. The JSON reports, for each scaled
run, the 20-draw summary and the seed-paired difference from ``central``;
for the channels, the 20-draw means, their sum and the interaction
(central minus the sum).

SPI benchmark: policyengine-uk-data storage/incomes_projection.csv (commit
fa4ebe9, "regenerate SPI projections from the 2023 outturn"): HMRC SPI
2023-24 outturn projected to 2026. This is the projection Microcosm's own SPI
calibration targets come from.

Outputs results/robustness/dividend_coverage_draws.csv (stamped draw
checkpoint, continued only with --resume) and
results/robustness/dividend_coverage.json.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uk_ai_study.checkpoint import Checkpoint, certified_data_build_id  # noqa: E402
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
SPI_THRESHOLD = 12_570
#: SPI 2026 projection by total-income band (incomes_projection.csv @ fa4ebe9), £
SPI_DIVIDEND_BANDS_2026 = {
    (12_570, 15_000): 194_992_287, (15_000, 20_000): 952_676_600,
    (20_000, 30_000): 3_710_424_654, (30_000, 40_000): 5_236_935_698,
    (40_000, 50_000): 9_359_629_759, (50_000, 70_000): 13_928_020_474,
    (70_000, 100_000): 10_518_441_062, (100_000, 150_000): 7_654_840_053,
    (150_000, 200_000): 3_844_133_651, (200_000, 300_000): 4_122_694_060,
    (300_000, 500_000): 3_821_848_818, (500_000, 1_000_000): 4_256_403_057,
    (1_000_000, float("inf")): 12_368_082_181,
}
SPI_SAVINGS_BANDS_2026 = {
    (12_570, 15_000): 305_302_209, (15_000, 20_000): 1_526_511_044,
    (20_000, 30_000): 3_487_576_327, (30_000, 40_000): 2_819_031_344,
    (40_000, 50_000): 2_083_631_863, (50_000, 70_000): 2_607_325_433,
    (70_000, 100_000): 1_805_071_453, (100_000, 150_000): 1_403_944_464,
    (150_000, 200_000): 724_257_065, (200_000, 300_000): 816_739_121,
    (300_000, 500_000): 749_884_622, (500_000, 1_000_000): 727_599_790,
    (1_000_000, float("inf")): 1_381_659_631,
}
#: the file's 12570-inf rows; the band rows above sum to these
SPI_DIVIDENDS_2026 = 79_969_122_354.0
SPI_SAVINGS_2026 = 20_438_534_364.0
SPI_SOURCE = (
    "policyengine-uk-data policyengine_uk_data/storage/incomes_projection.csv "
    "@ fa4ebe9: HMRC SPI 2023-24 outturn projected to 2026, "
    "dividend_income_amount and savings_interest_income_amount by total "
    "income band, bands from 12570 (income tax payers)"
)
#: scaled runs: name -> (income column, scaling method)
SCALED_RUNS = {
    "div_uniform": ("dividend_income", "uniform"),
    "div_banded": ("dividend_income", "banded"),
    "savings_uniform": ("savings_interest_income", "uniform"),
}
SPI_BANDS = {"dividend_income": SPI_DIVIDEND_BANDS_2026,
             "savings_interest_income": SPI_SAVINGS_BANDS_2026}
SPI_TOTALS = {"dividend_income": SPI_DIVIDENDS_2026,
              "savings_interest_income": SPI_SAVINGS_2026}
CHANNELS = ("displacement", "wage", "capital")


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
        "savings_tax": float((person(sim, "savings_income_tax") * pw).sum()),
    }


def deltas(base: dict, shocked: dict) -> dict:
    return {
        "exchequer_cost_bn": (base["gov"] - shocked["gov"]) / 1e9,
        "poverty_change_bhc_pp": 100 * (shocked["pov_bhc"] - base["pov_bhc"]),
        "poverty_change_ahc_pp": 100 * (shocked["pov_ahc"] - base["pov_ahc"]),
        "gini_change_pp": 100 * (shocked["gini"] - base["gini"]),
        "dividend_tax_change_bn": (shocked["dividend_tax"] - base["dividend_tax"]) / 1e9,
        "savings_tax_change_bn": (shocked["savings_tax"] - base["savings_tax"]) / 1e9,
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


def band_index(total_income: np.ndarray, bands) -> np.ndarray:
    """Index of each person's SPI band (-1 below the threshold)."""
    out = np.full(len(total_income), -1, dtype=int)
    for i, (lo, hi) in enumerate(bands):
        out[(total_income >= lo) & (total_income < hi)] = i
    return out


def scaling_factors(values, weights, total_income, column: str, method: str) -> tuple[np.ndarray, list]:
    """Per-person multiplier taking ``values`` to the SPI basis.

    Only people with total income >= SPI_THRESHOLD are scaled. ``uniform``
    applies SPI_total / Microcosm_total(taxpayers) to all of them;
    ``banded`` applies SPI_band / Microcosm_band within each band. A band
    with no Microcosm income keeps factor 1 and is flagged. Returns the
    multipliers and the band table.
    """
    bands = SPI_BANDS[column]
    idx = band_index(total_income, list(bands))
    taxpayer = idx >= 0
    factor = np.ones(len(values), dtype=float)
    table = []
    if method == "uniform":
        micro = float((values * weights)[taxpayer].sum())
        f = SPI_TOTALS[column] / micro if micro > 0 else 1.0
        factor[taxpayer] = f
    for i, ((lo, hi), spi) in enumerate(bands.items()):
        m = idx == i
        micro = float((values * weights)[m].sum())
        if method == "banded":
            f = spi / micro if micro > 0 else 1.0
            factor[m] = f
        else:
            f = factor[m][0] if m.any() else float("nan")
        table.append({"lower": lo, "upper": hi, "spi_bn": spi / 1e9, "microcosm_bn": micro / 1e9,
                      "factor": float(f), "empty_band": bool(micro <= 0)})
    return factor, table


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
    stamp_ckpt = Checkpoint(CSV, ("run", "seed"), script=__file__)

    # ---- unscaled baseline -------------------------------------------------
    baseline = microsimulation()
    provenance = bundle_provenance(baseline)
    persons = build_person_table(baseline, PERIOD)
    w = persons["weight"].to_numpy(float)
    total_income = person(baseline, "total_income").astype(float)
    taxpayer = total_income >= SPI_THRESHOLD
    coverage = {}
    for column in SPI_BANDS:
        v = persons[column].to_numpy(float)
        coverage[column] = {
            "microcosm_all_bn": float((v * w).sum() / 1e9),
            "microcosm_taxpayers_bn": float((v * w)[taxpayer].sum() / 1e9),
            "microcosm_below_threshold_bn": float((v * w)[~taxpayer].sum() / 1e9),
            "spi_bn": SPI_TOTALS[column] / 1e9,
            "holders_m": float(w[v > 0].sum() / 1e6),
        }
    b = metrics(baseline)
    arrays = base_arrays_for(baseline)
    del baseline
    gc.collect()

    masks = {}

    def draw(table_persons, seed):
        table = apply_shocks(table_persons, scenario, seed=seed)
        disp = table["displaced"].to_numpy()
        if seed in masks and not np.array_equal(masks[seed], disp):
            raise AssertionError(f"seed {seed}: displacement draw differs across runs")
        masks.setdefault(seed, disp)
        return table

    def run(name, seed, base, table, arrays_):
        if stamp_ckpt.is_done(name, seed):
            return
        sim = build_shocked_simulation(None, table, PERIOD, base_arrays=arrays_)
        m = metrics(sim)
        del sim
        gc.collect()
        row = {"run": name, "seed": seed,
               "displaced_weighted_m": float(w[table["displaced"].to_numpy()].sum() / 1e6)}
        row.update(deltas(base, m))
        stamp_ckpt.append(row)
        print(row, flush=True)

    # ---- central and its channels, unscaled --------------------------------
    for seed in range(N_DRAWS):
        full = draw(persons, seed)
        run("central", seed, b, full, arrays)
        for channel in ("displacement", "wage"):
            run(f"channel_{channel}", seed, b, channel_table(persons, full, channel), arrays)
        if seed == 0:
            run("channel_capital", 0, b, channel_table(persons, full, "capital"), arrays)

    # ---- SPI-scaled runs ---------------------------------------------------
    scaling = {}
    for name, (column, method) in SCALED_RUNS.items():
        values = persons[column].to_numpy(float)
        factor, table_ = scaling_factors(values, w, total_income, column, method)
        scaled_values = values * factor
        scaling[name] = {
            "column": column, "method": method, "bands": table_,
            "scaled_total_bn": float((scaled_values * w).sum() / 1e9),
            "scaled_taxpayers_bn": float((scaled_values * w)[taxpayer].sum() / 1e9),
        }
        scaled_persons = persons.copy()
        scaled_persons[column] = scaled_values
        base_scaled = microsimulation()
        base_scaled.set_input(column, PERIOD, scaled_values)
        bs = metrics(base_scaled)
        arrays_scaled = base_arrays_for(base_scaled)
        del base_scaled
        gc.collect()
        scaling[name]["baseline_dividend_tax_bn"] = bs["dividend_tax"] / 1e9
        scaling[name]["baseline_gov_balance_change_bn"] = (bs["gov"] - b["gov"]) / 1e9
        for seed in range(N_DRAWS):
            # the shocked table carries the scaled baseline x 1.398 in the
            # scaled column; build_shocked_simulation sets it as input
            run(name, seed, bs, draw(scaled_persons, seed), arrays_scaled)
        full0 = draw(scaled_persons, 0)
        run(f"{name}_channel_capital", 0, bs, channel_table(scaled_persons, full0, "capital"), arrays_scaled)

    # ---- summary -----------------------------------------------------------
    draws = stamp_ckpt.read()
    cols = ["exchequer_cost_bn", "poverty_change_bhc_pp", "poverty_change_ahc_pp",
            "gini_change_pp", "dividend_tax_change_bn", "savings_tax_change_bn",
            "displaced_weighted_m"]

    def series(name):
        d = draws[draws.run == name].sort_values("seed")
        if list(d["seed"]) != list(range(N_DRAWS)):
            raise ValueError(f"dividend coverage checkpoint is incomplete for {name}")
        return d

    central = series("central")
    out = {
        "provenance": provenance,
        "certified_data_build_id": certified_data_build_id(),
        "period": PERIOD,
        "draws": N_DRAWS,
        "seeds": [0, N_DRAWS - 1],
        "spi_source": SPI_SOURCE,
        "basis": (
            "SPI covers income tax payers (total income from £12,570). Scaling "
            "factors use Microcosm capital income of people with baseline total "
            "income >= £12,570 as the numerator and scale only their income; "
            "income below the threshold is left unchanged."
        ),
        "coverage": coverage,
        # legacy keys kept for the paper macros (dividends, uniform factor)
        "spi_dividends_2026_bn": SPI_DIVIDENDS_2026 / 1e9,
        "microcosm_dividends_2026_bn": coverage["dividend_income"]["microcosm_all_bn"],
        "microcosm_dividends_taxpayers_2026_bn": coverage["dividend_income"]["microcosm_taxpayers_bn"],
        "microcosm_dividend_holders_m": coverage["dividend_income"]["holders_m"],
        "microcosm_to_spi_ratio": coverage["dividend_income"]["microcosm_taxpayers_bn"]
        / (SPI_DIVIDENDS_2026 / 1e9),
        "scale_factor": SPI_DIVIDENDS_2026 / (coverage["dividend_income"]["microcosm_taxpayers_bn"] * 1e9),
        "savings_scale_factor": SPI_SAVINGS_2026
        / (coverage["savings_interest_income"]["microcosm_taxpayers_bn"] * 1e9),
        "baseline_dividend_tax_bn": b["dividend_tax"] / 1e9,
        "scaling": scaling,
        "central": summarise(central, cols),
    }
    channels = {}
    for ch in CHANNELS:
        if ch == "capital":
            d = draws[draws.run == "channel_capital"]
            if len(d) != 1:
                raise ValueError("capital channel checkpoint missing")
            channels[ch] = {c: {"mean": float(d[c].iloc[0]), "sd": 0.0} for c in cols}
        else:
            channels[ch] = summarise(series(f"channel_{ch}"), cols)
    out["channels"] = channels
    out["channels_sum"] = {c: float(sum(channels[ch][c]["mean"] for ch in CHANNELS)) for c in cols}
    out["channels_interaction"] = {
        c: float(out["central"][c]["mean"] - out["channels_sum"][c]) for c in cols
    }
    # seed-0 channel split kept for reference
    s0 = {ch: draws[(draws.run == f"channel_{ch}") & (draws.seed == 0)] for ch in CHANNELS}
    out["channels_seed0"] = {ch: {c: float(d[c].iloc[0]) for c in cols} for ch, d in s0.items()}
    out["channels_seed0_sum_cost_bn"] = float(sum(v["exchequer_cost_bn"] for v in out["channels_seed0"].values()))
    for name in SCALED_RUNS:
        scaled = series(name)
        diff = scaled[cols].to_numpy() - central[cols].to_numpy()
        out[name] = summarise(scaled, cols)
        out[f"paired_difference_{name}_minus_central"] = summarise(pd.DataFrame(diff, columns=cols), cols)
        cap = draws[draws.run == f"{name}_channel_capital"]
        out[f"{name}_channel_capital"] = {c: float(cap[c].iloc[0]) for c in cols}
    # legacy aliases for the uniform dividend run
    out["spi_scaled"] = out["div_uniform"]
    out["paired_difference_spi_minus_central"] = out["paired_difference_div_uniform_minus_central"]
    out["capital_channel_seed0_spi_scaled"] = out["div_uniform_channel_capital"]
    (OUT / "dividend_coverage.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("coverage", "channels_sum", "channels_interaction",
                                          "central")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
