"""Export headline results into public/data/ for the web embeds.

The blog post on policyengine.org iframes static HTML pages served from this
repo's Vercel deployment (public/embed-*.html). Those pages read the CSVs
written here, so re-running the analysis pipeline followed by this script and
redeploying updates the post automatically.

Usage: python analysis/export_site_data.py
"""

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "public" / "data"

SCENARIOS = ["low", "central", "central_youth_tilted", "high"]
SCENARIO_LABELS = {
    "low": "Low (1% displacement, no wage uplift)",
    "central": "Central (7% displacement, +2.6% wages)",
    "central_youth_tilted": "Central, youth-tilted",
    "high": "High (13% displacement, +2.6% wages)",
}


def load(name: str) -> dict:
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {s: load(s) for s in SCENARIOS}

    # Headline scenario table (budget, poverty, inequality, displacement)
    with open(OUT / "scenarios.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "label",
                "displacement_rate",
                "wage_uplift",
                "displaced_workers",
                "exchequer_cost_bn",
                "poverty_change_bhc_pp",
                "poverty_change_ahc_pp",
                "gini_baseline",
                "gini_shocked",
                "gini_change_pp",
            ]
        )
        for s in SCENARIOS:
            d = data[s]
            w.writerow(
                [
                    s,
                    SCENARIO_LABELS[s],
                    d["displacement_rate"],
                    d["wage_uplift"],
                    round(d["displaced_weighted"]),
                    round(d["exchequer_cost"] / 1e9, 2),
                    round(d["poverty_rate_change_bhc"] * 100, 2),
                    round(d["poverty_rate_change_ahc"] * 100, 2),
                    round(d["gini_baseline"], 4),
                    round(d["gini_shocked"], 4),
                    round((d["gini_shocked"] - d["gini_baseline"]) * 100, 2),
                ]
            )

    # Decile income change per scenario (long format)
    with open(OUT / "deciles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "decile", "income_change_gbp"])
        for s in SCENARIOS:
            for decile, change in data[s]["decile_income_change"].items():
                w.writerow([s, decile, round(change, 2)])

    # Age-band incidence (central scenario)
    with open(OUT / "age_bands.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["scenario", "age_band", "displacement_share", "income_change_gbp"]
        )
        for s in ["central", "central_youth_tilted"]:
            d = data[s]
            for band in d["age_band_displacement_share"]:
                w.writerow(
                    [
                        s,
                        band,
                        round(d["age_band_displacement_share"][band], 4),
                        round(d["age_band_income_change"][band], 2),
                    ]
                )

    # Regional summary (central scenario, from geo pipeline)
    src = RESULTS / "geo" / "region_summary.csv"
    if src.exists():
        (OUT / "regions.csv").write_text(src.read_text())

    # Five incidence families: 50-draw Monte Carlo means/sds, displaced count
    # as the 50-draw mean (the table caption says "50 paired draws")
    mc_path = RESULTS / "robustness" / "incidence_monte_carlo.json"
    draws_path = RESULTS / "robustness" / "incidence_draws_five.csv"
    if mc_path.exists() and draws_path.exists():
        with open(mc_path) as f:
            mc = json.load(f)
        sums, counts = {}, {}
        with open(draws_path) as f:
            for row in csv.DictReader(f):
                if row.get("wage_axis", "central") != "central":
                    continue
                fam = row["family"]
                sums[fam] = sums.get(fam, 0.0) + float(row["displaced_weighted_m"])
                counts[fam] = counts.get(fam, 0) + 1
        displaced = {k: sums[k] / counts[k] for k in sums}
        family_labels = {
            "exposure": "Exposure-proportional",
            "junior": "Junior-concentrated",
            "compression": "Expertise-compression",
            "uniform": "Uniform",
            "klein_top_loaded": "Top-loaded stress test",
        }
        with open(OUT / "incidence.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "family",
                    "label",
                    "displaced_m",
                    "cost_mean_bn",
                    "cost_sd_bn",
                    "pov_bhc_mean_pp",
                    "pov_bhc_sd_pp",
                    "gini_mean_pp",
                    "gini_sd_pp",
                ]
            )
            for key, label in family_labels.items():
                d = mc[key]
                w.writerow(
                    [
                        key,
                        label,
                        round(displaced.get(key, float("nan")), 2),
                        round(d["exchequer_cost_bn"]["mean"], 1),
                        round(d["exchequer_cost_bn"]["sd"], 1),
                        round(d["poverty_change_bhc_pp"]["mean"], 2),
                        round(d["poverty_change_bhc_pp"]["sd"], 2),
                        round(d["gini_change_pp"]["mean"], 2),
                        round(d["gini_change_pp"]["sd"], 2),
                    ]
                )

    # Adjustment margin (wage cuts vs displacement, fixed gross loss)
    for name, dest in [
        ("robustness/mixed_adjustment.csv", "wage_margin.csv"),
        ("policy/summary.csv", "policy.csv"),
        ("caseloads/summary.csv", "caseloads.csv"),
        ("jr16/fig4_1_transition_by_decile.csv", "transition.csv"),
        ("jr16/fig4_2_wage_gain_by_decile.csv", "wage_gains.csv"),
        ("jr16/fig4_3_capital_by_decile.csv", "capital.csv"),
        ("jr16/fig4_4_decomposition.csv", "decomposition.csv"),
        ("tax_composition/composition_grid.csv", "composition_grid.csv"),
        ("geo/constituency_impacts.csv", "constituency_impacts.csv"),
    ]:
        src = RESULTS / name
        if src.exists():
            (OUT / dest).write_text(src.read_text())

    # Shock-size grid: write exactly the columns chart-grid.html reads, by name
    # (the results file gained columns; the page indexes by position)
    grid_cols = ["unemployment_pct", "wage_pct", "avg_disposable_income_change_pct",
                 "net_revenue_change_bn", "net_revenue_change_pct_of_receipts", "gini_change_pp"]
    with open(RESULTS / "jr16" / "grid.csv") as f, open(OUT / "grid.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(grid_cols)
        for row in csv.DictReader(f):
            w.writerow([row[c] for c in grid_cols])

    # Policy reforms: 50-draw means (the post and the paper's table use these)
    pmc = RESULTS / "robustness" / "policy_monte_carlo.json"
    if pmc.exists():
        with open(pmc) as f:
            pm = json.load(f)
        reforms = [("R1", "R1_wage_insurance_exposure", "R1 Wage insurance"),
                   ("R2", "R2_uc_circuit_breaker_exposure", "R2 UC standard allowance +20%"),
                   ("R3", "R3_cap_suspension_taper_cut_exposure", "R3 Cap suspension + taper cut")]
        with open(OUT / "policy_mc.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reform", "label", "cost_mean_bn", "cost_sd_bn", "pov_bhc_mean_pp",
                        "pov_bhc_sd_pp", "gini_mean_pp", "cost_per_bhc_pp_bn"])
            for key, run, label in reforms:
                d = pm[run]
                cost = d["exchequer_cost_bn"]["mean"]
                pov = d["poverty_change_bhc_pp"]["mean"]
                w.writerow([key, label, round(cost, 1), round(d["exchequer_cost_bn"]["sd"], 1),
                            round(pov, 2), round(d["poverty_change_bhc_pp"]["sd"], 2),
                            round(d["gini_change_pp"]["mean"], 2), round(cost / -pov, 1)])

    # Per-family decile profiles for the incidence chart (seed 0)
    family_files = {
        "exposure": "exposure.json",
        "junior": "junior.json",
        "compression": "compression.json",
        "uniform": "uniform.json",
        "klein_top_loaded": "klein_top_loaded.json",
    }
    inc_dir = RESULTS / "incidence"
    if all((inc_dir / f).exists() for f in family_files.values()):
        with open(OUT / "incidence_deciles.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["family", "decile", "transition_share_pct", "income_change_pct"]
            )
            for key, fname in family_files.items():
                with open(inc_dir / fname) as fh:
                    d = json.load(fh)
                trans = d["decile_transition_share_pct"]
                income = d["decile_income_change_pct"]
                for decile in sorted(trans, key=int):
                    w.writerow(
                        [
                            key,
                            decile,
                            round(trans[decile], 3),
                            round(income[decile], 3),
                        ]
                    )

    # Paper figures, served at /figures/<name>.png
    figures = [
        "jr16/fig4_1_transition.png",
        "jr16/fig4_2_wages.png",
        "jr16/fig4_3_capital.png",
        "jr16/fig4_4_decomposition.png",
        "jr16/fig4_5_disposable_grid.png",
        "jr16/fig4_6_exchequer_grid.png",
        "jr16/fig4_7_gini_grid.png",
        "incidence/incidence_families.png",
        "policy/policy_reforms.png",
        "tax_composition/revenue_shortfall_phi.png",
        "caseloads/caseloads.png",
    ]
    figdir = OUT.parent / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for rel in figures:
        src = RESULTS / rel
        if src.exists():
            shutil.copy(src, figdir / Path(rel).name)

    print(f"Wrote embeds data to {OUT} and figures to {figdir}")


if __name__ == "__main__":
    main()
