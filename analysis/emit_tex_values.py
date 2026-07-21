#!/usr/bin/env python3
"""Mechanically emit paper/values_generated.tex from canonical results files.

Part of the R2-1 fix: headline numbers in the manuscript were hand-transcribed
from results/ JSON/CSV files, so a rebuild could silently desynchronise prose
and artifacts. This script derives every headline value from the canonical
results files and writes them as LaTeX \\newcommand macros. The .tex sections
can then migrate to \\input{values_generated} macros (and, later, fully
generated tables) instead of literals; this script does not edit the .tex
prose itself.

Usage:  python analysis/emit_tex_values.py   ->  paper/values_generated.tex

If a source file or key is missing (e.g. mid-refactor of the upstream
scripts), the macro is emitted as \\GENMISSING with a comment naming the
missing source, so a stale/incomplete tree fails visibly at LaTeX build time
rather than silently keeping old numbers.

Macros emitted (grouped by source):

  central/low/high presets (results/{central,low,high}.json,
      results/wage_margin_{central,pss}.json, results/central_ripple.json):
    \\genCentralCostBn \\genCentralPovBhcPp \\genCentralPovAhcPp
    \\genCentralGiniChangePp \\genBaselineGini \\genCentralDisplacedM
    \\genLowCostBn \\genHighCostBn \\genHighPovBhcPp
    \\genWageMarginCentralCostBn \\genWageMarginPssCostBn
    \\genWageMarginPssGiniChangePp \\genCentralRippleCostBn

  central 20-draw Monte Carlo (results/robustness/central_monte_carlo.json):
    \\genCentralMcCostMeanBn \\genCentralMcCostSdBn \\genCentralMcCostMinBn
    \\genCentralMcCostMaxBn \\genCentralMcPovBhcMeanPp \\genCentralMcPovBhcSdPp
    \\genCentralMcGiniMeanPp \\genCentralMcGiniSdPp

  incidence-family Monte Carlo (results/robustness/incidence_monte_carlo.json):
    \\genInc<Family>CostMeanBn \\genInc<Family>CostSdBn
    \\genInc<Family>PovBhcMeanPp \\genInc<Family>GiniMeanPp
    for <Family> in Exposure, Junior, Compression, Uniform, Klein; plus
    \\genIncFamilyCostMeanMinBn \\genIncFamilyCostMeanMaxBn
    \\genIncFamilyPovBhcMeanMinPp \\genIncFamilyPovBhcMeanMaxPp
    \\genIncFamilyGiniMeanMinPp \\genIncFamilyGiniMeanMaxPp

  seed-0 incidence families (results/incidence/summary_five.csv):
    \\genSeedZero<Family>CostBn \\genSeedZero<Family>PovBhcPp
    \\genSeedZero<Family>GiniPp   (same five families)

  duration / take-up sensitivities
      (results/robustness/duration_takeup_sensitivity.json):
    \\genDurSixMoCostMeanBn \\genDurSixMoCostSdBn \\genDurSixMoPovBhcMeanPp
    \\genDurSixMoPovBhcSdPp \\genTakeupCostMeanBn \\genTakeupCostSdBn
    \\genTakeupPovBhcMeanPp \\genTakeupPovBhcSdPp

  policy reforms, seed 0 (results/policy/summary.csv and R1 exposure JSON):
    \\genRoneCostBn \\genRoneCostPerPpAhcBn \\genRtwoCostBn
    \\genRtwoCostPerPpAhcBn \\genRthreeCostBn \\genRthreeCostPerPpAhcBn
    \\genRoneTopThreeDecileSharePct \\genRtwoBottomFiveDecileSharePct
    \\genRthreeBottomFiveDecileSharePct

  policy Monte Carlo (results/robustness/policy_monte_carlo.json):
    \\genRoneMcCostMeanBn \\genRoneMcCostSdBn \\genRoneMcPovBhcMeanPp
    \\genRoneMcPovBhcSdPp \\genRtwoMcCostMeanBn \\genRtwoMcCostSdBn
    \\genRtwoMcPovBhcMeanPp \\genRtwoMcPovBhcSdPp \\genRthreeMcCostMeanBn
    \\genRthreeMcCostSdBn \\genRthreeMcPovBhcMeanPp \\genRthreeMcPovBhcSdPp

  caseloads (results/caseloads/central.json):
    \\genCaseloadNetNewUcBenunitsK \\genCaseloadNewUcPersonsK
    \\genCaseloadUcSpendChangeBn

  adjustment margin (results/robustness/mixed_wage_adjustment.json):
    \\genMixedWageCutCostBn \\genMixedDisplacementCostBn
    \\genMixedWageCutPovBhcPp \\genMixedDisplacementPovBhcPp
    \\genMixedWageCutGiniPp \\genMixedDisplacementGiniPp

  scenario grid (results/jr16/grid.csv):
    \\genGridCells \\genGridGiniMinPp \\genGridGiniMaxPp

  index sensitivity (results/robustness/index_sensitivity_full.json):
    \\genIndexCostMinBn \\genIndexCostMaxBn \\genIndexGiniMinPp
    \\genIndexGiniMaxPp \\genIndexPovBhcMinPp \\genIndexPovBhcMaxPp

  dividend recycling (results/tax_composition/recycling_case.json):
    \\genRecyclingDividendsBn \\genRecyclingGiniChangePp
    (fields re-read on rebuild; robust to the pending x(1-CT) change)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "values_generated.tex"

FAMILY_MACRO = {
    "exposure": "Exposure",
    "junior": "Junior",
    "compression": "Compression",
    "uniform": "Uniform",
    "klein_top_loaded": "Klein",
}
REFORM_MACRO = {
    "R1_wage_insurance": "Rone",
    "R2_uc_circuit_breaker": "Rtwo",
    "R3_cap_suspension_taper_cut": "Rthree",
}

lines: list[str] = []
missing: list[str] = []


def emit(name: str, value, fmt: str = "{:.1f}", source: str = ""):
    """Emit one macro; on any failure emit a loud placeholder."""
    try:
        if callable(value):
            value = value()
        text = fmt.format(value) if not isinstance(value, str) else value
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    except Exception as exc:  # noqa: BLE001 — every miss must surface, not abort
        missing.append(f"{name} ({source or 'unknown source'}: {exc})")
        lines.append(f"\\newcommand{{\\{name}}}{{\\GENMISSING}}"
                     f" % MISSING: {source} -> {exc}")


def jload(rel: str) -> dict:
    return json.loads((R / rel).read_text())


def main() -> None:
    # --- presets -----------------------------------------------------------
    for rel, prefix in [("central.json", "Central"), ("low.json", "Low"),
                        ("high.json", "High")]:
        emit(f"gen{prefix}CostBn",
             lambda rel=rel: jload(rel)["exchequer_cost"] / 1e9, source=rel)
    c = "central.json"
    emit("genCentralPovBhcPp",
         lambda: jload(c)["poverty_rate_change_bhc"] * 100, "{:.2f}", c)
    emit("genCentralPovAhcPp",
         lambda: jload(c)["poverty_rate_change_ahc"] * 100, "{:.2f}", c)
    emit("genCentralGiniChangePp",
         lambda: (jload(c)["gini_shocked"] - jload(c)["gini_baseline"]) * 100,
         "{:.2f}", c)
    emit("genBaselineGini", lambda: jload(c)["gini_baseline"], "{:.3f}", c)
    emit("genCentralDisplacedM",
         lambda: jload(c)["displaced_weighted"] / 1e6, "{:.1f}", c)
    emit("genHighPovBhcPp",
         lambda: jload("high.json")["poverty_rate_change_bhc"] * 100,
         "{:.2f}", "high.json")
    for rel, name in [("wage_margin_central.json", "genWageMarginCentralCostBn"),
                      ("wage_margin_pss.json", "genWageMarginPssCostBn"),
                      ("central_ripple.json", "genCentralRippleCostBn")]:
        emit(name, lambda rel=rel: jload(rel)["exchequer_cost"] / 1e9, source=rel)
    emit("genWageMarginPssGiniChangePp",
         lambda: (jload("wage_margin_pss.json")["gini_shocked"]
                  - jload("wage_margin_pss.json")["gini_baseline"]) * 100,
         "{:.2f}", "wage_margin_pss.json")

    # --- central Monte Carlo ----------------------------------------------
    mc = "robustness/central_monte_carlo.json"
    for stat, suf, fmt in [("mean", "Mean", "{:.1f}"), ("sd", "Sd", "{:.1f}"),
                           ("min", "Min", "{:.1f}"), ("max", "Max", "{:.1f}")]:
        emit(f"genCentralMcCost{suf}Bn",
             lambda stat=stat: jload(mc)["exchequer_cost_bn"][stat], fmt, mc)
    for key, base in [("poverty_change_bhc_pp", "genCentralMcPovBhc"),
                      ("gini_change_pp", "genCentralMcGini")]:
        emit(f"{base}MeanPp", lambda key=key: jload(mc)[key]["mean"], "{:.2f}", mc)
        emit(f"{base}SdPp", lambda key=key: jload(mc)[key]["sd"], "{:.2f}", mc)

    # --- incidence-family Monte Carlo -------------------------------------
    imc = "robustness/incidence_monte_carlo.json"
    for fam, Fam in FAMILY_MACRO.items():
        emit(f"genInc{Fam}CostMeanBn",
             lambda fam=fam: jload(imc)[fam]["exchequer_cost_bn"]["mean"],
             "{:.1f}", imc)
        emit(f"genInc{Fam}CostSdBn",
             lambda fam=fam: jload(imc)[fam]["exchequer_cost_bn"]["sd"],
             "{:.1f}", imc)
        emit(f"genInc{Fam}PovBhcMeanPp",
             lambda fam=fam: jload(imc)[fam]["poverty_change_bhc_pp"]["mean"],
             "{:.2f}", imc)
        emit(f"genInc{Fam}GiniMeanPp",
             lambda fam=fam: jload(imc)[fam]["gini_change_pp"]["mean"],
             "{:.2f}", imc)

    def fam_span(key, agg):
        d = jload(imc)
        return agg(d[f][key]["mean"] for f in d)

    emit("genIncFamilyCostMeanMinBn",
         lambda: fam_span("exchequer_cost_bn", min), "{:.1f}", imc)
    emit("genIncFamilyCostMeanMaxBn",
         lambda: fam_span("exchequer_cost_bn", max), "{:.1f}", imc)
    emit("genIncFamilyPovBhcMeanMinPp",
         lambda: fam_span("poverty_change_bhc_pp", min), "{:.2f}", imc)
    emit("genIncFamilyPovBhcMeanMaxPp",
         lambda: fam_span("poverty_change_bhc_pp", max), "{:.2f}", imc)
    emit("genIncFamilyGiniMeanMinPp",
         lambda: fam_span("gini_change_pp", min), "{:.2f}", imc)
    emit("genIncFamilyGiniMeanMaxPp",
         lambda: fam_span("gini_change_pp", max), "{:.2f}", imc)

    # --- seed-0 incidence families ----------------------------------------
    s5 = "incidence/summary_five.csv"
    try:
        five = pd.read_csv(R / s5).set_index("family")
    except Exception:
        five = None
    for fam, Fam in FAMILY_MACRO.items():
        emit(f"genSeedZero{Fam}CostBn",
             lambda fam=fam: float(five.loc[fam, "exchequer_cost_bn"]),
             "{:.1f}", s5)
        emit(f"genSeedZero{Fam}PovBhcPp",
             lambda fam=fam: float(five.loc[fam, "poverty_change_bhc_pp"]),
             "{:.2f}", s5)
        emit(f"genSeedZero{Fam}GiniPp",
             lambda fam=fam: float(five.loc[fam, "gini_change_pp"]),
             "{:.2f}", s5)

    # --- duration / take-up sensitivities ---------------------------------
    dt = "robustness/duration_takeup_sensitivity.json"
    # "genDurSixMo" macro names kept for TeX stability; the variant was
    # renamed half_earnings_retention (R2-10) and duration_6m is accepted
    # only as a legacy fallback.
    for variant, pref in [("half_earnings_retention", "genDurSixMo"), ("takeup_70", "genTakeup")]:
        def res(variant=variant):
            d = jload(dt)
            key = variant if variant in d else next(
                k for k in d if k.startswith(variant.split("_")[0]))
            return d[key]["results"]
        emit(f"{pref}CostMeanBn",
             lambda res=res: res()["exchequer_cost_bn"]["mean"], "{:.1f}", dt)
        emit(f"{pref}CostSdBn",
             lambda res=res: res()["exchequer_cost_bn"]["sd"], "{:.1f}", dt)
        emit(f"{pref}PovBhcMeanPp",
             lambda res=res: res()["poverty_change_bhc_pp"]["mean"], "{:.2f}", dt)
        emit(f"{pref}PovBhcSdPp",
             lambda res=res: res()["poverty_change_bhc_pp"]["sd"], "{:.2f}", dt)

    # --- policy reforms, seed 0 -------------------------------------------
    ps = "policy/summary.csv"
    try:
        pol = pd.read_csv(R / ps)
        pol = pol[pol["family"] == "exposure"].set_index("reform")
    except Exception:
        pol = None
    for reform, Ref in REFORM_MACRO.items():
        emit(f"gen{Ref}CostBn",
             lambda reform=reform: float(pol.loc[reform, "gross_cost_bn"]),
             "{:.1f}", ps)
        emit(f"gen{Ref}CostPerPpAhcBn",
             lambda reform=reform: float(pol.loc[reform, "cost_per_pp_ahc_bn"]),
             "{:.1f}", ps)

    def decile_share(reform, deciles):
        d = jload(f"policy/{reform}_exposure.json")["decile_benefit_share_pct"]
        return sum(d[str(i)] for i in deciles)

    emit("genRoneTopThreeDecileSharePct",
         lambda: decile_share("R1_wage_insurance", range(8, 11)),
         "{:.0f}", "policy/R1_wage_insurance_exposure.json")
    emit("genRtwoBottomFiveDecileSharePct",
         lambda: decile_share("R2_uc_circuit_breaker", range(1, 6)),
         "{:.0f}", "policy/R2_uc_circuit_breaker_exposure.json")
    emit("genRthreeBottomFiveDecileSharePct",
         lambda: decile_share("R3_cap_suspension_taper_cut", range(1, 6)),
         "{:.0f}", "policy/R3_cap_suspension_taper_cut_exposure.json")

    # --- policy Monte Carlo -----------------------------------------------
    pmc = "robustness/policy_monte_carlo.json"
    pmc_keys = {"Rone": "R1_wage_insurance_exposure",
                "Rtwo": "R2_uc_circuit_breaker_exposure",
                "Rthree": "R3_cap_suspension_taper_cut_exposure"}
    for Ref, key in pmc_keys.items():
        emit(f"gen{Ref}McCostMeanBn",
             lambda key=key: jload(pmc)[key]["exchequer_cost_bn"]["mean"],
             "{:.1f}", pmc)
        emit(f"gen{Ref}McCostSdBn",
             lambda key=key: jload(pmc)[key]["exchequer_cost_bn"]["sd"],
             "{:.2f}", pmc)
        emit(f"gen{Ref}McPovBhcMeanPp",
             lambda key=key: jload(pmc)[key]["poverty_change_bhc_pp"]["mean"],
             "{:.2f}", pmc)
        emit(f"gen{Ref}McPovBhcSdPp",
             lambda key=key: jload(pmc)[key]["poverty_change_bhc_pp"]["sd"],
             "{:.2f}", pmc)

    # --- caseloads ---------------------------------------------------------
    cl = "caseloads/central.json"
    emit("genCaseloadNetNewUcBenunitsK",
         lambda: jload(cl)["net_new_uc_benunits_thousands"], "{:.0f}", cl)
    emit("genCaseloadNewUcPersonsK",
         lambda: jload(cl)["new_uc_persons_thousands"], "{:.0f}", cl)
    emit("genCaseloadUcSpendChangeBn",
         lambda: jload(cl)["uc_spend_change_bn"], "{:.1f}", cl)

    # --- adjustment margin -------------------------------------------------
    mw = "robustness/mixed_wage_adjustment.json"

    def mixed(end, key):
        rows = jload(mw)["mixed_adjustment"]
        return rows[0 if end == "wage" else -1][key]

    for end, Pref in [("wage", "genMixedWageCut"), ("disp", "genMixedDisplacement")]:
        emit(f"{Pref}CostBn",
             lambda end=end: mixed(end, "exchequer_cost_bn"), "{:.1f}", mw)
        emit(f"{Pref}PovBhcPp",
             lambda end=end: mixed(end, "poverty_change_bhc_pp"), "{:.2f}", mw)
        emit(f"{Pref}GiniPp",
             lambda end=end: mixed(end, "gini_change_pp"), "{:.2f}", mw)

    # --- scenario grid -----------------------------------------------------
    gr = "jr16/grid.csv"
    try:
        grid = pd.read_csv(R / gr)
    except Exception:
        grid = None
    emit("genGridCells", lambda: len(grid), "{:d}", gr)
    emit("genGridGiniMinPp", lambda: float(grid["gini_change_pp"].min()),
         "{:.2f}", gr)
    emit("genGridGiniMaxPp", lambda: float(grid["gini_change_pp"].max()),
         "{:.2f}", gr)

    # --- index sensitivity -------------------------------------------------
    ix = "robustness/index_sensitivity_full.json"

    def ix_span(key, agg):
        d = jload(ix)["results"]
        return agg(v[key] for v in d.values())

    emit("genIndexCostMinBn",
         lambda: ix_span("exchequer_cost_bn", min), "{:.1f}", ix)
    emit("genIndexCostMaxBn",
         lambda: ix_span("exchequer_cost_bn", max), "{:.1f}", ix)
    emit("genIndexGiniMinPp",
         lambda: ix_span("gini_change_pp", min), "{:.2f}", ix)
    emit("genIndexGiniMaxPp",
         lambda: ix_span("gini_change_pp", max), "{:.2f}", ix)
    emit("genIndexPovBhcMinPp",
         lambda: ix_span("poverty_change_bhc_pp", min), "{:.2f}", ix)
    emit("genIndexPovBhcMaxPp",
         lambda: ix_span("poverty_change_bhc_pp", max), "{:.2f}", ix)

    # --- dividend recycling -----------------------------------------------
    rc = "tax_composition/recycling_case.json"
    emit("genRecyclingDividendsBn",
         lambda: jload(rc)["recycled_dividends_bn"], "{:.1f}", rc)
    emit("genRecyclingGiniChangePp",
         lambda: jload(rc)["recycling_minus_central"]["gini_pp"], "{:.2f}", rc)

    # --- write -------------------------------------------------------------
    header = [
        "% values_generated.tex — machine-generated by analysis/emit_tex_values.py",
        f"% generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        "from canonical files under results/. DO NOT EDIT BY HAND.",
        "% \\GENMISSING marks values whose canonical source was absent at emit "
        "time; it errors at LaTeX build time by design.",
        "\\newcommand{\\GENMISSING}{\\errmessage{emit_tex_values: missing "
        "canonical result}}",
        "",
    ]
    OUT.write_text("\n".join(header + lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} macros)")
    if missing:
        print(f"WARNING: {len(missing)} macros missing canonical sources:")
        for m in missing:
            print("  " + m)


if __name__ == "__main__":
    main()
