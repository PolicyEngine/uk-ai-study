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

If a source file or key is missing, a source is INCOMPLETE (a checkpoint or
grid with fewer cells than the design, e.g. 40 of the 132 appendix grid
cells), or a source is STALE (its rows carry another certified data build,
or another code fingerprint than the script that writes it), the macro is
emitted as a visible placeholder (\\GENPENDING, a boxed "pending" that works
in text and math; whole table-row macros get \\GENPENDINGROW) with a
comment naming the source and reason. The paper therefore always builds,
and every pending value is visible on the page. ``--strict`` exits non-zero
if anything is pending (use it for the final build).

Signed twins. Prose that welds a sign or a direction onto a number ("+X",
"loses X") is wrong when the number changes sign. Every macro in SIGNED_TWINS
also gets a twin with "Signed" inserted before its unit suffix (for example
\\genIndexGiniMinPp -> \\genIndexGiniMinSignedPp), formatted with an
explicit sign for math mode. Twins of loss-framed macros (which negate a
change so that a loss prints as a positive number) carry the signed change
itself: negative is a loss.

Macros emitted (grouped by source):

  central/low/high presets (results/{central,low,high}.json,
      results/wage_margin_{central,pss}.json, results/central_ripple.json):
    \\genCentralCostBn \\genCentralPovBhcPp \\genCentralPovAhcPp
    \\genCentralGiniChangePp \\genBaselineGini \\genCentralDisplacedM
    \\genLowCostBn \\genHighCostBn \\genHighPovBhcPp
    \\genWageMarginCentralCostBn \\genWageMarginPssCostBn
    \\genWageMarginPssGiniChangePp \\genCentralRippleCostBn

  central 50-draw Monte Carlo (results/robustness/central_monte_carlo.json):
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
import re
import sys
import time
from pathlib import Path

import numpy as np
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
emitted: set[str] = set()

# ---------------------------------------------------------------------------
# Completeness and staleness guards
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))
from uk_ai_study.checkpoint import (  # noqa: E402
    certified_data_build_id,
    code_fingerprint,
    read_stamped_csv,
)


class Pending(Exception):
    """A source exists but is incomplete or stale."""


def _keys(*cols):
    return lambda d: len(set(map(tuple, d[list(cols)].itertuples(index=False))))


#: rel -> (expected count, counter, producing script). The counter returns the
#: number of distinct design cells/draws present; checkpoint CSVs are also
#: checked against the producing script's code fingerprint.
EXPECTED_CSV = {
    "jr16/grid.csv": (66, _keys("unemployment_pct", "wage_pct"), "replicate_jr16.py"),
    "appendix/grid_deciles_capital.csv": (
        132, _keys("unemployment_pct", "wage_pct", "capital_shock"), "appendix.py"),
    "appendix/decomposition_ci.csv": (10, _keys("decile"), None),
    # 30 rows, and only from the full 20-draw design
    "appendix/decile_ranking_sensitivity.csv": (
        30, lambda d: _keys("ranking", "decile")(d) if (d["draws"] == 20).all() else -1, None),
    "robustness/incidence_draws_five.csv": (250, _keys("seed", "family"), "monte_carlo_families.py"),
    "robustness/policy_draws.csv": (250, _keys("seed", "reform", "family"), "monte_carlo_families.py"),
    "robustness/central_cost_components_draws.csv": (50, _keys("seed"), "monte_carlo_families.py"),
    "robustness/duration_takeup_draws.csv": (40, _keys("variant", "seed"), "sensitivity_duration_takeup.py"),
    "robustness/duration_reform_draws.csv": (5, _keys("seed"), "sensitivity_duration_reforms.py"),
    "robustness/measured_wage_tier_draws.csv": (60, _keys("variant", "seed"), "sensitivity_wage_tier.py"),
    # 20 x (central, displacement, wage) + capital + 3 scaled runs x (20 + capital)
    "robustness/dividend_coverage_draws.csv": (124, _keys("run", "seed"), "sensitivity_dividend_coverage.py"),
    "tax_composition/composition_sims.csv": (10, _keys("unemployment_pct"), "tax_composition.py"),
    # the five families plus the two compound wage-axis variants
    "incidence/summary_five.csv": (7, _keys("family"), None),
}
#: summaries that are only valid with their (complete, current) checkpoint
REQUIRES = {
    "robustness/incidence_monte_carlo.json": ["robustness/incidence_draws_five.csv"],
    "robustness/central_monte_carlo.json": ["robustness/incidence_draws_five.csv"],
    "robustness/policy_monte_carlo.json": ["robustness/policy_draws.csv"],
    "robustness/central_cost_decomposition.json": ["robustness/central_cost_components_draws.csv"],
    "robustness/duration_takeup_sensitivity.json": ["robustness/duration_takeup_draws.csv",
                                                   "robustness/incidence_draws_five.csv"],
    "robustness/duration_reform_ranking.json": ["robustness/duration_reform_draws.csv"],
    "robustness/measured_wage_tier_sensitivity.json": ["robustness/measured_wage_tier_draws.csv"],
    "robustness/dividend_coverage.json": ["robustness/dividend_coverage_draws.csv"],
    "tax_composition/composition_grid.csv": ["tax_composition/composition_sims.csv"],
    "tax_composition/recycling_case.json": ["tax_composition/composition_sims.csv"],
}
_checked: dict[str, str | None] = {}


def _build_id() -> str:
    return certified_data_build_id()


def _check_build_id(rel: str, value) -> None:
    if value is not None and str(value) != _build_id():
        raise Pending(f"{rel} is from data build {value}, current is {_build_id()}")


def validate(rel: str) -> None:
    """Raise Pending/FileNotFoundError unless ``rel`` is complete and current."""
    if rel in _checked:
        if _checked[rel]:
            raise Pending(_checked[rel])
        return
    try:
        for dep in REQUIRES.get(rel, []):
            validate(dep)
        path = R / rel
        if rel.endswith(".csv"):
            d = read_stamped_csv(path)
            if "certified_data_build_id" in d.columns:
                for v in set(d["certified_data_build_id"]):
                    _check_build_id(rel, v)
            if rel in EXPECTED_CSV:
                n, count, script = EXPECTED_CSV[rel]
                have = count(d)
                if have != n or len(d) != n:
                    raise Pending(f"{rel} is incomplete: {have} of {n} cells ({len(d)} rows)")
                if script is not None:
                    want = code_fingerprint(ROOT / "analysis" / script)
                    got = set(d.get("code_fingerprint", pd.Series(["(none)"])).astype(str))
                    if got != {want}:
                        raise Pending(f"{rel} was written by other code ({sorted(got)}, "
                                      f"current {script} is {want})")
        else:
            d = json.loads(path.read_text())
            if isinstance(d, dict):
                _check_build_id(rel, d.get("certified_data_build_id"))
                _check_build_id(rel, (d.get("_metadata") or {}).get("certified_data_build_id"))
        _checked[rel] = None
    except Pending as exc:
        _checked[rel] = str(exc)
        raise


def jload(rel: str) -> dict:
    validate(rel)
    return json.loads((R / rel).read_text())


#: twin name -> sign multiplier (-1 for loss-framed macros that negate a
#: change). See the module docstring.
SIGNED_TWINS = {
    # "+X" welds
    "genIndexGiniMinPp": 1, "genIndexGiniMaxPp": 1, "genIndexPovBhcMinPp": 1,
    "genIndexPovBhcMaxPp": 1, "genCentralMcPovBhcMeanPp": 1, "genCentralMcGiniMeanPp": 1,
    "genCentralMcPovAhcMeanPp": 1, "genCentralMcRelPovBhcMeanPp": 1,
    "genDivCentralGiniMeanPp": 1, "genDivSpiGiniMeanPp": 1, "genDivCentralPovBhcMeanPp": 1,
    "genDivSpiPovBhcMeanPp": 1, "genDurSixMoPovBhcMeanPp": 1, "genTakeupPovBhcMeanPp": 1,
    "genGridGiniMinPp": 1, "genGridGiniMaxPp": 1, "genIncFamilyGiniMeanMinPp": 1,
    "genIncFamilyGiniMeanMaxPp": 1, "genIncFamilyPovBhcMeanMinPp": 1,
    "genIncFamilyPovBhcMeanMaxPp": 1, "genRecyclingGiniChangePp": 1,
    "genCentralGiniChangePp": 1, "genCentralPovAhcPp": 1,
    "genDecompMarketWorstPct": 1, "genDecompBenWorstPct": 1, "genDecompTaxWorstPct": 1,
    "genDecompDispWorstPct": 1, "genWageGainDOnePct": 1, "genWageGainDTenPct": 1,
    "genWageGainMinPct": 1, "genWageGainMaxPct": 1,
    # costs that can turn into gains
    "genCentralMcCostMeanBn": 1, "genIncFamilyCostMeanMinBn": 1, "genIncFamilyCostMeanMaxBn": 1,
    "genCompPhiHalfBn": -1, "genCompPhiOneBn": -1, "genCompPhiZeroBn": -1, "genCompPhiOneGapBn": -1,
    # loss-framed (negated) macros: twin = the signed change
    "genDecompCiMaxLossPct": -1, "genGridRevLossBn": -1, "genGridRevLossPct": -1,
    "genGridSevenZeroLossBn": -1, "genGridSevenZeroLossPct": -1, "genGridSevenThreeLossBn": -1,
    "genAppGridHarshDOnePct": -1, "genAppGridHarshDTenPct": -1,
    "genBSevenAiDTenPct": -1, "genBSevenUniDTenPct": -1,
    "genMixedNetEmpZeroPct": -1, "genMixedNetEmpOnePct": -1, "genRoneMcPovBhcAvertedPp": -1,
    "genLowGainBn": -1,
}
for _fam in ("Exposure", "Junior", "Compression", "Uniform", "Klein"):
    for _k in ("CostMeanBn", "PovBhcMeanPp", "PovAhcMeanPp", "GiniMeanPp"):
        SIGNED_TWINS[f"genInc{_fam}{_k}"] = 1
for _ref in ("Rone", "Rtwo", "Rthree"):
    SIGNED_TWINS[f"gen{_ref}McPovBhcMeanPp"] = 1
_UNIT = re.compile(r"(Pp|Pct|Bn|M|K|Gbp)$")


def twin_name(name: str) -> str:
    m_ = _UNIT.search(name)
    return name[: m_.start()] + "Signed" + m_.group(1) if m_ else name + "Signed"


def signed_fmt(fmt: str) -> str:
    m_ = re.fullmatch(r"\{:(,?)\.(\d)f\}", fmt)
    if not m_:
        return SIGNED2
    return "{:+" + m_.group(1) + "." + m_.group(2) + "f}"


#: macros that stand for whole table rows (placeholder must end the row)
TABLE_ROW_MACROS = {
    "genPresetRows", "genIndexRows", "genIncidenceRows", "genWageMarginRows",
    "genMixedRows", "genSurvivorRows", "genJobLossRows", "genPolicyRows", "genRippleRows",
}


def _placeholder(name: str) -> str:
    return "\\GENPENDINGROW" if name in TABLE_ROW_MACROS else "\\GENPENDING"


_NUMBER = re.compile(r"[+-]?\d[\d,]*(\.\d+)?")
#: name -> numeric value of every macro whose text is a plain number
values: dict[str, float] = {}


def abs_name(name: str) -> str:
    """Magnitude twin: "Signed" -> "Abs", else "Abs" before the unit suffix."""
    if "Signed" in name:
        return name.replace("Signed", "Abs")
    m_ = _UNIT.search(name)
    return name[: m_.start()] + "Abs" + m_.group(1) if m_ else name + "Abs"


def _line(name: str, text: str) -> None:
    # a value that rounds to zero prints without a sign ("-0.0" -> "0.0")
    if re.fullmatch(r"[+-]0(\.0+)?", text):
        text = text[1:]
    emitted.add(name)
    lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    if _NUMBER.fullmatch(text):
        values[name] = float(text.replace(",", ""))
        # Magnitude twin for every signed or negative number, so prose can
        # state the direction in words ("raises the cost by \genXAbsBn")
        # instead of welding a sign onto a pound figure. DIRECTIONS below
        # guards the words against a sign change.
        if text[0] in "+-":
            an = abs_name(name)
            if an not in emitted:
                emitted.add(an)
                lines.append(f"\\newcommand{{\\{an}}}{{{text.lstrip('+-')}}}")


#: Macros whose sign the prose states in words (with the Abs twin): +1 means
#: the text says the value is positive (a cost, a rise), -1 negative (a gain,
#: a fall). A sign change fails the emit, so the words are re-checked.
DIRECTIONS = {
    # channel split, 20 paired draws (Section 4.1, abstract, intro)
    "genChannelMcDispCostSignedBn": 1, "genChannelMcWageCostSignedBn": -1,
    "genChannelMcCapitalCostSignedBn": -1, "genChannelMcInteractionCostSignedBn": 1,
    "genChannelMcExclCapitalCostSignedBn": 1, "genChannelMcSumCostSignedBn": 1,
    "genChannelMcCentralCostSignedBn": 1,
    # central cost by component, 50 draws
    "genCostTotalSignedBn": 1, "genCostIncomeTaxSignedBn": -1, "genCostDividendTaxSignedBn": -1,
    "genCostSavingsTaxSignedBn": -1, "genCostOtherIncomeTaxSignedBn": 1,
    "genCostEmployeeNiSignedBn": 1, "genCostEmployerNiSignedBn": 1, "genCostUcSignedBn": 1,
    "genCostChildcareSignedBn": -1, "genCostExclEmployerNiSignedBn": -1,
    "genCostNarrowCashSignedBn": -1, "genCostBasisGapSignedBn": 1,
    # paired incidence differences
    "genPairKleinExpCostMean": 1, "genPairCompExpCostMean": 1, "genPairUniExpCostMean": -1,
    "genPairJunExpCostMean": -1,
    # duration hybrid: a net gain
    "genDurSixMoCostMeanBn": -1,
    # index sensitivity, grid, tax composition, dividend coverage
    "genIndexDsitDiffBn": -1, "genIndexEloundouDiffBn": -1,
    "genGridRevSevenZeroBn": -1, "genGridRevSevenZeroPct": -1, "genGridRevSevenThreeBn": 1,
    "genCompPhiZeroSignedBn": -1, "genCompPhiHalfSignedBn": -1, "genCompPhiOneSignedBn": -1,
    "genCompPhiOneGapSignedBn": -1, "genCompPhiZeroGovSignedBn": -1,
    "genDivDiffCostBn": 1, "genCovDivUniformDiffCostSignedBn": 1,
    "genCovDivBandedDiffCostSignedBn": 1, "genCovSavUniformDiffCostSignedBn": 1,
    # gender: household income falls for both sexes
    "genGenderFemaleIncomePct": -1, "genGenderMaleIncomePct": -1,
}


def check_directions() -> None:
    for name, sign in DIRECTIONS.items():
        if name not in values:
            if name in emitted:
                continue  # pending: already reported
            missing.append(f"{name} (DIRECTION: macro not emitted)")
        elif values[name] * sign <= 0:
            missing.append(f"{name} (DIRECTION: value {values[name]} contradicts the "
                           f"{'positive' if sign > 0 else 'negative'} direction the text states)")


def emit(name: str, value, fmt: str = "{:.1f}", source: str = ""):
    """Emit one macro; on any failure emit a visible placeholder."""
    if name in emitted:  # first definition wins (twins can pre-empt a name)
        return
    try:
        if callable(value):
            value = value()
        if not isinstance(value, str) and not np.isfinite(float(value)):
            raise Pending(f"non-finite value {value}")
        text = fmt.format(value) if not isinstance(value, str) else value
        _line(name, text)
        if name in SIGNED_TWINS and not isinstance(value, str):
            tw = twin_name(name)
            if tw not in emitted:
                _line(tw, signed_fmt(fmt).format(SIGNED_TWINS[name] * float(value)))
    except Exception as exc:  # noqa: BLE001 — every miss must surface, not abort
        kind = "PENDING" if isinstance(exc, Pending) else "MISSING"
        reason = str(exc).replace("\n", " ").replace(str(ROOT) + "/", "")
        missing.append(f"{name} ({source or 'unknown source'}: {kind} {reason})")
        names = [name] + ([twin_name(name)] if name in SIGNED_TWINS else [])
        for n in names:
            if n in emitted:
                continue
            emitted.add(n)
            lines.append(f"\\newcommand{{\\{n}}}{{{_placeholder(n)}}}"
                         f" % {kind}: {source} -> {reason}")

# ---------------------------------------------------------------------------
# Extended macros (Microcosm revision): every number the paper states, so the
# prose and tables are generated from results/ rather than transcribed.
# Signed values are emitted with an explicit sign ("+1.81", "-0.44") and are
# meant to be typeset in math mode ($\gen...$).
# ---------------------------------------------------------------------------
SIGNED2 = "{:+.2f}"
SIGNED1 = "{:+.1f}"


def csv(rel: str) -> pd.DataFrame:
    validate(rel)
    return read_stamped_csv(R / rel)


def row_tex(cells) -> str:
    return " & ".join(cells) + r" \\"


def m(x: float, fmt: str) -> str:
    """Number for a table cell, typeset in math mode so minus signs render."""
    return "$" + fmt.format(x) + "$"


def emit_extended() -> None:
    # --- presets table and preset prose -----------------------------------
    for rel, P in [("low.json", "Low"), ("central.json", "Central"), ("high.json", "High")]:
        emit(f"gen{P}DisplacedTwoM", lambda rel=rel: jload(rel)["displaced_weighted"] / 1e6, "{:.2f}", rel)
        emit(f"gen{P}PovBhcSignedPp", lambda rel=rel: jload(rel)["poverty_rate_change_bhc"] * 100, SIGNED2, rel)
        emit(f"gen{P}PovAhcSignedPp", lambda rel=rel: jload(rel)["poverty_rate_change_ahc"] * 100, SIGNED2, rel)
        emit(f"gen{P}GiniSignedPp",
             lambda rel=rel: (jload(rel)["gini_shocked"] - jload(rel)["gini_baseline"]) * 100, SIGNED2, rel)
        emit(f"gen{P}CostSignedBn", lambda rel=rel: jload(rel)["exchequer_cost"] / 1e9, SIGNED1, rel)
    emit("genLowGainBn", lambda: -jload("low.json")["exchequer_cost"] / 1e9, "{:.1f}", "low.json")

    def preset_rows():
        out = []
        for rel, label in [("low.json", "Low"), ("central.json", "Central"), ("high.json", "High")]:
            d = jload(rel)
            out.append(row_tex([
                label, f"{d['displaced_weighted'] / 1e6:.2f}",
                m(d["exchequer_cost"] / 1e9, "{:.1f}"),
                m(d["poverty_rate_change_bhc"] * 100, SIGNED2),
                m(d["poverty_rate_change_ahc"] * 100, SIGNED2),
                m((d["gini_shocked"] - d["gini_baseline"]) * 100, SIGNED2),
            ]))
        return "\n".join(out)
    emit("genPresetRows", preset_rows, "{}", "presets")

    # --- age bands ---------------------------------------------------------
    for rel, P in [("central.json", "Central"), ("central_youth_tilted.json", "Youth")]:
        for band, B in [("16-24", "Young"), ("35-44", "Prime"), ("55-64", "Older")]:
            emit(f"genAge{P}{B}SharePct",
                 lambda rel=rel, band=band: jload(rel)["age_band_displacement_share"][band] * 100,
                 "{:.1f}", rel)
        # the largest displaced age band, named in the text rather than assumed
        emit(f"genAge{P}LargestBand",
             lambda rel=rel: max(jload(rel)["age_band_displacement_share"].items(),
                                 key=lambda kv: kv[1])[0].replace("-+", " and over").replace("-", "--"),
             "{}", rel)
        emit(f"genAge{P}LargestSharePct",
             lambda rel=rel: max(jload(rel)["age_band_displacement_share"].values()) * 100, "{:.1f}", rel)

    # --- channel profiles (50-draw means) ---------------------------------
    t = "jr16/fig4_1_transition_by_decile.csv"
    emit("genTransDOnePct", lambda: csv(t).set_index("decile").loc[1, "share_transitioning"] * 100, "{:.2f}", t)
    emit("genTransDTenPct", lambda: csv(t).set_index("decile").loc[10, "share_transitioning"] * 100, "{:.2f}", t)
    emit("genTransMonotone",
         lambda: "strictly monotone" if csv(t)["share_transitioning"].is_monotonic_increasing else "not monotone",
         "{}", t)
    wg = "jr16/fig4_2_wage_gain_by_decile.csv"
    emit("genWageGainDOnePct", lambda: csv(wg).set_index("decile").loc[1, "pct_change_employment_income"] * 100, "{:.2f}", wg)
    emit("genWageGainDTenPct", lambda: csv(wg).set_index("decile").loc[10, "pct_change_employment_income"] * 100, "{:.2f}", wg)
    emit("genWageGainMinPct", lambda: csv(wg)["pct_change_employment_income"].min() * 100, "{:.2f}", wg)
    emit("genWageGainMaxPct", lambda: csv(wg)["pct_change_employment_income"].max() * 100, "{:.2f}", wg)
    cp = "jr16/fig4_3_capital_by_decile.csv"
    emit("genCapTopSharePct", lambda: csv(cp).set_index("decile").loc[10, "share_of_all_capital_income"] * 100, "{:.1f}", cp)
    emit("genCapTopTwoSharePct",
         lambda: csv(cp).set_index("decile").loc[[9, 10], "share_of_all_capital_income"].sum() * 100, "{:.1f}", cp)
    emit("genCapBottomSharePct", lambda: csv(cp).set_index("decile").loc[1, "share_of_all_capital_income"] * 100, "{:.1f}", cp)
    emit("genCapTopBn", lambda: csv(cp).set_index("decile").loc[10, "aggregate_capital_income"] / 1e9, "{:.1f}", cp)
    emit("genCapTotalBn", lambda: csv(cp)["aggregate_capital_income"].sum() / 1e9, "{:.0f}", cp)

    # --- household mediation (seed-0 decomposition; 20-draw means) ---------
    dc = "jr16/fig4_4_decomposition.csv"
    for d, D in [(1, "One"), (5, "Five"), (10, "Ten")]:
        for col, C in [("market_income", "Market"), ("disposable_income", "Disp"),
                       ("benefits", "Ben"), ("tax_and_contributions", "Tax")]:
            emit(f"genDecomp{C}D{D}Pct",
                 lambda d=d, col=col: csv(dc).set_index("decile").loc[d, col] * 100, "{:.1f}", dc)
            emit(f"genDecomp{C}D{D}SignedPct",
                 lambda d=d, col=col: csv(dc).set_index("decile").loc[d, col] * 100, SIGNED1, dc)
    def worst_decile():
        d = csv(dc).set_index("decile")
        return int(d["disposable_income"].idxmin())
    emit("genDecompWorstDecile", worst_decile, "{:d}", dc)
    for col, C in [("market_income", "Market"), ("disposable_income", "Disp"),
                   ("benefits", "Ben"), ("tax_and_contributions", "Tax")]:
        emit(f"genDecomp{C}WorstPct",
             lambda col=col: csv(dc).set_index("decile").loc[worst_decile(), col] * 100, "{:.1f}", dc)
    emit("genDecompAbsorbedWorstPct",
         lambda: 100 * (1 - csv(dc).set_index("decile").loc[worst_decile(), "disposable_income"]
                        / csv(dc).set_index("decile").loc[worst_decile(), "market_income"]), "{:.0f}", dc)
    emit("genDecompResidualMaxPct", lambda: csv(dc)["residual"].abs().max() * 100, "{:.1f}", dc)
    emit("genDecompAbsorbedDTenPct",
         lambda: 100 * (1 - csv(dc).set_index("decile").loc[10, "disposable_income"]
                        / csv(dc).set_index("decile").loc[10, "market_income"]), "{:.0f}", dc)
    ci = "appendix/decomposition_ci.csv"
    emit("genDecompCiDOnePct", lambda: -csv(ci).set_index("decile").loc[1, "disposable_change_pct"], "{:.2f}", ci)
    emit("genDecompCiDTenPct", lambda: -csv(ci).set_index("decile").loc[10, "disposable_change_pct"], "{:.2f}", ci)
    emit("genDecompCiMaxLossPct", lambda: -csv(ci)["disposable_change_pct"].min(), "{:.2f}", ci)
    emit("genDecompCiDOneSignedPct", lambda: csv(ci).set_index("decile").loc[1, "disposable_change_pct"], SIGNED2, ci)
    emit("genDecompCiDTenSignedPct", lambda: csv(ci).set_index("decile").loc[10, "disposable_change_pct"], SIGNED2, ci)
    emit("genDecompCiDTenLoPct", lambda: csv(ci).set_index("decile").loc[10, "ci_low"], SIGNED2, ci)
    emit("genDecompCiDTenHiPct", lambda: csv(ci).set_index("decile").loc[10, "ci_high"], SIGNED2, ci)
    emit("genDecompCiDOneLoPct", lambda: csv(ci).set_index("decile").loc[1, "ci_low"], SIGNED2, ci)
    emit("genDecompCiDOneHiPct", lambda: csv(ci).set_index("decile").loc[1, "ci_high"], SIGNED2, ci)
    emit("genDecompCiMaxLossDecile",
         lambda: int(csv(ci).set_index("decile")["disposable_change_pct"].idxmin()), "{:d}", ci)

    # --- central Monte Carlo: relative line and AHC ------------------------
    mc = "robustness/central_monte_carlo.json"
    emit("genCentralMcRelPovBhcMeanPp", lambda: jload(mc)["relative_poverty_change_bhc_pp"]["mean"], "{:.2f}", mc)
    emit("genCentralMcRelPovBhcSdPp", lambda: jload(mc)["relative_poverty_change_bhc_pp"]["sd"], "{:.2f}", mc)
    emit("genCentralMcPovAhcMeanPp", lambda: jload(mc)["poverty_change_ahc_pp"]["mean"], "{:.2f}", mc)
    emit("genCentralMcPovAhcSdPp", lambda: jload(mc)["poverty_change_ahc_pp"]["sd"], "{:.2f}", mc)
    def central_displaced(agg):
        d = csv("robustness/incidence_draws_five.csv")
        return agg(d[d.family == "exposure"]["displaced_weighted_m"])
    emit("genCentralMcDisplacedMinM", lambda: central_displaced(min), "{:.2f}", "robustness/incidence_draws_five.csv")
    emit("genCentralMcDisplacedMaxM", lambda: central_displaced(max), "{:.2f}", "robustness/incidence_draws_five.csv")
    emit("genCentralMcDisplacedMeanM", lambda: float(np.mean(list(
        csv("robustness/incidence_draws_five.csv").query("family == 'exposure'")["displaced_weighted_m"]))),
        "{:.2f}", "robustness/incidence_draws_five.csv")
    emit("genCentralMcCostCv", lambda: 100 * jload(mc)["exchequer_cost_bn"]["sd"]
         / jload(mc)["exchequer_cost_bn"]["mean"], "{:.0f}", mc)
    emit("genCentralMcCostHalfBn", lambda: jload(mc)["exchequer_cost_bn"]["mean"] / 2, "{:.1f}", mc)
    # central exposure-family mean on the seeds the duration/take-up stage uses
    def dur_central():
        d = csv("robustness/incidence_draws_five.csv")
        n = int(jload("robustness/duration_takeup_sensitivity.json")["n_draws"])
        return float(d[(d.family == "exposure") & (d.seed < n)]["exchequer_cost_bn"].mean())
    emit("genDurCentralCostMeanBn", dur_central, "{:.1f}", "robustness/duration_takeup_sensitivity.json")
    emit("genDurCentralCostHalfBn", lambda: dur_central() / 2, "{:.1f}",
         "robustness/duration_takeup_sensitivity.json")

    # --- poverty decomposition --------------------------------------------
    pd_ = "robustness/poverty_decomposition.json"
    emit("genPovNetPp", lambda: jload(pd_)["net_poverty_change_pp"], "{:.2f}", pd_)
    emit("genPovInflowM", lambda: jload(pd_)["gross_inflow_persons"] / 1e6, "{:.2f}", pd_)
    emit("genPovInflowPp", lambda: jload(pd_)["gross_inflow_pp"], "{:.2f}", pd_)
    emit("genPovOutflowM", lambda: jload(pd_)["gross_outflow_persons"] / 1e6, "{:.2f}", pd_)
    emit("genPovOutflowPp", lambda: jload(pd_)["gross_outflow_pp"], "{:.2f}", pd_)
    emit("genPovInDisplacedHhPct", lambda: 100 * jload(pd_)["newly_poor_share_in_displaced_worker_household"], "{:.0f}", pd_)
    emit("genPovTopDecilePct",
         lambda: 100 * jload(pd_)["newly_poor_decile_shares_baseline_equiv_income"]["10"], "{:.0f}", pd_)
    emit("genPovEntrantRatio", lambda: jload(pd_)["newly_poor_median_baseline_income_to_line_ratio"], "{:.2f}", pd_)
    emit("genPovEntrantShortfallPct", lambda: jload(pd_)["newly_poor_median_postshock_shortfall_pct_of_line"], "{:.0f}", pd_)
    emit("genPovSoleEarnerPct",
         lambda: 100 * (1 - jload(pd_)["newly_poor_share_displaced_hh_with_other_earner"]), "{:.0f}", pd_)
    emit("genPovBaselineBhcPct", lambda: 100 * jload(pd_)["poverty_rate_bhc_baseline"], "{:.1f}", pd_)

    # --- index sensitivity -------------------------------------------------
    ix = "robustness/index_sensitivity_full.json"
    labels = [("c_aioe", "C-AIOE (central)"), ("felten_aioe", "Felten AIOE"),
              ("eloundou_beta", "Eloundou beta"), ("dsit_aioe", "DSIT AIOE"), ("dsit_llm", "DSIT LLM")]

    def index_rows():
        d = jload(ix)["results"]
        return "\n".join(row_tex([
            lab, m(d[k]["exchequer_cost_bn"], "{:.1f}"), m(d[k]["poverty_change_bhc_pp"], SIGNED2),
            m(d[k]["gini_change_pp"], SIGNED2), f"{d[k]['transition_share_decile1_pct']:.2f}",
            f"{d[k]['transition_share_decile10_pct']:.2f}"]) for k, lab in labels)
    emit("genIndexRows", index_rows, "{}", ix)

    def ix_dev():
        d = jload(ix)["results"]
        c = d["c_aioe"]["exchequer_cost_bn"]
        return max((v["exchequer_cost_bn"] / c - 1) * 100 for v in d.values())
    emit("genIndexMaxDevPct", ix_dev, "{:+.1f}", ix)
    emit("genIndexEloundouDiffBn", lambda: jload(ix)["results"]["eloundou_beta"]["exchequer_cost_bn"]
         - jload(ix)["results"]["c_aioe"]["exchequer_cost_bn"], SIGNED1, ix)
    emit("genIndexDsitDiffBn", lambda: jload(ix)["results"]["dsit_aioe"]["exchequer_cost_bn"]
         - jload(ix)["results"]["c_aioe"]["exchequer_cost_bn"], SIGNED1, ix)

    def ix_ratio(agg):
        d = jload(ix)["results"]
        return agg(v["transition_share_decile10_pct"] / v["transition_share_decile1_pct"] for v in d.values())
    emit("genIndexRatioMin", lambda: ix_ratio(min), "{:.1f}", ix)
    emit("genIndexRatioMax", lambda: ix_ratio(max), "{:.1f}", ix)
    emit("genIndexAllSurvive",
         lambda: "yes" if all(v["all_three_survive"] for v in jload(ix)["results"].values()) else "no", "{}", ix)

    # --- incidence families ------------------------------------------------
    imc = "robustness/incidence_monte_carlo.json"
    s5 = "incidence/summary_five.csv"
    fam_labels = [("exposure", "Exposure-proportional"), ("junior", "Junior-concentrated"),
                  ("compression", "Expertise-compression"), ("uniform", "Uniform"),
                  ("klein_top_loaded", "Klein-anchored stress test")]
    for fam, Fam in FAMILY_MACRO.items():
        emit(f"genInc{Fam}PovBhcSdPp", lambda fam=fam: jload(imc)[fam]["poverty_change_bhc_pp"]["sd"], "{:.2f}", imc)
        emit(f"genInc{Fam}PovAhcMeanPp", lambda fam=fam: jload(imc)[fam]["poverty_change_ahc_pp"]["mean"], "{:.2f}", imc)
        emit(f"genInc{Fam}GiniSdPp", lambda fam=fam: jload(imc)[fam]["gini_change_pp"]["sd"], "{:.2f}", imc)

    def incidence_rows():
        d = jload(imc)
        draws = csv("robustness/incidence_draws_five.csv")
        out = []
        for fam, lab in fam_labels:
            f = d[fam]
            out.append(row_tex([
                lab, f"{draws.query('family == @fam')['displaced_weighted_m'].mean():.2f}",
                f"{f['exchequer_cost_bn']['mean']:.1f} $\\pm$ {f['exchequer_cost_bn']['sd']:.1f}",
                f"${f['poverty_change_bhc_pp']['mean']:+.2f}$ $\\pm$ {f['poverty_change_bhc_pp']['sd']:.2f}",
                f"${f['poverty_change_ahc_pp']['mean']:+.2f}$ $\\pm$ {f['poverty_change_ahc_pp']['sd']:.2f}",
                f"${f['gini_change_pp']['mean']:+.2f}$ $\\pm$ {f['gini_change_pp']['sd']:.2f}",
            ]))
        return "\n".join(out)
    emit("genIncidenceRows", incidence_rows, "{}", imc)

    # four stylised families (the Klein stress test excluded), 50-draw means
    def stylised(key, agg):
        d = jload(imc)
        return agg(d[f][key]["mean"] for f in ("exposure", "junior", "compression", "uniform"))
    emit("genIncStylisedPovBhcMeanMinSignedPp", lambda: stylised("poverty_change_bhc_pp", min), SIGNED2, imc)
    emit("genIncStylisedPovBhcMeanMaxSignedPp", lambda: stylised("poverty_change_bhc_pp", max), SIGNED2, imc)
    # mean weighted number displaced per family over the 50 paired draws
    for fam, Fam in FAMILY_MACRO.items():
        emit(f"genInc{Fam}DisplacedMeanM",
             lambda fam=fam: csv("robustness/incidence_draws_five.csv").query("family == @fam")[
                 "displaced_weighted_m"].mean(), "{:.2f}", "robustness/incidence_draws_five.csv")

    def stylised_span():
        d = jload(imc)
        means = [d[f]["exchequer_cost_bn"]["mean"] for f in ("exposure", "junior", "compression", "uniform")]
        return max(means) - min(means)
    emit("genIncStylisedCostSpanBn", stylised_span, "{:.1f}", imc)
    emit("genIncKleinTransDOnePct",
         lambda: jload("incidence/klein_top_loaded.json")["decile_transition_share_pct"]["1"], "{:.2f}",
         "incidence/klein_top_loaded.json")
    emit("genIncKleinTransDTenPct",
         lambda: jload("incidence/klein_top_loaded.json")["decile_transition_share_pct"]["10"], "{:.2f}",
         "incidence/klein_top_loaded.json")

    # paired draw-level differences (shared seeds), normal 95% interval
    dr = "robustness/incidence_draws_five.csv"

    def paired(a, b, col):
        d = csv(dr)
        x = d[d.family == a].set_index("seed")[col]
        y = d[d.family == b].set_index("seed")[col]
        diff = (x - y).dropna()
        mean = float(diff.mean())
        half = 1.96 * float(diff.std(ddof=1)) / len(diff) ** 0.5
        return mean, mean - half, mean + half

    for name, a, b, col in [
        ("CompExpPov", "compression", "exposure", "poverty_change_bhc_pp"),
        ("UniExpPov", "uniform", "exposure", "poverty_change_bhc_pp"),
        ("UniExpCost", "uniform", "exposure", "exchequer_cost_bn"),
        ("JunExpCost", "junior", "exposure", "exchequer_cost_bn"),
        ("JunExpPov", "junior", "exposure", "poverty_change_bhc_pp"),
        ("CompExpCost", "compression", "exposure", "exchequer_cost_bn"),
        ("UniExpGini", "uniform", "exposure", "gini_change_pp"),
        ("KleinCompGini", "klein_top_loaded", "compression", "gini_change_pp"),
        ("KleinExpCost", "klein_top_loaded", "exposure", "exchequer_cost_bn"),
    ]:
        for i, S in enumerate(("Mean", "Lo", "Hi")):
            emit(f"genPair{name}{S}", lambda a=a, b=b, col=col, i=i: paired(a, b, col)[i], SIGNED2, dr)
        emit(f"genPair{name}Sig",
             lambda a=a, b=b, col=col: "distinguishable" if (paired(a, b, col)[1] > 0 or paired(a, b, col)[2] < 0)
             else "indistinguishable", "{}", dr)

    # --- wage margin and adjustment-margin tables -------------------------
    for rel, P in [("wage_margin_central.json", "WageMarginCentral"), ("wage_margin_pss.json", "WageMarginPss")]:
        emit(f"gen{P}PovBhcSignedPp", lambda rel=rel: jload(rel)["poverty_rate_change_bhc"] * 100, SIGNED2, rel)
        emit(f"gen{P}PovAhcSignedPp", lambda rel=rel: jload(rel)["poverty_rate_change_ahc"] * 100, SIGNED2, rel)
        emit(f"gen{P}GiniSignedPp",
             lambda rel=rel: (jload(rel)["gini_shocked"] - jload(rel)["gini_baseline"]) * 100, SIGNED2, rel)
    emit("genGrossCutSharePct", lambda: jload("wage_margin_central.json")["gross_cut_share_target"] * 100,
         "{:.2f}", "wage_margin_central.json")
    emit("genGrossCutShareRaw", lambda: jload("wage_margin_central.json")["gross_cut_share_target"],
         "{:.5f}", "wage_margin_central.json")

    def wagemargin_rows():
        out = []
        for rel, lab in [("central.json", "Central displacement"),
                         ("wage_margin_central.json", "Wage margin (C-AIOE gradient)"),
                         ("wage_margin_pss.json", "Wage margin (PSS gradient)")]:
            d = jload(rel)
            out.append(row_tex([
                lab, m(d["exchequer_cost"] / 1e9, "{:.1f}"), m(d["poverty_rate_change_bhc"] * 100, SIGNED2),
                m(d["poverty_rate_change_ahc"] * 100, SIGNED2),
                m((d["gini_shocked"] - d["gini_baseline"]) * 100, SIGNED2)]))
        return "\n".join(out)
    emit("genWageMarginRows", wagemargin_rows, "{}", "wage margin")

    ma = "robustness/mixed_adjustment.csv"
    sw = "robustness/survivor_wage_grid.csv"
    lam_names = [("mixed_lambda_0", "Zero", r"$\lambda=0$"), ("mixed_lambda_0.25", "Quarter", r"$\lambda=.25$"),
                 ("mixed_lambda_0.5", "Half", r"$\lambda=.50$"), ("mixed_lambda_0.75", "ThreeQ", r"$\lambda=.75$"),
                 ("mixed_lambda_1", "One", r"$\lambda=1$")]
    sw_names = [("survivor_wage_-0.050", "MinusFive", r"$-5.0\%$"), ("survivor_wage_-0.026", "MinusTwo", r"$-2.6\%$"),
                ("survivor_wage_+0.000", "Zero", r"$0.0\%$"), ("survivor_wage_+0.026", "PlusTwo", r"$+2.6\%$"),
                ("survivor_wage_+0.050", "PlusFive", r"$+5.0\%$")]

    def adj_rows(rel, names):
        d = csv(rel).set_index("scenario")
        return "\n".join(row_tex([
            lab, f"{d.loc[k, 'displaced_weighted_m']:.2f}", m(d.loc[k, "exchequer_cost_bn"], "{:.1f}"),
            m(d.loc[k, "poverty_change_bhc_pp"], SIGNED2), m(d.loc[k, "gini_change_pp"], SIGNED2)])
            for k, _, lab in names)
    emit("genMixedRows", lambda: adj_rows(ma, lam_names), "{}", ma)
    emit("genSurvivorRows", lambda: adj_rows(sw, sw_names), "{}", sw)
    for rel, names, pre in [(ma, lam_names, "Mixed"), (sw, sw_names, "Survivor")]:
        for k, N, _ in names:
            emit(f"gen{pre}{N}CostBn", lambda rel=rel, k=k: csv(rel).set_index("scenario").loc[k, "exchequer_cost_bn"],
                 "{:.1f}", rel)
            emit(f"gen{pre}{N}PovBhcPp",
                 lambda rel=rel, k=k: csv(rel).set_index("scenario").loc[k, "poverty_change_bhc_pp"], SIGNED2, rel)
            emit(f"gen{pre}{N}GiniPp",
                 lambda rel=rel, k=k: csv(rel).set_index("scenario").loc[k, "gini_change_pp"], SIGNED2, rel)
    emit("genMixedNetEmpZeroPct",
         lambda: -csv(ma).set_index("scenario").loc["mixed_lambda_0", "net_employment_income_change_pct"], "{:.2f}", ma)
    emit("genMixedNetEmpOnePct",
         lambda: -csv(ma).set_index("scenario").loc["mixed_lambda_1", "net_employment_income_change_pct"], "{:.2f}", ma)

    # --- shock-size grid ---------------------------------------------------
    gr = "jr16/grid.csv"

    def cell(u, w, col):
        g = csv(gr)
        return float(g[(g.unemployment_pct == u) & (g.wage_pct == w)][col].iloc[0])
    emit("genGridDispMaxPct", lambda: cell(0, 5, "avg_disposable_income_change_pct"), SIGNED1, gr)
    emit("genGridDispMinPct", lambda: cell(10, 0, "avg_disposable_income_change_pct"), SIGNED1, gr)
    emit("genGridReceiptsBn", lambda: cell(0, 5, "net_revenue_change_bn") * 100
         / cell(0, 5, "net_revenue_change_pct_of_receipts"), "{:.0f}", gr)
    emit("genGridRevGainBn", lambda: cell(0, 5, "net_revenue_change_bn"), "{:.1f}", gr)
    emit("genGridRevGainPct", lambda: cell(0, 5, "net_revenue_change_pct_of_receipts"), "{:.1f}", gr)
    emit("genGridRevLossBn", lambda: -cell(10, 0, "net_revenue_change_bn"), "{:.1f}", gr)
    emit("genGridRevLossPct", lambda: -cell(10, 0, "net_revenue_change_pct_of_receipts"), "{:.1f}", gr)
    emit("genGridSevenZeroLossBn", lambda: -cell(7, 0, "net_revenue_change_bn"), "{:.1f}", gr)
    emit("genGridSevenZeroLossPct", lambda: -cell(7, 0, "net_revenue_change_pct_of_receipts"), "{:.1f}", gr)
    emit("genGridSevenThreeLossBn", lambda: -cell(7, 3, "net_revenue_change_bn"), "{:.1f}", gr)
    # central 7 per cent / 2.6 per cent cell, linearly interpolated between
    # the 2 and 3 per cent wage columns
    emit("genGridCentralCashCostBn",
         lambda: -(cell(7, 2, "net_revenue_change_bn")
                   + 0.6 * (cell(7, 3, "net_revenue_change_bn") - cell(7, 2, "net_revenue_change_bn"))),
         "{:.0f}", gr)
    emit("genGridCentralCashChangeBn",
         lambda: cell(7, 2, "net_revenue_change_bn")
         + 0.6 * (cell(7, 3, "net_revenue_change_bn") - cell(7, 2, "net_revenue_change_bn")),
         SIGNED1, gr)
    for u, w in [(3, 0), (4, 0), (7, 0), (7, 3), (7, 4), (7, 5)]:
        emit(f"genGridDisp{['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten'][u]}"
             f"{['Zero','One','Two','Three','Four','Five'][w]}Pct",
             lambda u=u, w=w: cell(u, w, "avg_disposable_income_change_pct"), SIGNED2, gr)
        emit(f"genGridRev{['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten'][u]}"
             f"{['Zero','One','Two','Three','Four','Five'][w]}Bn",
             lambda u=u, w=w: cell(u, w, "net_revenue_change_bn"), SIGNED1, gr)
        emit(f"genGridRev{['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten'][u]}"
             f"{['Zero','One','Two','Three','Four','Five'][w]}Pct",
             lambda u=u, w=w: cell(u, w, "net_revenue_change_pct_of_receipts"), SIGNED1, gr)
    emit("genGridCapOnlyRevBn", lambda: cell(0, 0, "net_revenue_change_bn"), "{:.1f}", gr)
    emit("genGridCapOnlyDispPct", lambda: cell(0, 0, "avg_disposable_income_change_pct"), SIGNED1, gr)
    emit("genGridGiniPositiveCells", lambda: int((csv(gr)["gini_change_pp"] > 0).sum()), "{:d}", gr)

    def argmin_cell(col):
        g = csv(gr)
        r = g.loc[g[col].idxmin()]
        return f"{int(r.unemployment_pct)} per cent displacement, {int(r.wage_pct)} per cent wage growth"
    emit("genGridGiniMinCell", lambda: argmin_cell("gini_change_pp"), "{}", gr)
    emit("genGridDispZeroLine",
         lambda: cell(7, 3, "avg_disposable_income_change_pct"), SIGNED1, gr)

    gd = "appendix/grid_deciles_capital.csv"

    def gdf(capital):
        g = csv(gd)
        return g[g.capital_shock == capital]
    emit("genAppGridDecileMaxPct",
         lambda: gdf(True)[[f"decile{d}_change_pct" for d in range(1, 11)]].max().max(), SIGNED1, gd)
    emit("genAppGridDecileMinPct",
         lambda: gdf(True)[[f"decile{d}_change_pct" for d in range(1, 11)]].min().min(), SIGNED1, gd)
    emit("genAppGridHarshDOnePct",
         lambda: -float(gdf(True).query("unemployment_pct == 10 and wage_pct == 0")["decile1_change_pct"].iloc[0]),
         "{:.1f}", gd)
    emit("genAppGridHarshDTenPct",
         lambda: -float(gdf(True).query("unemployment_pct == 10 and wage_pct == 0")["decile10_change_pct"].iloc[0]),
         "{:.1f}", gd)
    emit("genAppGridNoCapGiniPositive", lambda: int((gdf(False)["gini_change_pp"] > 1e-9).sum()), "{:d}", gd)
    emit("genAppGridNoCapGiniMaxPp", lambda: gdf(False)["gini_change_pp"].max(), "{:.2f}", gd)
    emit("genAppGridNoCapGiniMinPp", lambda: gdf(False)["gini_change_pp"].min(), SIGNED2, gd)

    # --- appendix descriptives --------------------------------------------
    bd = "appendix/baseline_distributions.csv"
    for col, N in [("b1_market_income_less_capital", "Market"), ("b2_capital_income", "Capital"),
                   ("b3_benefits", "Benefits"), ("b4_tax_and_ni", "TaxNi"), ("b5_disposable_income", "Disposable")]:
        emit(f"genBase{N}Bn", lambda col=col: csv(bd)[col].sum(), "{:,.0f}", bd)
    jl = "appendix/job_loss_by_major_group.csv"

    def jobloss_rows():
        d = csv(jl)
        titles = {1: "Managers, directors and senior officials", 2: "Professional occupations",
                  3: "Associate professional occupations", 4: "Administrative and secretarial occupations",
                  5: "Skilled trades occupations", 6: "Caring, leisure and other service occupations",
                  7: "Sales and customer service occupations", 8: "Process, plant and machine operatives",
                  9: "Elementary occupations"}
        return "\n".join(row_tex([f"{int(r.soc_major_group)} {titles[int(r.soc_major_group)]}",
                                  m(r.c_aioe, "{:.2f}"), f"{r.employment_m:.2f}", f"{r.job_loss_pct:.1f}"])
                         for r in d.itertuples())
    emit("genJobLossRows", jobloss_rows, "{}", jl)
    emit("genJobLossEmployeesM", lambda: csv(jl)["employment_m"].sum(), "{:.1f}", jl)
    b7 = "appendix/b7_uniform_vs_ai_decile.csv"
    emit("genBSevenAiDTenPct", lambda: -csv(b7).set_index("decile").loc[10, "ai_shock_pct"], "{:.1f}", b7)
    emit("genBSevenUniDTenPct", lambda: -csv(b7).set_index("decile").loc[10, "uniform_shock_pct"], "{:.1f}", b7)
    for dec, D in [(8, "Eight"), (9, "Nine")]:
        emit(f"genBSevenAiD{D}SignedPct",
             lambda dec=dec: csv(b7).set_index("decile").loc[dec, "ai_shock_pct"], SIGNED1, b7)
        emit(f"genBSevenUniD{D}SignedPct",
             lambda dec=dec: csv(b7).set_index("decile").loc[dec, "uniform_shock_pct"], SIGNED1, b7)

    # --- gender ------------------------------------------------------------
    ge = "appendix/gender_incidence.json"
    for key, N, fmt, scale in [("female_mean_exposure", "FemaleExposure", "{:.2f}", 1),
                               ("male_mean_exposure", "MaleExposure", "{:.2f}", 1),
                               ("female_displacement_rate_mean", "FemaleRatePct", "{:.1f}", 100),
                               ("male_displacement_rate_mean", "MaleRatePct", "{:.1f}", 100),
                               ("female_share_of_employment", "FemaleEmpSharePct", "{:.1f}", 100),
                               ("female_share_of_displaced_mean", "FemaleDispSharePct", "{:.1f}", 100),
                               ("female_income_change_pct", "FemaleIncomePct", "{:.1f}", 1),
                               ("male_income_change_pct", "MaleIncomePct", "{:.1f}", 1)]:
        emit(f"genGender{N}", lambda key=key, scale=scale: jload(ge)[key] * scale, fmt, ge)

    # --- caseloads ---------------------------------------------------------
    cs = "caseloads/summary.csv"

    def cl(scn, col):
        return float(csv(cs).set_index("scenario").loc[scn, col])
    emit("genCaseloadNewUcBenunitsK", lambda: cl("central", "new_uc_benunits_thousands"), "{:,.0f}", cs)
    emit("genCaseloadExitUcBenunitsK", lambda: cl("central", "exiting_uc_benunits_thousands"), "{:,.0f}", cs)
    emit("genCaseloadNewUcHouseholdsK", lambda: cl("central", "new_uc_households_thousands"), "{:,.0f}", cs)
    emit("genCaseloadUcBaselineBn", lambda: cl("central", "uc_spend_baseline_bn"), "{:.1f}", cs)
    emit("genCaseloadHousingBn", lambda: jload("caseloads/central.json")["uc_housing_element_gross_entitlement_change_bn"],
         "{:.1f}", "caseloads/central.json")
    emit("genCaseloadSharePct", lambda: 100 * cl("central", "uc_spend_change_bn") / 334.0, "{:.1f}", cs)
    for scn, N in [("high", "High"), ("low", "Low"), ("incidence_junior", "Junior"), ("incidence_uniform", "Uniform")]:
        emit(f"genCaseload{N}NetK", lambda scn=scn: cl(scn, "net_new_uc_benunits_thousands"), "{:,.0f}", cs)
        emit(f"genCaseload{N}SpendBn", lambda scn=scn: cl(scn, "uc_spend_change_bn"), "{:.1f}", cs)

    # --- policy ------------------------------------------------------------
    ps = "policy/summary.csv"
    pol_labels = [("R1_wage_insurance", "R1 Wage insurance"), ("R2_uc_circuit_breaker", r"R2 UC allowance $+20\%$"),
                  ("R3_cap_suspension_taper_cut", r"R3 Cap suspension $+$ taper cut")]

    def policy_rows():
        d = csv(ps)
        d = d[d.family == "exposure"].set_index("reform")
        return "\n".join(row_tex([
            lab, f"{d.loc[k, 'gross_cost_bn']:.1f}", m(d.loc[k, "poverty_change_bhc_pp"], SIGNED2),
            m(d.loc[k, "poverty_change_ahc_pp"], SIGNED2), m(d.loc[k, "gini_change_pp"], SIGNED2),
            f"{d.loc[k, 'cost_per_pp_bhc_bn']:.1f}", f"{d.loc[k, 'cost_per_pp_ahc_bn']:.1f}"])
            for k, lab in pol_labels)
    emit("genPolicyRows", policy_rows, "{}", ps)

    def pol(reform, family, col):
        d = csv(ps)
        return float(d[(d.reform == reform) & (d.family == family)][col].iloc[0])
    emit("genRoneMeanTransferGbp", lambda: pol("R1_wage_insurance", "exposure", "mean_transfer_per_displaced_gbp"),
         "{:,.0f}", ps)
    emit("genRoneDisplacedM", lambda: pol("R1_wage_insurance", "exposure", "displaced_weighted_m"), "{:.2f}", ps)
    emit("genRoneJuniorCostPerPpAhcBn", lambda: pol("R1_wage_insurance", "junior", "cost_per_pp_ahc_bn"), "{:.1f}", ps)
    emit("genRoneUniformCostPerPpAhcBn", lambda: pol("R1_wage_insurance", "uniform", "cost_per_pp_ahc_bn"), "{:.1f}", ps)
    emit("genRoneCostMinFamBn", lambda: min(pol("R1_wage_insurance", f, "gross_cost_bn")
                                            for f in ("exposure", "junior", "uniform")), "{:.1f}", ps)
    emit("genRoneCostMaxFamBn", lambda: max(pol("R1_wage_insurance", f, "gross_cost_bn")
                                            for f in ("exposure", "junior", "uniform")), "{:.1f}", ps)
    emit("genRoneToRtwoPovRatio", lambda: pol("R1_wage_insurance", "exposure", "poverty_change_bhc_pp")
         / pol("R2_uc_circuit_breaker", "exposure", "poverty_change_bhc_pp"), "{:.1f}", ps)
    emit("genRoneToRtwoCostRatio", lambda: pol("R1_wage_insurance", "exposure", "gross_cost_bn")
         / pol("R2_uc_circuit_breaker", "exposure", "gross_cost_bn"), "{:.1f}", ps)
    emit("genRoneToRthreeCostRatio", lambda: pol("R1_wage_insurance", "exposure", "gross_cost_bn")
         / pol("R3_cap_suspension_taper_cut", "exposure", "gross_cost_bn"), "{:.1f}", ps)
    pmc = "robustness/policy_monte_carlo.json"
    emit("genRoneMcPovBhcAvertedPp", lambda: -jload(pmc)["R1_wage_insurance_exposure"]["poverty_change_bhc_pp"]["mean"],
         "{:.2f}", pmc)
    emit("genRoneMcPovBhcAvertedSdPp", lambda: jload(pmc)["R1_wage_insurance_exposure"]["poverty_change_bhc_pp"]["sd"],
         "{:.2f}", pmc)
    dr_ = "robustness/duration_reform_ranking.json"
    emit("genDurRoneCostPerPpBn", lambda: jload(dr_)["r1_cost_per_pp_bn"]["mean"], "{:.0f}", dr_)
    emit("genDurRoneCostPerPpSdBn", lambda: jload(dr_)["r1_cost_per_pp_bn"]["sd"], "{:.0f}", dr_)
    emit("genDurRtwoCostPerPpBn", lambda: jload(dr_)["r2_cost_per_pp_bn"]["mean"], "{:.1f}", dr_)
    emit("genDurRtwoCostPerPpSdBn", lambda: jload(dr_)["r2_cost_per_pp_bn"]["sd"], "{:.1f}", dr_)
    emit("genDurReformDraws", lambda: jload(dr_)["n_draws"], "{:d}", dr_)
    dt = "robustness/duration_takeup_sensitivity.json"
    emit("genDurTakeupDraws", lambda: jload(dt)["n_draws"], "{:d}", dt)
    emit("genDurSixMoCostCutPct", lambda: 100 * (1 - jload(dt)["half_earnings_retention"]["results"]["exchequer_cost_bn"]["mean"]
                                                 / jload("robustness/central_monte_carlo.json")["exchequer_cost_bn"]["mean"]),
         "{:.0f}", dt)

    # --- tax composition ---------------------------------------------------
    tc = "tax_composition/composition_grid.csv"

    def comp(phi, col):
        g = csv(tc)
        return float(g[(g.unemployment_pct == 7) & (g.phi == phi)][col].iloc[0])
    emit("genCompWageBillBn", lambda: comp(0.5, "w_lost_bn"), "{:.1f}", tc)
    emit("genCompEffRatePct", lambda: 100 * comp(0.5, "effective_labour_tax_rate"), "{:.1f}", tc)
    emit("genCompLabourTaxBn", lambda: comp(0.5, "labour_tax_lost_bn"), "{:.1f}", tc)
    emit("genCompPhiZeroBn", lambda: -comp(0.5, "microsim_net_revenue_change_bn"), "{:.1f}", tc)
    emit("genCompPhiHalfBn", lambda: -comp(0.5, "net_revenue_change_with_ct_bn"), "{:.1f}", tc)
    emit("genCompPhiOneBn", lambda: -comp(1.0, "net_revenue_change_with_ct_bn"), "{:.1f}", tc)
    emit("genCompPhiOneGapBn", lambda: -comp(1.0, "delta_vs_full_labour_taxation_bn"), "{:.1f}", tc)
    rc = "tax_composition/recycling_case.json"
    emit("genRecyclingPovBhcPp", lambda: jload(rc)["recycling_minus_central"]["poverty_bhc_pp"], SIGNED2, rc)
    emit("genRecyclingPovAhcPp", lambda: jload(rc)["recycling_minus_central"]["poverty_ahc_pp"], SIGNED2, rc)

    # --- ripple ------------------------------------------------------------
    def ripple_rows():
        out = []
        for rel, lab in [("central.json", "0 (central)"), ("central_ripple_low.json", "0.15"),
                         ("central_ripple.json", "0.30"), ("central_ripple_high.json", "0.50")]:
            d = jload(rel)
            out.append(row_tex([lab, m(d["exchequer_cost"] / 1e9, "{:.1f}"),
                                m(d["poverty_rate_change_bhc"] * 100, SIGNED2),
                                m(d["poverty_rate_change_ahc"] * 100, SIGNED2),
                                m((d["gini_shocked"] - d["gini_baseline"]) * 100, SIGNED2)]))
        return "\n".join(out)
    emit("genRippleRows", ripple_rows, "{}", "ripple")
    emit("genRipplePovExtraPp", lambda: (jload("central_ripple.json")["poverty_rate_change_bhc"]
                                         - jload("central.json")["poverty_rate_change_bhc"]) * 100, "{:.2f}", "ripple")
    emit("genRippleCostRiseBn", lambda: (jload("central_ripple.json")["exchequer_cost"]
                                         - jload("central.json")["exchequer_cost"]) / 1e9, "{:.1f}", "ripple")
    emit("genRippleCostRisePct", lambda: 100 * (jload("central_ripple.json")["exchequer_cost"]
                                                / jload("central.json")["exchequer_cost"] - 1), "{:.0f}", "ripple")

    # --- recession backcast ------------------------------------------------
    rb = "robustness/recession_backcast.json"

    def rbmean(key):
        return float(np.mean([d[key] for d in jload(rb)["draws"]]))
    emit("genBackcastPovBhcPp", lambda: rbmean("poverty_change_bhc_pp"), SIGNED2, rb)
    emit("genBackcastPovAhcPp", lambda: rbmean("poverty_change_ahc_pp"), SIGNED2, rb)
    emit("genBackcastRelPovPp", lambda: rbmean("relative_poverty_change_bhc_pp"), SIGNED2, rb)
    emit("genBackcastGiniPp", lambda: rbmean("gini_change_pp"), SIGNED2, rb)
    emit("genBackcastPerPointPp", lambda: rbmean("poverty_change_bhc_pp") / 2.6, SIGNED2, rb)
    emit("genUniformPerPointPp",
         lambda: jload("robustness/incidence_monte_carlo.json")["uniform"]["poverty_change_bhc_pp"]["mean"] / 7,
         SIGNED2, "robustness/incidence_monte_carlo.json")

    # --- dividend coverage and channel split -------------------------------
    dv = "robustness/dividend_coverage.json"
    emit("genDivMicrocosmBn", lambda: jload(dv)["microcosm_dividends_2026_bn"], "{:.1f}", dv)
    emit("genDivSpiBn", lambda: jload(dv)["spi_dividends_2026_bn"], "{:.1f}", dv)
    emit("genDivRatioPct", lambda: 100 * (jload(dv)["microcosm_to_spi_ratio"] - 1), "{:.0f}", dv)
    emit("genDivRatioAbsPct", lambda: abs(100 * (jload(dv)["microcosm_to_spi_ratio"] - 1)), "{:.0f}", dv)
    emit("genDivRatioDirection",
         lambda: "above" if jload(dv)["microcosm_to_spi_ratio"] >= 1 else "below", "{}", dv)
    emit("genDivHoldersM", lambda: jload(dv)["microcosm_dividend_holders_m"], "{:.1f}", dv)
    emit("genDivScaleFactor", lambda: jload(dv)["scale_factor"], "{:.3f}", dv)
    emit("genDivDraws", lambda: jload(dv)["draws"], "{:d}", dv)
    for run, N in [("central", "Central"), ("spi_scaled", "Spi")]:
        emit(f"genDiv{N}CostMeanBn", lambda run=run: jload(dv)[run]["exchequer_cost_bn"]["mean"], "{:.1f}", dv)
        emit(f"genDiv{N}CostSdBn", lambda run=run: jload(dv)[run]["exchequer_cost_bn"]["sd"], "{:.1f}", dv)
        emit(f"genDiv{N}PovBhcMeanPp", lambda run=run: jload(dv)[run]["poverty_change_bhc_pp"]["mean"], "{:.2f}", dv)
        emit(f"genDiv{N}GiniMeanPp", lambda run=run: jload(dv)[run]["gini_change_pp"]["mean"], "{:.2f}", dv)
    emit("genDivDiffCostBn", lambda: jload(dv)["paired_difference_spi_minus_central"]["exchequer_cost_bn"]["mean"],
         SIGNED1, dv)
    emit("genDivDiffGiniPp", lambda: jload(dv)["paired_difference_spi_minus_central"]["gini_change_pp"]["mean"],
         SIGNED2, dv)
    for ch, N in [("displacement", "Disp"), ("wage", "Wage"), ("capital", "Capital")]:
        emit(f"genChannel{N}CostBn", lambda ch=ch: jload(dv)["channels_seed0"][ch]["exchequer_cost_bn"], SIGNED1, dv)
        emit(f"genChannel{N}GiniPp", lambda ch=ch: jload(dv)["channels_seed0"][ch]["gini_change_pp"], SIGNED2, dv)
        emit(f"genChannel{N}PovBhcPp", lambda ch=ch: jload(dv)["channels_seed0"][ch]["poverty_change_bhc_pp"], SIGNED2, dv)
    emit("genChannelCapitalDivTaxBn", lambda: jload(dv)["channels_seed0"]["capital"]["dividend_tax_change_bn"],
         "{:.1f}", dv)
    emit("genChannelCapitalSpiCostBn", lambda: jload(dv)["capital_channel_seed0_spi_scaled"]["exchequer_cost_bn"],
         SIGNED1, dv)
    emit("genChannelSumCostBn", lambda: jload(dv)["channels_seed0_sum_cost_bn"], "{:.1f}", dv)

    # --- quota realisation (seed-0 figures carry their realised rate) ------
    for rel, P in [("central.json", "Central"), ("low.json", "Low"), ("high.json", "High"),
                   ("central_youth_tilted.json", "Youth")]:
        emit(f"gen{P}RealisedRatePct", lambda rel=rel: 100 * jload(rel)["realised_displacement_rate"],
             "{:.2f}", rel)
        emit(f"gen{P}PrescribedRatePct", lambda rel=rel: 100 * jload(rel)["displacement_rate"],
             "{:.0f}", rel)
    for fam, Fam in FAMILY_MACRO.items():
        emit(f"genSeedZero{Fam}RealisedRatePct",
             lambda fam=fam: 100 * float(csv("incidence/summary_five.csv").set_index("family")
                                         .loc[fam, "realised_displacement_rate"]), "{:.2f}",
             "incidence/summary_five.csv")
    for stat, S in [("mean", "Mean"), ("min", "Min"), ("max", "Max"), ("sd", "Sd")]:
        emit(f"genCentralMcRealisedRate{S}Pct",
             lambda stat=stat: 100 * jload(mc)["realised_displacement_rate"][stat], "{:.2f}", mc)
    def employees_m():
        return jload("central.json")["employees_weighted"] / 1e6
    for rel, names, pre in [(ma, lam_names, "Mixed"), (sw, sw_names, "Survivor")]:
        for k, N, _ in names:
            emit(f"gen{pre}{N}RealisedRatePct",
                 lambda rel=rel, k=k: 100 * csv(rel).set_index("scenario").loc[k, "displaced_weighted_m"]
                 / employees_m(), "{:.2f}", rel)
    for fam, Fam in [("exposure", "Exposure"), ("junior", "Junior"), ("uniform", "Uniform")]:
        emit(f"genRone{Fam}RealisedRatePct",
             lambda fam=fam: 100 * pol("R1_wage_insurance", fam, "displaced_weighted_m") / employees_m(),
             "{:.2f}", ps)
    emit("genChannelSeedZeroRealisedRatePct",
         lambda: 100 * jload(dv)["channels_seed0"]["displacement"]["displaced_weighted_m"] / employees_m(),
         "{:.2f}", dv)
    emit("genGridSevenRealisedRatePct",
         lambda: 100 * float(csv(gr).query("unemployment_pct == 7 and wage_pct == 0")
                             ["realised_displacement_rate"].iloc[0]), "{:.2f}", gr)

    # --- decile-1 ranking sensitivity (20-draw means) ----------------------
    rk = "appendix/decile_ranking_sensitivity.csv"
    rankings = [("hbai", "Hbai"), ("pre_pension", "PrePension"), ("hbai_excl_large_ppc", "Excl")]

    def rank(r, d, col):
        x = csv(rk)
        return float(x[(x.ranking == r) & (x.decile == d)][col].iloc[0])
    for r, R_ in rankings:
        for d, D in [(1, "One"), (2, "Two"), (10, "Ten")]:
            emit(f"genDecileRank{R_}D{D}SignedPct", lambda r=r, d=d: rank(r, d, "disposable_change_pct"),
                 SIGNED2, rk)
            emit(f"genDecileRank{R_}D{D}SignedGbp", lambda r=r, d=d: rank(r, d, "per_capita_change_gbp"),
                 "{:+,.0f}", rk)
        emit(f"genDecileRank{R_}DOneLoPct", lambda r=r: rank(r, 1, "ci_low"), SIGNED2, rk)
        emit(f"genDecileRank{R_}DOneHiPct", lambda r=r: rank(r, 1, "ci_high"), SIGNED2, rk)
    emit("genDecileRankExclHouseholds", lambda: int(csv(rk)["excluded_households_rows"].iloc[0]), "{:,d}", rk)
    emit("genDecileRankExclPersonsK", lambda: 1000 * float(csv(rk)["excluded_persons_m"].iloc[0]), "{:,.0f}", rk)

    def d1_verdict():
        vals = [rank(r, 1, "disposable_change_pct") for r, _ in rankings]
        lows = [rank(r, 1, "ci_low") for r, _ in rankings]
        highs = [rank(r, 1, "ci_high") for r, _ in rankings]
        same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        separated = all(lo > 0 for lo in lows) or all(hi < 0 for hi in highs)
        if same_sign and separated:
            return "keeps its sign under every ranking"
        return "is not robust to how households are ranked"
    # text macro: what may be said about decile 1 (never "gains" on its own)
    emit("genDecileOneRobustness", d1_verdict, "{}", rk)
    emit("genDecileRankLargePpcFloorGbp", lambda: 10_000, "{:,d}", rk)

    # --- headline cost by component and basis (50-draw means) -------------
    cd = "robustness/central_cost_decomposition.json"
    for key, N in [("total", "Total"), ("income_tax", "IncomeTax"), ("dividend_tax", "DividendTax"),
                   ("savings_tax", "SavingsTax"), ("other_income_tax", "OtherIncomeTax"),
                   ("employee_ni", "EmployeeNi"), ("employer_ni", "EmployerNi"),
                   ("universal_credit", "Uc"), ("childcare", "Childcare"), ("other", "Other"),
                   ("excluding_employer_ni", "ExclEmployerNi"), ("narrow_cash", "NarrowCash"),
                   ("gov_balance_minus_narrow_cash", "BasisGap")]:
        emit(f"genCost{N}SignedBn", lambda key=key: jload(cd)[key]["mean"], SIGNED1, cd)
        emit(f"genCost{N}SdBn", lambda key=key: jload(cd)[key]["sd"], "{:.1f}", cd)
    emit("genCostEmployerNiSharePct",
         lambda: 100 * jload(cd)["employer_ni"]["mean"] / jload(cd)["total"]["mean"], "{:.0f}", cd)

    # --- channel split, 20 paired draws ------------------------------------
    for ch, N in [("displacement", "Disp"), ("wage", "Wage"), ("capital", "Capital")]:
        emit(f"genChannelMc{N}CostSignedBn", lambda ch=ch: jload(dv)["channels"][ch]["exchequer_cost_bn"]["mean"],
             SIGNED1, dv)
        emit(f"genChannelMc{N}PovBhcSignedPp",
             lambda ch=ch: jload(dv)["channels"][ch]["poverty_change_bhc_pp"]["mean"], SIGNED2, dv)
        emit(f"genChannelMc{N}GiniSignedPp",
             lambda ch=ch: jload(dv)["channels"][ch]["gini_change_pp"]["mean"], SIGNED2, dv)
    emit("genChannelMcSumCostSignedBn", lambda: jload(dv)["channels_sum"]["exchequer_cost_bn"], SIGNED2, dv)
    emit("genChannelMcInteractionCostSignedBn",
         lambda: jload(dv)["channels_interaction"]["exchequer_cost_bn"], SIGNED2, dv)
    emit("genChannelMcCentralCostSignedBn", lambda: jload(dv)["central"]["exchequer_cost_bn"]["mean"],
         SIGNED2, dv)
    emit("genChannelMcExclCapitalCostSignedBn",
         lambda: jload(dv)["central"]["exchequer_cost_bn"]["mean"]
         - jload(dv)["channels"]["capital"]["exchequer_cost_bn"]["mean"], SIGNED1, dv)

    # --- capital-income coverage: banded dividends, savings ----------------
    emit("genDivTaxpayerBn", lambda: jload(dv)["microcosm_dividends_taxpayers_2026_bn"], "{:.1f}", dv)
    emit("genDivBelowThresholdBn",
         lambda: jload(dv)["coverage"]["dividend_income"]["microcosm_below_threshold_bn"], "{:.2f}", dv)
    emit("genDivTopBandMicrocosmBn",
         lambda: jload(dv)["scaling"]["div_banded"]["bands"][-1]["microcosm_bn"], "{:.2f}", dv)
    emit("genDivTopBandSpiBn", lambda: jload(dv)["scaling"]["div_banded"]["bands"][-1]["spi_bn"], "{:.2f}", dv)
    emit("genDivTopBandFactor", lambda: jload(dv)["scaling"]["div_banded"]["bands"][-1]["factor"], "{:.2f}", dv)
    emit("genSavMicrocosmBn", lambda: jload(dv)["coverage"]["savings_interest_income"]["microcosm_all_bn"],
         "{:.1f}", dv)
    emit("genSavTaxpayerBn", lambda: jload(dv)["coverage"]["savings_interest_income"]["microcosm_taxpayers_bn"],
         "{:.1f}", dv)
    emit("genSavSpiBn", lambda: jload(dv)["coverage"]["savings_interest_income"]["spi_bn"], "{:.1f}", dv)
    emit("genSavScaleFactor", lambda: jload(dv)["savings_scale_factor"], "{:.3f}", dv)
    for run, N in [("div_uniform", "DivUniform"), ("div_banded", "DivBanded"),
                   ("savings_uniform", "SavUniform")]:
        emit(f"genCov{N}CostMeanBn", lambda run=run: jload(dv)[run]["exchequer_cost_bn"]["mean"], "{:.1f}", dv)
        emit(f"genCov{N}CostSdBn", lambda run=run: jload(dv)[run]["exchequer_cost_bn"]["sd"], "{:.1f}", dv)
        emit(f"genCov{N}PovBhcSignedPp", lambda run=run: jload(dv)[run]["poverty_change_bhc_pp"]["mean"],
             SIGNED2, dv)
        emit(f"genCov{N}GiniSignedPp", lambda run=run: jload(dv)[run]["gini_change_pp"]["mean"], SIGNED2, dv)
        pdk = f"paired_difference_{run}_minus_central"
        emit(f"genCov{N}DiffCostSignedBn", lambda pdk=pdk: jload(dv)[pdk]["exchequer_cost_bn"]["mean"],
             SIGNED1, dv)
        emit(f"genCov{N}DiffGiniSignedPp", lambda pdk=pdk: jload(dv)[pdk]["gini_change_pp"]["mean"],
             SIGNED2, dv)
        emit(f"genCov{N}DiffPovBhcSignedPp", lambda pdk=pdk: jload(dv)[pdk]["poverty_change_bhc_pp"]["mean"],
             SIGNED2, dv)
        emit(f"genCov{N}CapitalCostSignedBn",
             lambda run=run: jload(dv)[f"{run}_channel_capital"]["exchequer_cost_bn"], SIGNED1, dv)

    # --- tax composition: capital-off basis --------------------------------
    emit("genCompPhiZeroGovSignedBn", lambda: comp(0.5, "microsim_gov_balance_change_bn"), SIGNED1, tc)

    # --- data coverage (Microcosm occupation) -------------------------------
    oc = "robustness/occupation_vs_ashe.csv"
    emit("genOccMaxAbsDiffPp", lambda: csv(oc)["diff_pp"].abs().max(), "{:.1f}", oc)
    ov = "robustness/occupation_coverage.json"
    emit("genDataHouseholds", lambda: jload(ov)["households_rows"], "{:,d}", ov)
    emit("genDataPersons", lambda: jload(ov)["persons_rows"], "{:,d}", ov)
    emit("genDataPopulationM", lambda: jload(ov)["population_m"], "{:.1f}", ov)
    emit("genDataEmployeesM", lambda: jload(ov)["employees_m"], "{:.1f}", ov)
    emit("genDataEmployeeRows", lambda: jload(ov)["employees_rows"], "{:,d}", ov)
    emit("genDataEmployeesSocPct", lambda: 100 * jload(ov)["employees_with_major_group_share"], "{:.1f}", ov)
    emit("genDataEmployeeKishN", lambda: jload(ov)["employee_kish_effective_n"], "{:,.0f}", ov)


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
        return agg(d[f][key]["mean"] for f in FAMILY_MACRO)

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
    def five():
        return csv(s5).set_index("family")
    for fam, Fam in FAMILY_MACRO.items():
        emit(f"genSeedZero{Fam}CostBn",
             lambda fam=fam: float(five().loc[fam, "exchequer_cost_bn"]),
             "{:.1f}", s5)
        emit(f"genSeedZero{Fam}PovBhcPp",
             lambda fam=fam: float(five().loc[fam, "poverty_change_bhc_pp"]),
             "{:.2f}", s5)
        emit(f"genSeedZero{Fam}GiniPp",
             lambda fam=fam: float(five().loc[fam, "gini_change_pp"]),
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
    def pol():
        d = csv(ps)
        return d[d["family"] == "exposure"].set_index("reform")
    for reform, Ref in REFORM_MACRO.items():
        emit(f"gen{Ref}CostBn",
             lambda reform=reform: float(pol().loc[reform, "gross_cost_bn"]),
             "{:.1f}", ps)
        emit(f"gen{Ref}CostPerPpAhcBn",
             lambda reform=reform: float(pol().loc[reform, "cost_per_pp_ahc_bn"]),
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
    emit("genGridCells", lambda: len(csv(gr)), "{:d}", gr)
    emit("genGridGiniMinPp", lambda: float(csv(gr)["gini_change_pp"].min()),
         "{:.2f}", gr)
    emit("genGridGiniMaxPp", lambda: float(csv(gr)["gini_change_pp"].max()),
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

    emit_extended()
    check_directions()

    # --- write -------------------------------------------------------------
    header = [
        "% values_generated.tex — machine-generated by analysis/emit_tex_values.py",
        f"% generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        "from canonical files under results/. DO NOT EDIT BY HAND.",
        f"% certified data build: {_build_id()}",
        f"% {len(missing)} macro(s) pending (source missing, incomplete or stale).",
        "% \\GENPENDING marks a value whose canonical source was missing,",
        "% incomplete or stale at emit time: a boxed 'pending' that builds in",
        "% text and math. \\GENPENDINGROW stands in for a whole table row.",
        "\\providecommand{\\GENPENDING}{\\fbox{\\textbf{\\textcolor{red}{pending}}}}",
        "\\providecommand{\\GENPENDINGROW}{\\multicolumn{2}{l}{\\GENPENDING} \\\\}",
        "\\providecommand{\\GENMISSING}{\\GENPENDING}",
        "",
    ]
    OUT.write_text("\n".join(header + lines) + "\n")
    print(f"wrote {OUT} ({len(emitted)} macros)")
    if missing:
        print(f"WARNING: {len(missing)} macros pending:")
        for m in missing:
            print("  " + m)
        if "--strict" in sys.argv:
            sys.exit(1)


if __name__ == "__main__":
    main()
