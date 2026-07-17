"""Build UK SOC2020 major-group GenAI expertise measures (PSS, PPG, expertise).

Maps Hosseini & Lichtinger (2026), "Generative AI, Expertise, and Effective
Labor Supply" (SSRN 6059674; https://github.com/s-mahdihosseini/GenAI_Expertise)
occupation-level measures from O*NET-SOC 2018 onto the nine UK SOC2020 major
groups, mirroring the repo's existing exposure-crosswalk convention
(PolicyEngine/populace#325):

    O*NET-SOC 2018 (8-digit) -> US SOC 2018 (strip .XX suffix; simple mean)
    -> US SOC 2010 (BLS soc_2010_to_2018_crosswalk.xlsx, Nov 2017; unweighted
       many-to-many mean)
    -> ISCO-08 (BLS isco_soc_crosswalk.xls, Aug 2012/upd. Jun 2015; unweighted
       mean)
    -> UK SOC2020 unit group (ONS SOC 2020 Volume 2 coding index, Dec 2025
       edition, unique SOC2020 x ISCO-08 pairs; unweighted mean)
    -> major group 1-9 (employment-weighted mean, ASHE 2025 Table 14 jobs).

ISCO-08 major group 0 (armed forces) has no SOC2020 counterpart and drops out
automatically (no coding-index SOC2020 unit group maps to ISCO 0xxx).

Sources / licences:
- GenAI_Expertise Final_Occupation_Dataset.xlsx (Hosseini & Lichtinger 2026,
  public GitHub release).
- BLS crosswalks: US public domain. bls.gov blocks non-browser downloads, so
  the script falls back to verified GitHub mirrors of the identical files.
- ONS SOC 2020 Volume 2 coding index: Open Government Licence v3.0.
- ASHE 2025 Table 14 unit-group jobs: ONS, OGL v3.0 (copy vendored from
  PolicyEngine/populace#325).

Output: uk_ai_study/data/uk_soc2020_major_group_genai_expertise.csv
Run:    python analysis/build_pss_crosswalk.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data"  # gitignored raw-download cache
OUT = REPO / "uk_ai_study" / "data" / "uk_soc2020_major_group_genai_expertise.csv"

SOURCES = {
    # Hosseini & Lichtinger (2026) public data release.
    "Final_Occupation_Dataset.xlsx": (
        "https://raw.githubusercontent.com/s-mahdihosseini/GenAI_Expertise/"
        "main/Final_Occupation_Dataset.xlsx"
    ),
    # BLS US SOC 2010 <-> 2018 crosswalk (Nov 2017). Canonical URL
    # https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx is blocked
    # for non-browser agents; mirror is a byte-identical copy.
    "soc_2010_to_2018_crosswalk.xlsx": (
        "https://raw.githubusercontent.com/TWalstrum/Concordances/main/"
        "bls_soc/data/raw/soc_2010_to_2018_crosswalk.xlsx"
    ),
    # BLS ISCO-08 <-> US SOC 2010 crosswalk (2012, upd. 2015). Canonical URL
    # https://www.bls.gov/soc/soccrosswalks/isco_soc_crosswalk.xls (blocked;
    # mirrored copy retains BLS authorship metadata).
    "isco_soc_crosswalk.xls": (
        "https://raw.githubusercontent.com/kmazurek95/typology-paper/main/"
        "data/raw/isco_soc_crosswalk.xls"
    ),
    # ONS SOC 2020 Volume 2 coding index (Dec 2025 edition, OGL v3.0).
    "ons_soc2020_vol2_coding_index.xlsx": (
        "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/"
        "standardoccupationalclassificationsoc/soc2020/"
        "soc2020volume2codingrulesandconventions/"
        "soc2020volume2thecodingindexexcel03122025.xlsx"
    ),
    # ASHE 2025 Table 14 unit-group employment (jobs), vendored in
    # PolicyEngine/populace#325.
    "ashe_table14_2025_soc4.csv": (
        "https://raw.githubusercontent.com/PolicyEngine/populace/"
        "feature/uk-ai-exposure-analysis/packages/populace-build/src/populace/"
        "build/uk_runtime/occupation_targets_data/ashe_table14_2025_soc4.csv"
    ),
}

MEASURES = {
    "pss": "pss",
    "pred_productivity_effect": "ppg",
    "avg_mmwo_without": "expertise_baseline_months",
    "avg_mmwo_combined": "expertise_post_months",
}


def download() -> None:
    RAW.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        path = RAW / name
        if not path.exists():
            print(f"downloading {name}")
            subprocess.run(["curl", "-sL", "-o", str(path), url], check=True)


def build() -> pd.DataFrame:
    # 1. O*NET-SOC 2018 -> US SOC 2018 (simple mean across .XX detail codes).
    occ = pd.read_excel(RAW / "Final_Occupation_Dataset.xlsx")
    occ["soc2018"] = occ["onetsoccode"].str[:7]
    soc2018 = occ.groupby("soc2018")[list(MEASURES)].mean()

    # 2. US SOC 2018 -> US SOC 2010 (BLS, unweighted many-to-many mean).
    cw1018 = pd.read_excel(
        RAW / "soc_2010_to_2018_crosswalk.xlsx", header=8, dtype=str
    ).rename(columns={"2010 SOC Code": "soc2010", "2018 SOC Code": "soc2018"})
    cw1018 = cw1018.dropna(subset=["soc2010", "soc2018"])
    soc2010 = (
        cw1018.merge(soc2018, on="soc2018", how="inner")
        .groupby("soc2010")[list(MEASURES)]
        .mean()
    )

    # 3. US SOC 2010 -> ISCO-08 (BLS, unweighted mean).
    isco_cw = pd.read_excel(
        RAW / "isco_soc_crosswalk.xls",
        sheet_name="ISCO-08 to 2010 SOC",
        header=6,
        dtype=str,
    ).rename(columns={"ISCO-08 Code": "isco08", "2010 SOC Code": "soc2010"})
    isco_cw = isco_cw.dropna(subset=["isco08", "soc2010"])
    isco_cw["isco08"] = isco_cw["isco08"].str.strip().str.zfill(4)
    isco_cw["soc2010"] = isco_cw["soc2010"].str.strip()
    isco = (
        isco_cw.merge(soc2010, on="soc2010", how="inner")
        .groupby("isco08")[list(MEASURES)]
        .mean()
    )

    # 4. ISCO-08 -> UK SOC2020 unit groups (ONS coding index, unique pairs,
    #    unweighted mean). ISCO major group 0 (armed forces) has no SOC2020
    #    counterpart and is dropped implicitly.
    idx = pd.read_excel(
        RAW / "ons_soc2020_vol2_coding_index.xlsx",
        sheet_name="SOC2020 coding index",
        dtype=str,
    )
    pairs = (
        idx[["SOC_2020", "ISCO-08 code based on SOC2020"]]
        .rename(columns={"SOC_2020": "soc2020", "ISCO-08 code based on SOC2020": "isco08"})
        .dropna()
    )
    pairs["soc2020"] = pairs["soc2020"].str.strip()
    pairs["isco08"] = pairs["isco08"].str.split(".").str[0].str.zfill(4)
    pairs = pairs[pairs["soc2020"].str.fullmatch(r"[1-9]\d{3}")].drop_duplicates()
    unit = (
        pairs.merge(isco, on="isco08", how="inner")
        .groupby("soc2020")[list(MEASURES)]
        .mean()
        .reset_index()
    )

    # 5. Major group 1-9, employment-weighted (ASHE 2025 Table 14 jobs),
    #    matching the repo's uk_soc2020_major_group_ai_exposure.csv convention.
    ashe = pd.read_csv(RAW / "ashe_table14_2025_soc4.csv", dtype={"soc_code": str})
    unit = unit.merge(
        ashe[["soc_code", "employment_jobs"]],
        left_on="soc2020",
        right_on="soc_code",
        how="left",
    )
    unit["soc2020_major_group"] = unit["soc2020"].str[0].astype(int)

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["employment_jobs"]
        out = {}
        for src in MEASURES:
            m = g[src].notna() & w.notna()
            out[MEASURES[src]] = (
                np.average(g.loc[m, src], weights=w[m]) if m.any() else g[src].mean()
            )
        out["n_source_occupations"] = int(g["pss"].notna().sum())
        return pd.Series(out)

    major = (
        unit.groupby("soc2020_major_group")
        .apply(agg, include_groups=False)
        .reset_index()
    )
    major["n_source_occupations"] = major["n_source_occupations"].astype(int)
    return major


def main() -> None:
    download()
    major = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    major.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(major.to_string(index=False))

    exposure = pd.read_csv(
        REPO / "uk_ai_study" / "data" / "uk_soc2020_major_group_ai_exposure.csv"
    )
    merged = major.merge(exposure, on="soc2020_major_group")
    rho, p = spearmanr(merged["pss"], merged["c_aioe"])
    print(f"\nSpearman rank correlation, PSS vs c_aioe (9 major groups): "
          f"rho={rho:.3f} (p={p:.4f})")


if __name__ == "__main__":
    main()
