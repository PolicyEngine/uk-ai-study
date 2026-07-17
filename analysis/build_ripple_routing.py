"""Build the 9x9 ripple-effect routing matrix R (origin -> destination).

The ripple sensitivity (reduced-form Acemoglu & Restrepo 2022 propagation)
routes each displaced worker's labour supply across destination SOC2020
major groups. The preferred source — a pairwise retraining-cost matrix from
Hosseini & Lichtinger's GenAI_Expertise release
(https://github.com/s-mahdihosseini/GenAI_Expertise) — does NOT exist: the
repo (checked 2026-07-17 via the GitHub contents API) ships only
occupation-LEVEL measures (Final_Occupation_Dataset.xlsx,
Final_Task_Dataset.xlsx: mmwo/PSS/PPG per occupation, no origin x destination
pairs). We therefore use the documented fallback:

    EMPLOYMENT-PROPORTIONAL ROUTING — R[o, l] proportional to destination
    major group l's employment share (ASHE 2025 Table 14 jobs, the same
    employment weights used throughout this repo), excluding the origin
    group: R[o, o] = 0 and each row renormalised to sum to 1.

Source: data/ashe_table14_2025_soc4.csv (ONS ASHE 2025 Table 14 unit-group
jobs, OGL v3.0, vendored from PolicyEngine/populace#325 — the same file
analysis/build_pss_crosswalk.py uses).

Output: uk_ai_study/data/uk_soc2020_major_group_ripple_routing.csv
Run:    python analysis/build_ripple_routing.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ASHE = REPO / "data" / "ashe_table14_2025_soc4.csv"
OUT = REPO / "uk_ai_study" / "data" / "uk_soc2020_major_group_ripple_routing.csv"


def build() -> pd.DataFrame:
    ashe = pd.read_csv(ASHE, dtype={"soc_code": str})
    ashe = ashe[ashe["soc_code"].str.fullmatch(r"[1-9]\d{3}")]
    ashe["major"] = ashe["soc_code"].str[0].astype(int)
    emp = ashe.groupby("major")["employment_jobs"].sum().reindex(range(1, 10))
    share = emp / emp.sum()

    matrix = np.zeros((9, 9))
    for o in range(9):
        row = share.to_numpy().copy()
        row[o] = 0.0  # zero diagonal: no routing back into the origin group
        matrix[o] = row / row.sum()

    groups = list(range(1, 10))
    out = pd.DataFrame(matrix, index=groups, columns=[f"dest_{g}" for g in groups])
    out.index.name = "origin_major_group"
    return out.reset_index()


def main() -> None:
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
