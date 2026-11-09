"""AI-exposure attachment: SOC2020 major group -> C-AIOE scores.

Exposure data (packaged, open-licensed) comes from the populace PR #325
crosswalk: Felten AIOE x Pizzinelli complementarity adjustment (C-AIOE),
aggregated to SOC2020 major groups with ASHE 2025 Table 14 employment
weights.

Each person's major group is Microcosm's ``soc2020_major_group`` (see
``uk_ai_study.data.soc_major_group``), coded 1-9 like this table.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

VALID_MAJOR_GROUPS = tuple(range(1, 10))


def load_major_group_exposure() -> pd.DataFrame:
    """The SOC2020 major-group exposure table, indexed by group 1-9."""
    path = resources.files("uk_ai_study") / "data" / "uk_soc2020_major_group_ai_exposure.csv"
    table = pd.read_csv(str(path))
    return table.set_index("soc2020_major_group")


def exposure_for_major_group(
    groups: np.ndarray | pd.Series,
    measure: str = "c_aioe",
) -> np.ndarray:
    """Per-person exposure score from major-group codes (1-9; legacy FRS
    1000-9000 codes are also accepted).

    Unknown/missing groups return NaN.
    """
    table = load_major_group_exposure()
    codes = pd.to_numeric(pd.Series(np.asarray(groups)), errors="coerce")
    codes = codes.where(codes.isin(VALID_MAJOR_GROUPS) | codes.isin([g * 1000 for g in VALID_MAJOR_GROUPS]))
    codes = codes.where(codes < 10, codes / 1000)
    return codes.map(table[measure]).to_numpy(dtype=float)
