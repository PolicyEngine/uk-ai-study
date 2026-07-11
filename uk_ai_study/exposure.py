"""AI-exposure attachment: FRS adult.tab SOC codes -> C-AIOE scores.

Exposure data (packaged, open-licensed) comes from the populace PR #325
crosswalk: Felten AIOE x Pizzinelli complementarity adjustment (C-AIOE),
aggregated to SOC2020 major groups with ASHE 2025 Table 14 employment
weights.

The person-ID join uses the *current* policyengine-uk-data convention,
``person_id = SERNUM * 1000 + PERSON`` (policyengine_uk_data/datasets/
frs.py), verified to match 100% of FRS 2024-25 adults.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

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
    """Per-person exposure score from major-group codes (1-9 or FRS 1000-9000).

    Unknown/missing groups return NaN.
    """
    table = load_major_group_exposure()
    codes = pd.to_numeric(pd.Series(np.asarray(groups)), errors="coerce")
    codes = codes.where(codes.isin(VALID_MAJOR_GROUPS) | codes.isin([g * 1000 for g in VALID_MAJOR_GROUPS]))
    codes = codes.where(codes < 10, codes / 1000)
    return codes.map(table[measure]).to_numpy(dtype=float)


def load_frs_adult_soc(adult_tab_path: str | Path) -> pd.Series:
    """SOC2020 major group (1000-9000) keyed by ``SERNUM*1000 + PERSON``."""
    adult = pd.read_csv(adult_tab_path, sep="\t", usecols=["SERNUM", "PERSON", "SOC2020"])
    keys = (adult["SERNUM"].astype("int64") * 1000 + adult["PERSON"].astype("int64"))
    if keys.duplicated().any():
        raise ValueError("adult.tab has duplicated SERNUM/PERSON keys.")
    soc = pd.to_numeric(adult["SOC2020"], errors="coerce")
    soc = soc.where(soc.isin([g * 1000 for g in VALID_MAJOR_GROUPS]))
    return pd.Series(soc.to_numpy(), index=keys.to_numpy(), name="soc_major_group")


def attach_soc_major_group(
    person_ids: np.ndarray | pd.Series,
    adult_tab_path: str | Path,
) -> np.ndarray:
    """FRS SOC major group (1000-9000, NaN for children/no-SOC) per person."""
    lookup = load_frs_adult_soc(adult_tab_path)
    ids = pd.to_numeric(pd.Series(np.asarray(person_ids)), errors="raise").astype("int64")
    return ids.map(lookup).to_numpy(dtype=float)
