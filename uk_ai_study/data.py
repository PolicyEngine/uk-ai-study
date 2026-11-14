"""Microcosm UK data access, pinned through policyengine.py.

Every simulation in this study — baseline, shocked and reformed — is built
with ``policyengine.uk.managed_microsimulation()``, which resolves the
certified Microcosm UK 2024-25 build named in the policyengine.py release
bundle (``pyproject.toml`` pins ``policyengine[uk]``) and records its
provenance on ``sim.policyengine_bundle``. The dataset lives in a private
Hugging Face repository, so a token with access must be available
(``HUGGING_FACE_TOKEN`` or a saved ``huggingface-cli login``). The file is
materialised once under ``data/`` relative to the working directory.

Occupation comes from the dataset's own ``soc2020_major_group`` (SOC 2020
major group of the main or last job, carried from the FRS ``SOC2020`` field
and calibrated to APS employment by group), coded 1-9 exactly like the
exposure table, so no join to raw survey files is needed.
"""

from __future__ import annotations

import os

import numpy as np

PERIOD = 2026


def _ensure_token() -> None:
    """Expose a saved Hugging Face login to policyengine.py as an env var."""
    if os.environ.get("HUGGING_FACE_TOKEN"):
        return
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover
        return
    token = get_token()
    if token:
        os.environ["HUGGING_FACE_TOKEN"] = token


def microsimulation(reform=None):
    """A Microcosm UK ``Microsimulation`` pinned to the policyengine.py bundle.

    ``reform`` (a PolicyEngine reform dict) is forwarded to the country
    package's constructor.
    """
    import policyengine as pe

    _ensure_token()
    if reform is None:
        return pe.uk.managed_microsimulation()
    return pe.uk.managed_microsimulation(reform=reform)


def bundle_provenance(sim) -> dict:
    """The model and data release pins recorded on a managed simulation."""
    bundle = getattr(sim, "policyengine_bundle", None) or {}
    keys = (
        "bundle_id",
        "policyengine_version",
        "model_version",
        "default_dataset",
        "certified_data_build_id",
        "certified_data_artifact_sha256",
    )
    return {k: bundle.get(k) for k in keys}


def soc_major_group(sim, period: int = PERIOD) -> np.ndarray:
    """SOC 2020 major group (1-9; NaN when none is recorded) per person."""
    from policyengine_uk.variables.household.income.soc2020_major_group import (
        SOC2020_MAJOR_GROUP_CODES,
        SOC2020MajorGroup,
    )

    raw = np.asarray(sim.calculate("soc2020_major_group", period=period, map_to="person").values)
    if raw.dtype.kind in "iu":  # enum indices -> member names
        names = np.array([m.name for m in SOC2020MajorGroup])[raw]
    else:
        names = raw.astype(str)
    unknown = set(np.unique(names)) - set(SOC2020_MAJOR_GROUP_CODES) - {"NONE"}
    if unknown:
        raise ValueError(f"unmapped soc2020_major_group values: {sorted(unknown)}")
    codes = np.array([SOC2020_MAJOR_GROUP_CODES.get(n, np.nan) for n in names], dtype=float)
    return codes
