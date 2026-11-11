"""Certified, opt-in checkpoints for the long simulation stages.

Every draw-level checkpoint CSV carries two stamp columns on every row:

``certified_data_build_id``
    the Microcosm build pinned by the installed policyengine.py bundle
    (read from the bundle manifest, no simulation needed);
``code_fingerprint``
    a short sha256 over ``uk_ai_study/*.py`` and the script that writes the
    checkpoint, so a change to the shock mechanics or to the stage itself
    invalidates its partial results.

Resuming is opt-in (``--resume`` on the command line, as in
``replicate_jr16.py grid --resume``). Without it an existing checkpoint is
discarded and the stage recomputes from scratch. With it, a checkpoint whose
rows do not all carry the current stamp (including one written before the
stamp existed) is refused with an error rather than silently mixed into new
results.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

STAMP_COLUMNS = ("certified_data_build_id", "code_fingerprint")
PACKAGE_DIR = Path(__file__).resolve().parent


class StaleCheckpointError(RuntimeError):
    """A checkpoint was written under a different data build or code."""


@lru_cache(maxsize=1)
def certified_data_build_id() -> str:
    """The certified Microcosm UK build id pinned by policyengine.py."""
    import importlib.metadata as md

    manifest = Path(
        md.distribution("policyengine").locate_file(
            "policyengine/data/bundle/manifest.json"
        )
    )
    data = json.loads(manifest.read_text())
    uk = data["data_releases"]["uk"]
    build_id = uk.get("build_id") or uk["certified_data_artifact"]["build_id"]
    if not build_id:
        raise RuntimeError("policyengine bundle manifest has no UK build id")
    return str(build_id)


def code_fingerprint(script: str | Path | None = None) -> str:
    """Short sha256 over the package modules and the calling script."""
    h = hashlib.sha256()
    files = sorted(PACKAGE_DIR.glob("*.py"))
    if script is not None:
        files.append(Path(script).resolve())
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    # prefixed so CSV readers never parse it as a number
    return "c" + h.hexdigest()[:16]


def resume_requested(argv=None) -> bool:
    return "--resume" in (sys.argv if argv is None else argv)


def current_stamp(script: str | Path | None = None) -> dict:
    return {
        "certified_data_build_id": certified_data_build_id(),
        "code_fingerprint": code_fingerprint(script),
    }


class Checkpoint:
    """Append-only draw checkpoint with stamp verification.

    ``key`` names the columns that identify a completed unit of work (for
    example ``("seed", "family")``). ``done`` is the set of completed keys
    when resuming, otherwise empty.
    """

    def __init__(
        self,
        path: str | Path,
        key: tuple[str, ...],
        *,
        resume: bool | None = None,
        script: str | Path | None = None,
        stamp: dict | None = None,
    ):
        self.path = Path(path)
        self.key = tuple(key)
        self.resume = resume_requested() if resume is None else resume
        self.stamp = stamp if stamp is not None else current_stamp(script)
        self.done: set = set()
        if not self.path.exists():
            return
        if not self.resume:
            print(f"checkpoint {self.path}: discarding existing file "
                  "(pass --resume to continue it)", flush=True)
            self.path.unlink()
            return
        prev = read_stamped_csv(self.path)
        verify_stamp(prev, self.stamp, self.path)
        self.done = set(map(tuple, prev[list(self.key)].itertuples(index=False)))
        print(f"checkpoint {self.path}: resuming with {len(self.done)} "
              "completed units", flush=True)

    def is_done(self, *key) -> bool:
        return tuple(key) in self.done

    def append(self, row: dict) -> None:
        full = {**row, **self.stamp}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([full]).to_csv(
            self.path, mode="a", header=not self.path.exists(), index=False
        )
        self.done.add(tuple(full[k] for k in self.key))

    def read(self) -> pd.DataFrame:
        df = read_stamped_csv(self.path)
        verify_stamp(df, self.stamp, self.path)
        return df


def read_stamped_csv(path: str | Path) -> pd.DataFrame:
    """Read a checkpoint CSV keeping the stamp columns as strings."""
    return pd.read_csv(path, dtype={c: str for c in STAMP_COLUMNS})


def verify_stamp(df: pd.DataFrame, stamp: dict, path: str | Path = "") -> None:
    """Raise StaleCheckpointError unless every row carries ``stamp``."""
    for col, want in stamp.items():
        if col not in df.columns:
            raise StaleCheckpointError(
                f"refusing to resume {path}: no {col} column (written before "
                "checkpoints were stamped); delete it or run without --resume"
            )
        bad = df[col].astype(str) != str(want)
        if bad.any():
            seen = sorted(set(df.loc[bad, col].astype(str)))
            raise StaleCheckpointError(
                f"refusing to resume {path}: {int(bad.sum())} row(s) have "
                f"{col} {seen}, current is {want!r}; delete it or run without "
                "--resume"
            )


def stamp_frame(df: pd.DataFrame, stamp: dict) -> pd.DataFrame:
    """Copy of ``df`` with the stamp columns set on every row."""
    out = df.copy()
    for col, value in stamp.items():
        out[col] = value
    return out
