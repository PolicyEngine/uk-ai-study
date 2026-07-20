"""Write and validate provenance for the paper-facing result snapshot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.json"
INPUTS = (
    ROOT / "data/frs_2024_25.h5",
    ROOT / "data/frs_2024_25/UKDA-9563-tab/tab/adult.tab",
    ROOT / "uk_ai_study/data/uk_soc2020_major_group_ai_exposure.csv",
    ROOT / "uk_ai_study/data/uk_soc2020_major_group_genai_expertise.csv",
    ROOT / "uk_ai_study/data/uk_soc2020_major_group_ripple_routing.csv",
)
REQUIRED = (
    "central.json", "low.json", "high.json", "jr16/grid.csv",
    "robustness/central_monte_carlo.json", "robustness/incidence_monte_carlo.json",
    "incidence/summary_five.csv", "policy/summary.csv", "caseloads/summary.csv",
    "geo/constituency_impacts.csv", "tax_composition/recycling_case.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write() -> None:
    files = sorted(p for p in RESULTS.rglob("*") if p.is_file() and p != MANIFEST)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("policyengine-uk", "policyengine-core", "numpy", "pandas")
        },
        "inputs": {
            str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            if p.exists() else {"missing": True}
            for p in INPUTS
        },
        "artifacts": {
            str(p.relative_to(RESULTS)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in files
        },
        "command": "bash analysis/regenerate_all.sh",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")


def validate() -> None:
    if not MANIFEST.exists():
        raise SystemExit("results/manifest.json is missing")
    payload = json.loads(MANIFEST.read_text())
    missing = [name for name in REQUIRED if not (RESULTS / name).exists()]
    changed = [
        name for name, meta in payload["artifacts"].items()
        if not (RESULTS / name).exists() or sha256(RESULTS / name) != meta["sha256"]
    ]
    missing_inputs = [name for name, meta in payload["inputs"].items() if meta.get("missing")]
    if missing or changed or missing_inputs:
        raise SystemExit(json.dumps({"missing_artifacts": missing, "changed": changed,
                                    "missing_inputs": missing_inputs}, indent=2))
    print(f"validated {len(payload['artifacts'])} artifacts against one snapshot")


if __name__ == "__main__":
    {"write": write, "validate": validate}[sys.argv[1]]()
