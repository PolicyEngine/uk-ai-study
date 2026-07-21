#!/bin/bash
# Thin wrapper around the manifest/DAG rebuild driver (issue #1, R2-1).
#
#   analysis/regenerate_all.sh              attested clean-room rebuild + atomic publish
#   analysis/regenerate_all.sh --dry-run    print the DAG and manifest coverage
#   analysis/regenerate_all.sh --check      verify results/ against manifest + attestation
#
# All logic lives in analysis/rebuild.py; see its module docstring.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 analysis/rebuild.py "$@"
