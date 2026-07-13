#!/bin/bash
# Full regeneration from an empty results/ (issue #1 fix checklist).
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf results
mkdir -p results/jr16 results/appendix results/robustness results/paper_scenarios

python analysis/run_all.py
python analysis/replicate_jr16.py figs
python analysis/replicate_jr16.py grid
python analysis/appendix.py fast
python analysis/appendix.py decomp
python analysis/appendix.py grids
python analysis/robustness.py all
python analysis/paper_scenarios.py
python analysis/gender.py
python analysis/figures.py
echo "REGENERATION COMPLETE"
