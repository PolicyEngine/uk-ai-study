#!/bin/bash
# Regenerate every paper-facing artifact from one environment and data snapshot.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d results && -n "$(find results -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup="results.backup.${stamp}"
  mv results "$backup"
  echo "Moved prior artifacts to $backup"
fi

mkdir -p results/{jr16,appendix,robustness,paper_scenarios,incidence,policy,caseloads,geo,tax_composition}

python analysis/run_all.py
python analysis/replicate_jr16.py figs
python analysis/replicate_jr16.py grid
python analysis/appendix.py fast
python analysis/appendix.py decomp
python analysis/appendix.py grids
python analysis/robustness.py all
python analysis/paper_scenarios.py
python analysis/gender.py
python analysis/incidence_scenarios.py
python analysis/measured_incidence.py
python analysis/monte_carlo_families.py
python analysis/policy_counterfactuals.py
python analysis/sensitivity_duration_takeup.py
python analysis/sensitivity_duration_reforms.py
python analysis/sensitivity_enhanced_dataset.py
python analysis/sensitivity_wage_tier.py
python analysis/index_sensitivity.py
python analysis/caseloads.py
python analysis/geo_impact.py
python analysis/geo_choropleth.py
python analysis/tax_composition.py
python analysis/figures.py
python analysis/artifact_manifest.py write
python analysis/artifact_manifest.py validate
echo "REGENERATION COMPLETE"
