# Exposure crosswalk data

Provenance: built in PolicyEngine/populace#325 (SOC2020 AI-exposure crosswalk).

- `uk_soc2020_ai_exposure.csv` — unit-group (4-digit SOC2020) exposure indices:
  C-AIOE, complementarity θ (reconstructed from O*NET with a documented level
  offset), Felten AIOE, Eloundou β, DSIT AIOE/LLM scores. Built by chaining
  SOC2018→SOC2010→ISCO-08→SOC2020 crosswalks with unweighted many-to-many
  means; ASHE 2025 Table 14 employment weights cover 340 of 412 unit groups.
- `uk_soc2020_major_group_ai_exposure.csv` — the nine 1-digit major-group
  aggregates used by `exposure.py` (employment-weighted means of the above).

Sanity check: the `c_aioe` column of the major-group file must reproduce the
nine values in `results/appendix/job_loss_by_major_group.csv`
(0.619, 0.604, 0.472, 0.744, −0.333, −0.140, 0.269, −0.539, −0.729).
