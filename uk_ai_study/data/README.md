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

- `uk_soc2020_major_group_genai_expertise.csv` — major-group (1-digit)
  GenAI expertise measures from Hosseini & Lichtinger (2026), "Generative AI,
  Expertise, and Effective Labor Supply" (SSRN 6059674), public data release
  at https://github.com/s-mahdihosseini/GenAI_Expertise
  (Final_Occupation_Dataset.xlsx): `pss` (potential supply shift), `ppg`
  (predicted productivity gain = AI-exposed work-volume share x 0.5),
  `expertise_baseline_months` / `expertise_post_months` (required training
  months without / with a GenAI assistant, `avg_mmwo_without` /
  `avg_mmwo_combined`). Built by `analysis/build_pss_crosswalk.py`:
  O*NET-SOC 2018 -> US SOC 2018 (strip suffix, simple mean) -> US SOC 2010
  (BLS Nov 2017 crosswalk, US public domain) -> ISCO-08 (BLS 2012 crosswalk,
  US public domain) -> SOC2020 unit groups (ONS SOC 2020 Volume 2 coding
  index, Dec 2025 edition, OGL v3.0) with unweighted many-to-many means,
  then aggregated to major groups with ASHE 2025 Table 14 employment weights
  (same convention as the exposure files above). ISCO-08 major group 0
  (armed forces) has no SOC2020 counterpart and is dropped. Spearman rank
  correlation of `pss` with `c_aioe` across the nine major groups: 0.78.
