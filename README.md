# uk-ai-study

**Who bears the AI shock? The distributional and fiscal incidence in the UK.** Who loses work is at least as uncertain as how much work is lost, so this study treats incidence — exposure-proportional, junior-concentrated, uniform, and author-designed top-loaded stress tests — as an explicit scenario axis rather than assuming the shock hits the top. This study traces AI's employment, wage and capital shocks through the full UK tax-benefit system with PolicyEngine UK, using the scenario architecture of Doorley, O'Connor, O'Shea & Tuda (2026), *Artificial intelligence and income inequality in Ireland*, ESRI/DoF Report 16 ([PDF](https://www.esri.ie/system/files/publications/JR16_0.pdf)).

The study asks: if generative AI displaces some workers, raises the wages of
the rest, and raises returns to capital, what happens to the Exchequer,
poverty, and income inequality — and who bears it, by income decile and age?

## Method — exactly how we do it

The pipeline is `C-AIOE exposure → employment/wage/capital shocks (JR16 eqs
3.4/3.5) → tax-benefit microsimulation`, with these UK substitutions:

| JR16 (Ireland) | This study (UK) |
|---|---|
| SILC microdata | **Microcosm UK 2024-25** (PolicyEngine's calibrated FRS-based dataset, pinned by `policyengine[uk]==6.16.1`) |
| SWITCH microsimulation | **PolicyEngine UK** 2.174.1, via `policyengine.uk.managed_microsimulation()` |
| ISCO occupations | SOC2020 major groups from Microcosm's `soc2020_major_group` |
| C-AIOE (Pizzinelli et al. 2023) | Same measure, via the populace PR #325 UK crosswalk |

### 1. Exposure (`uk_ai_study/exposure.py`, `uk_ai_study/data.py`)

Each Microcosm person carries `soc2020_major_group`, the SOC 2020 major
group of their main or last job (policyengine-uk 2.173.0+, carried from the
FRS `SOC2020` field and calibrated to APS employment by major group and
region; imputed from income for SPI-drawn records). It is coded 1-9, the same
key as the exposure table, so no join to raw survey files is needed
(`uk_ai_study.data.soc_major_group`). Each major group then gets its
**C-AIOE** score (Felten AIOE × Pizzinelli complementarity adjustment) and
complementarity **θ** from the packaged crosswalk
(`uk_ai_study/data/uk_soc2020_major_group_ai_exposure.csv`, derived in
[populace PR #325](https://github.com/PolicyEngine/populace/pull/325)
from open-licensed sources: Felten et al. 2021, O*NET-reconstructed θ per IMF
WP/23/216, ASHE 2025 Table 14 employment weights; OGL v3 / CC BY 4.0 / MIT).
Persons without a major group receive the employment-weighted mean score;
only employees (positive employment income) enter the employment shock.

### 2. Shocks (`uk_ai_study/shocks.py`)

- **Employment (eq 3.4):** aggregate displaced = `displacement_rate ×
  employees (weighted)`; employees without an observed SOC code form a
  pseudo-group with mean-imputed exposure, so the displacement and
  wage-uplift universes coincide. Quotas are allocated across major groups ∝
  `employment × mean C-AIOE` and realised by systematic sampling on a random
  permutation within each group, with prescribed first-order inclusion
  probabilities that are equal for all records in a group (probability tilts
  such as the youth multiplier enter as tilts to these inclusion
  probabilities); a record's grossing weight enters only through the quota
  accounting, never its inclusion probability
  ([#1](https://github.com/PolicyEngine/uk-ai-study/issues/1), findings 6 and
  R2-5). Displaced workers are fully out of work:
  `employment_income = 0`, `hours_worked = 0`, employee pension
  contributions (employee, employer, salary- and bonus-sacrifice and
  personal), hours and statutory sick, maternity and paternity pay zeroed
  (`TRANSITION_ZEROED_VARIABLES`; personal contributions are zeroed because
  Microcosm carries some very large ones that HBAI income subtracts),
  `employment_status = UNEMPLOYED` (the shared transition constructor
  `build_shocked_simulation`, finding 4).
- **Wage (eq 3.5):** surviving workers get % uplifts ∝ θ, normalised by the
  employment-weighted mean θ over baseline workers, so the uplift averages
  the headline rate over all baseline employees but is paid to survivors
  only (JR16-literal — the
  estimand decision on
  [#1](https://github.com/PolicyEngine/uk-ai-study/issues/1), finding 5;
  per-seed conservation tested in `tests/test_shocks.py`).
- **Capital:** interest and dividend income scaled by
  `(1.005% + 0.4pp)/1.005% ≈ 1.398` (JR16's return-to-capital shock).

**Scenario presets** (all overridable; the +0.4pp capital shock is on in
every preset, as in all JR16 scenarios):

| Preset | Displacement | Wage | Source |
|---|---|---|---|
| `central` | 7% | +2.6% | JR16's central calibration (§3.2), converting Briggs & Kodnani (2023) task-exposure and productivity figures into displacement and wage rates |
| `low` | 1% | 0% | Sensitivity case; JR16 §3.2 attributes ~1% to Acemoglu (2025), but his 0.9–1.1% is a ten-year GDP figure, not an employment effect ([#1](https://github.com/PolicyEngine/uk-ai-study/issues/1), finding 11) |
| `high` | 13% | +2.6% | Sensitivity case, not tied to a current source: the version of Brynjolfsson, Chandar & Chen the paper cites (Aug 2026) reports a 19% relative employment decline for 22–25-year-olds in the most exposed US jobs (Nov 2025 version: 16%). A cohort-specific relative decline treated as an economy-wide absolute rate, so an upper bound; changing it requires rerunning the preset |
| `central_youth_tilted` | 7% | +2.6% | + Klein Teeselink (2025) junior/total ratio 5.8/4.5 tilting draws toward ages 16–24 |

The `youth_displacement_multiplier` extends JR16 (which draws randomly within
groups) toward the seniority-biased evidence in Klein Teeselink (2025) and
Hosseini & Lichtinger (2026).

### 3. Microsimulation (`uk_ai_study/runner.py`)

Baseline and shocked simulations are built with
`policyengine.uk.managed_microsimulation()` (`uk_ai_study.data.microsimulation`),
so every run uses the certified Microcosm build and records its provenance; the shocked run receives the modified `employment_income`,
`savings_interest_income`, `dividend_income` and `employment_status` via
`set_input`. Reported deltas (shocked − baseline):

- **Exchequer cost** — change in `gov_balance`
- **Poverty** — BHC and AHC person-weighted rates
- **Gini** — of equivalised household disposable income (HBAI cash concept,
  `equiv_hbai_household_net_income`, matching the poverty concept — #1,
  finding 2)
- **By baseline income decile** and **by age band** (16-24 … 65+) — mean
  HBAI-household-net-income change, plus each band's share of the displaced

## Data

All simulations run on **Microcosm UK 2024-25**, build
`microcosm-uk-2024-25-national-20261107T020434Z-eeb6093e` (sha256
`0a9b77db…1f25`), the default dataset of the `policyengine[uk]==6.16.1`
release bundle. Microcosm starts from the FRS 2024-25, adds SPI-drawn
top-income records, imputes variables the survey lacks and calibrates
household weights to about 1,500 administrative targets (HMRC SPI, ONS, DWP,
OBR and others). It lives in a private Hugging Face repository
(`policyengine/populace-uk-private`), so you need a token with access.

The earlier version of the study used PolicyEngine's processed, uncalibrated
FRS 2024-25 file (`frs_2024_25.h5`) plus a join to the raw UKDA `adult.tab`
for occupation. That file had two build defects (household-level variables
attached in row order rather than by serial number, giving about 62m people
and 22.9m employees; and dividends joined on the wrong person key, leaving
eight records with any dividends), so the study moved to Microcosm. Every result from
the earlier file is superseded; PR #18 records the change.

## Reproduce

```bash
uv sync                                   # Python 3.13, policyengine[uk]==6.16.1 (uv.lock)
huggingface-cli login                     # or export HUGGING_FACE_TOKEN=hf_... (read access to policyengine/populace-uk-private)
uv run python analysis/download_data.py   # optional: fetch the Microcosm h5 into data/ (sha256-checked)
uv run python analysis/run_all.py         # presets -> results/*.json
uv run bash analysis/regenerate_all.sh    # or: the full attested result set, built from empty results/
uv run python analysis/emit_tex_values.py # paper/values_generated.tex from results/
uv run pytest tests/                      # unit tests
```

`analysis/rebuild.py --dry-run` lists every stage and its outputs;
`--check` verifies `results/` and the committed `paper/values_generated.tex`
and `paper/main.pdf` against the hashes in `results/BUILD_MANIFEST.json`. A
full rebuild runs roughly 950 Microcosm simulations for the original stages
plus about 350 for runs 15 and 16, the take-up runs and the wage-cut run;
each needs several GB of memory, so run one stage at a time on a laptop. `data/` is gitignored;
`results/` holds aggregates only. Runs 15 and 16 and the wage-cut run also
write household arrays outside the repository
(`../ai-study-household-arrays`, `../ai-study-wage-cut-arrays`);
`results/next_runs/obr_household_deciles.csv` is rebuilt from the first.

### What the attestation covers

`results/BUILD_MANIFEST.json` attests the hashes of the committed tree; it
is not a record that every file came out of one clean-room build. The last
full clean-room build predates five stages: `obr_benchmark` and
`capital_variants` (runs 15 and 16, 13 Nov 2026), `obr_household_deciles`,
`takeup` (14 Nov 2026) and `wage_cut` (13 Nov 2026). Each was run on its own
against the same certified Microcosm build and committed; the attestation
lists them under `outside_clean_room`, and `rebuild.py --check` prints them
on every run. Their manifest stages exist, so a full `rebuild.py` would
regenerate them from empty `results/`; that full rebuild has not been rerun
since they were added.

In November 2026 the code fingerprint was extended to cover the sibling
modules each stage imports, and duplicated constants were moved to one place
(review of PR #18, findings M7 and M8). Those edits changed no simulation
but did change the fingerprints that stamp the committed draws, so the draws
were restamped rather than rerun: `analysis/restamp_checkpoints.py` proves
every edit non-behavioural (identical normalised syntax trees before and
after; newly covered helper modules last changed before the draws were
written) and then replaces only the stamp. `results/RESTAMP_LOG.json`
records the old and new fingerprint of every checkpoint and the verdict for
every file the fingerprints cover.

Long stages checkpoint every draw or grid cell to CSV. Each row is stamped
with the certified Microcosm build id and a fingerprint of the code that
wrote it (`uk_ai_study/checkpoint.py`). A stage resumes from its checkpoint
only when given `--resume`, and refuses a checkpoint whose stamp differs;
without the flag it recomputes from scratch. `emit_tex_values.py` writes a
visible boxed "pending" for any value whose source is missing, incomplete
(fewer cells or draws than the design) or stale, so the paper builds with
`tectonic paper/main.tex` at any point of a rerun; `--strict` fails instead.

## Known limitations

- **1-digit exposure only**: the exposure crosswalk and Microcosm's observed
  occupation field are both at SOC2020 major-group level, so within-group
  variation is lost. Microcosm's sub-major group is imputed and unused.
- **Uneven calibrated weights**: the employee records have a Kish effective
  sample size of a few thousand, so single displacement draws are noisy and
  the realised weighted count displaced varies around the 7% quota. Report
  Monte Carlo means (50 draws for the incidence families and policy runs).
- **Dividend coverage**: Microcosm's 2026 dividend total runs above the HMRC
  SPI projection for that year, mostly above £1m of total income;
  `analysis/sensitivity_dividend_coverage.py` reruns the central scenario
  with dividends scaled to the SPI total.
- **Occupation of SPI-drawn records** is imputed from income, not observed.
- Displaced workers are current-period unemployed for the full year;
  JR16's "9+ months unemployed, contributory benefits exhausted" contract is
  not fully expressible in PolicyEngine UK inputs. Under the central
  assumption they keep their benefit unit's survey-drawn UC take-up flag and
  receive no new-style JSA; `analysis/displaced_takeup_runs.py` reports
  runs with full take-up and with new-style JSA.
- **Model year**: `uk_ai_study.data.PERIOD = 2026`, which PolicyEngine UK
  runs on 2026-27 tax and benefit rates; every script imports it.
- Self-employed are outside all shocks (as in JR16).
- No constituency or local-authority breakdown: the earlier maps used the
  enhanced FRS 2023-24 and were dropped with it. Microcosm's dense
  local-area build could restore them.

## References

- Doorley, O'Connor, O'Shea & Tuda (2026), ESRI/DoF Report No. 16.
- Pizzinelli et al. (2023), IMF WP/23/216 (C-AIOE, θ).
- Felten, Raj & Seamans (2021) (AIOE).
- Briggs & Kodnani (2023), Goldman Sachs.
- Acemoglu (2025), *Economic Policy* 40(121).
- Brynjolfsson, Chandar & Chen (2026), "Canaries in the Coal Mine?", August 2026 version (first circulated August 2025).
- Klein Teeselink (2025), SSRN 5516798.
- Hosseini & Lichtinger (2026), SSRN 5425555.
