# uk-ai-study

**Who bears the AI shock? The distributional and fiscal incidence in the UK.** Who loses work is at least as uncertain as how much work is lost, so this study treats incidence — exposure-proportional, junior-concentrated, uniform, and author-designed top-loaded stress tests — as an explicit scenario axis rather than assuming the shock hits the top. The UK pairs one of the most AI-exposed workforces in the advanced economies with a tax-benefit system designed to insure shocks at the bottom of the income distribution. This study traces AI's employment, wage and capital shocks through the full UK tax-benefit system with PolicyEngine UK, using the scenario architecture of Doorley, O'Connor, O'Shea & Tuda (2026), *Artificial intelligence and income inequality in Ireland*, ESRI/DoF Report 16 ([PDF](https://www.esri.ie/system/files/publications/JR16_0.pdf)).

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
  contributions, salary sacrifice and statutory pay zeroed,
  `employment_status = UNEMPLOYED` (the shared transition constructor
  `build_shocked_simulation`, finding 4).
- **Wage (eq 3.5):** surviving workers get % uplifts ∝ θ, normalised by the
  employment-weighted mean θ over baseline workers (JR16-literal — the
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
| `high` | 13% | +2.6% | Brynjolfsson, Chandar & Chen — 13% per early drafts (Nov 2025 version: 16%); cohort-specific relative decline treated as economy-wide absolute, upper bound |
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
eight records with any dividends), so the study moved to Microcosm. The paper's
appendix compares the headline results across the two datasets.

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
`--check` verifies `results/` against `results/BUILD_MANIFEST.json`. A full
rebuild runs roughly 900 Microcosm simulations; each needs several GB of
memory, so run one stage at a time on a laptop. `data/` is gitignored;
`results/` holds aggregates only.

## Known limitations

- **1-digit exposure only**: the exposure crosswalk and Microcosm's observed
  occupation field are both at SOC2020 major-group level, so within-group
  variation is lost. Microcosm's sub-major group is imputed and unused.
- **Uneven calibrated weights**: the employee records have a Kish effective
  sample size of a few thousand, so single displacement draws are noisy and
  the realised weighted count displaced varies around the 7% quota. Report
  Monte Carlo means (50 draws for the incidence families and policy runs).
- **Dividend coverage**: Microcosm's 2026 dividend total runs above the HMRC
  SPI projection for 2026, mostly above £1m of total income;
  `analysis/sensitivity_dividend_coverage.py` reruns the central scenario
  with dividends scaled to the SPI total.
- **Occupation of SPI-drawn records** is imputed from income, not observed.
- Displaced workers are current-period unemployed; JR16's "9+ months
  unemployed, contributory benefits exhausted" contract is not fully
  expressible in PolicyEngine UK inputs.
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
- Brynjolfsson, Chandar & Chen (2025), "Canaries in the Coal Mine?".
- Klein Teeselink (2025), SSRN 5516798.
- Hosseini & Lichtinger (2026), SSRN 5425555.
