# uk-ai-study

**When the shock hits the top: the fiscal and distributional incidence of AI in the United Kingdom.** The UK pairs one of the most AI-exposed workforces in the advanced economies with a tax-benefit system designed to insure shocks at the bottom of the income distribution. This study traces AI's employment, wage and capital shocks through the full UK tax-benefit system with PolicyEngine UK, using the scenario architecture of Doorley, O'Connor, O'Shea & Tuda (2026), *Artificial intelligence and income inequality in Ireland*, ESRI/DoF Report 16 ([PDF](https://www.esri.ie/system/files/publications/JR16_0.pdf)).

The study asks: if generative AI displaces some workers, raises the wages of
the rest, and raises returns to capital, what happens to the Exchequer,
poverty, and income inequality — and who bears it, by income decile and age?

## Method — exactly how we do it

The pipeline is `C-AIOE exposure → employment/wage/capital shocks (JR16 eqs
3.4/3.5) → tax-benefit microsimulation`, with these UK substitutions:

| JR16 (Ireland) | This study (UK) |
|---|---|
| SILC microdata | FRS 2024-25 (`frs_2024_25.h5`, PolicyEngine build) |
| SWITCH microsimulation | **PolicyEngine UK** (latest release) |
| ISCO occupations | SOC2020 major groups from raw FRS `adult.tab` |
| C-AIOE (Pizzinelli et al. 2023) | Same measure, via the populace PR #325 UK crosswalk |

### 1. Exposure (`uk_ai_study/exposure.py`)

Each FRS adult carries a 1-digit SOC2020 major group in the raw UKDA
`adult.tab` (`SOC2020`, coded 1000–9000). We join it onto the PolicyEngine
person table with **`person_id = SERNUM*1000 + PERSON`** — the current
`policyengine-uk-data` convention, verified to match 100% of FRS 2024-25
adults. Each major group then gets its **C-AIOE** score (Felten AIOE ×
Pizzinelli complementarity adjustment) and complementarity **θ** from the
packaged crosswalk (`uk_ai_study/data/uk_soc2020_major_group_ai_exposure.csv`,
derived in [populace PR #325](https://github.com/PolicyEngine/populace/pull/325)
from open-licensed sources: Felten et al. 2021, O*NET-reconstructed θ per IMF
WP/23/216, ASHE 2025 Table 14 employment weights; OGL v3 / CC BY 4.0 / MIT).
Persons without a SOC code (children, non-workers) receive the mean score —
they never enter the employment shock, which conditions on positive earnings.

### 2. Shocks (`uk_ai_study/shocks.py`)

- **Employment (eq 3.4):** aggregate displaced = `displacement_rate ×
  employees (weighted)`; employees without an observed SOC code form a
  pseudo-group with mean-imputed exposure, so the displacement and
  wage-uplift universes coincide. Quotas are allocated across major groups ∝
  `employment × mean C-AIOE` and realised by random draws with uniform
  ordering keys within each group (the survey weight enters only through
  quota consumption, so inclusion probability does not depend on a record's
  grossing weight — [#1](https://github.com/PolicyEngine/uk-ai-study/issues/1),
  finding 6). Displaced workers are fully out of work:
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

Baseline and shocked `policyengine_uk.Microsimulation` runs on the same
dataset; the shocked run receives the modified `employment_income`,
`savings_interest_income`, `dividend_income` and `employment_status` via
`set_input`. Reported deltas (shocked − baseline):

- **Exchequer cost** — change in `gov_balance`
- **Poverty** — BHC and AHC person-weighted rates
- **Gini** — of equivalised household disposable income (HBAI cash concept,
  `equiv_hbai_household_net_income`, matching the poverty concept — #1,
  finding 2)
- **By baseline income decile** and **by age band** (16-24 … 65+) — mean
  HBAI-household-net-income change, plus each band's share of the displaced

## Reproduce

```bash
conda create -n ukai python=3.13 -y && conda activate ukai
pip install -e .
export HUGGING_FACE_TOKEN=hf_...   # needs access to policyengine/policyengine-uk-data
python analysis/download_data.py    # FRS h5 + raw UKDA zip (adult.tab) -> data/
python analysis/run_all.py          # all presets -> results/*.json
bash analysis/regenerate_all.sh     # full snapshot; prior results moved to a timestamped backup
python analysis/artifact_manifest.py validate  # verify inputs and aggregate artifacts
python -m pytest tests/             # shock-mechanics unit tests
```

Microdata is licensed (UKDS EUL) and never committed; `data/` is gitignored.
Results in `results/` are aggregates only.
Every complete regeneration writes `results/manifest.json` with the git
commit, package versions, input checksums and checksums for every aggregate
artifact. Cross-workflow comparisons are valid only within one manifest.

## Known limitations (v0.1)

- **1-digit exposure only**: within-major-group exposure variation is lost;
  JR16 uses finer occupations. A QRF imputation from 4-digit LFS SOC (as in
  populace PR #325) is the planned upgrade.
- **Plain FRS weights**, not the calibrated enhanced FRS (its household
  cloning breaks the `adult.tab` ID join; needs the SOC merge moved upstream
  of cloning).
- Displaced workers are current-period unemployed; JR16's "9+ months
  unemployed, contributory benefits exhausted" contract is not fully
  expressible in PolicyEngine UK inputs.
- Self-employed are outside all shocks (as in JR16). Decile figures average
  50 seeded draws; the central preset is Monte-Carlo'd over 20 draws
  (`analysis/robustness.py`); grid cells are single-draw (seed=0).

## References

- Doorley, O'Connor, O'Shea & Tuda (2026), ESRI/DoF Report No. 16.
- Pizzinelli et al. (2023), IMF WP/23/216 (C-AIOE, θ).
- Felten, Raj & Seamans (2021) (AIOE).
- Briggs & Kodnani (2023), Goldman Sachs.
- Acemoglu (2025), *Economic Policy* 40(121).
- Brynjolfsson, Chandar & Chen (2025), "Canaries in the Coal Mine?".
- Klein Teeselink (2025), SSRN 5516798.
- Hosseini & Lichtinger (2026), SSRN 5425555.
