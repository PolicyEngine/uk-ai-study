# uk-ai-study

**A UK replication of ESRI JR16 — Doorley, O'Connor, O'Shea & Tuda (2026), *Artificial intelligence and income inequality in Ireland* ([PDF](https://www.esri.ie/system/files/publications/JR16_0.pdf)) — using PolicyEngine UK in place of SWITCH.**

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
  employed (weighted)`; allocated across major groups ∝ `employment × mean
  C-AIOE`, realised by weighted random draws within each group (probability ∝
  individual exposure × survey weight). Displaced workers get
  `employment_income = 0` and `employment_status = UNEMPLOYED`.
- **Wage (eq 3.5):** an uplift pool of `wage_uplift × surviving wage bill` is
  distributed across surviving workers ∝ θ × earnings — AI complements
  high-θ occupations.
- **Capital:** interest and dividend income scaled by
  `(1.005% + 0.4pp)/1.005% ≈ 1.398` (JR16's return-to-capital shock).

**Scenario presets** (all overridable literature anchors):

| Preset | Displacement | Wage | Source |
|---|---|---|---|
| `central` | 7% | +2.6% | Briggs & Kodnani (2023); wage figure is JR16's adopted median (fn.3, §3.2) |
| `low` | 1% | 0% | Acemoglu (2025, *Economic Policy* 40(121)), employment-only per JR16 fn.8 |
| `high` | 13% | +2.6% | Brynjolfsson, Chandar & Chen (2025) — cohort-specific, upper bound |
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
- **Gini** — of equivalised household net income
- **By baseline income decile** and **by age band** (16-24 … 65+) — mean
  household-net-income change, plus each band's share of the displaced

## Reproduce

```bash
conda create -n ukai python=3.13 -y && conda activate ukai
pip install -e .
export HUGGING_FACE_TOKEN=hf_...   # needs access to policyengine/policyengine-uk-data
python analysis/download_data.py    # FRS h5 + raw UKDA zip (adult.tab) -> data/
python analysis/run_all.py          # all presets -> results/*.json
```

Microdata is licensed (UKDS EUL) and never committed; `data/` is gitignored.
Results in `results/` are aggregates only.

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
- Self-employed are outside all shocks (as in JR16). Single seeded draw per
  scenario (seed=0); JR16 averages over draws.

## References

- Doorley, O'Connor, O'Shea & Tuda (2026), ESRI/DoF Report No. 16.
- Pizzinelli et al. (2023), IMF WP/23/216 (C-AIOE, θ).
- Felten, Raj & Seamans (2021) (AIOE).
- Briggs & Kodnani (2023), Goldman Sachs.
- Acemoglu (2025), *Economic Policy* 40(121).
- Brynjolfsson, Chandar & Chen (2025), "Canaries in the Coal Mine?".
- Klein Teeselink (2025), SSRN 5516798.
- Hosseini & Lichtinger (2026), SSRN 5425555.
