# Archived Extensions Summary (pre-Round-3)

> Historical snapshot only. Superseded by the attested Round-3 rebuild and
> retained outside `results/` so an exact-tree attestation cannot mistake it
> for a current generated output.

Five verified extension workstreams to the UK AI labour-shock study. All numbers below
were computed by the scripts listed and independently re-derived in verification; prose
errors flagged by verification are corrected here. Central scenario throughout unless
stated: 7% displacement / +2.6% wage uplift / +0.4pp capital return, seed 0, period 2026,
HBAI income concepts (person-representative Gini on equivalised BHC income; person-weighted
poverty).

---

## 1. Policy counterfactuals (`results/policy/`)

**Script:** `analysis/policy_counterfactuals.py`
**Outputs:** `R1_wage_insurance_{exposure,junior,uniform}.json`, `R2_uc_circuit_breaker_exposure.json`, `R3_cap_suspension_taper_cut_exposure.json`, `summary.csv`, `policy_reforms.png`

**Estimand:** M(shock+reform) − M(shock), fixed seed-0 shock draw; reforms apply for 2026 only.

| Reform | Cost (£bn) | Δpoverty BHC / AHC (pp) | ΔGini (pp) | £bn per pp AHC averted |
|---|---|---|---|---|
| R1 wage insurance, exposure (50% of lost earnings, £15k cap) | 21.18 gross | −1.53 / −1.65 | −0.97 | 12.8 |
| R1, junior incidence | 21.38 gross | · / −1.50 | · | 14.3 |
| R1, uniform incidence | 20.27 gross | · / −1.72 | · | 11.8 |
| R2 UC +20% standard allowance | 5.82 net | −0.64 / −0.56 | −0.31 | 10.4 (9.1 on BHC) |
| R3 benefit-cap suspension + taper 55%→45% | 4.68 net | −0.46 / −0.45 | −0.23 | 10.3 |

**Key finding:** UC-side stabilisers (R2/R3) beat wage insurance on cost per point of AHC
poverty averted (~£10.3–10.4bn vs £12.8–14.3bn) because wage insurance is
earnings-proportional — ~51% of the R1 benefit is experienced in baseline deciles 8–10
(person-weighted household-gain share, not a strict £-share), while R2/R3 concentrate
71%/67% in deciles 1–5. R1 buys ~3x more total poverty reduction at ~4x cost.

**Assumptions:** R1 is post-simulation, non-taxable and means-test-disregarded (gross =
net cost; brackets the generous end); poverty lines held at shocked-world values (poverty
rule reconstruction agrees with the model 100%). R2/R3 costs are net-of-clawback
gov_balance deltas; ~1.61m displaced, mean R1 transfer £13,173. "Taper freeze" is not a
definable parameter in policyengine-uk 2.89.2; R3 is cap thresholds ×1000 + taper 0.45.

---

## 2. Tax composition and dividend recycling (`results/tax_composition/`)

**Script:** `analysis/tax_composition.py`
**Outputs:** `composition_grid.csv` (11 displacement rates × phi ∈ {0.25, 0.5, 0.75, 1.0}), `recycling_case.json`, `revenue_shortfall_phi.png` (legend layout fixed post-verification)

**Headlines (7% displacement):**
- Displaced wage bill W_lost = £77.9bn; effective IT+NICs rate on displaced workers'
  earnings = 25.0% (pro-rata attribution), so labour tax lost = £19.5bn.
- At phi=0.5 (half the lost wage bill reappears as corporate profits, taxed at 25%),
  CT recouped = £9.7bn; the central microsim net-revenue shortfall falls from £21.5bn to
  £11.8bn. At phi=1 the channel is roughly revenue-neutral: **+£0.03bn surplus**
  (corrected: the build report stated −£0.03bn). Because the effective labour rate on
  displaced earnings ≈ the 25% CT main rate, the shortfall is almost exactly
  (1−phi)·0.25·W_lost.
- Dividend recycling (phi=0.5, payout ratio 0.5 as a parameter): £19.5bn distributed
  pro-rata to existing FRS dividend holders. Gini +0.62pp (0.3201→0.3262); poverty
  BHC/AHC unchanged (0.0pp) — recycled dividends reach no household near the poverty
  line. Mean HBAI net-income gains: **d5 +£1,873, d6 +£318, d7 +£613, d8 +£1,302,
  d10 +£467; exactly £0 in deciles 1–4 and 9** (corrected: build report omitted d6/d7).

**Assumptions:** static accounting on top of the microsim; CT incidence not modelled; CT
base = accounting profit; labour-rate attribution ignores schedule ordering; FRS dividend
records sparse (lumpy decile gains); payout ratio is a parameter, not an estimate.

---

## 3. Measured-incidence family — Klein Teeselink 2025 (`results/incidence/`)

**Script:** `analysis/measured_incidence.py` (+ `analysis/figures.py`)
**Outputs:** `measured.json`, `summary_five.csv`, `incidence_families.png` (five families)

Fifth incidence family shaped by KT (SSRN 5516798): displacement probability ∝ shifted
C-AIOE exposure × junior mult 1.29 × London mult 1.5 × wage-tier mult (9.6 above / 0.6
below weighted-median earnings). Aggregate shock size stays at the JR16 central preset.

**Headlines (seed 0, 2026):** displaced 1.61m weighted; exchequer cost £32.5bn/yr —
largest of the five families (exposure £18.4bn, compression £24.5bn); poverty +2.29pp
BHC / +2.61pp AHC (also largest); Gini +1.02pp. Most top-tilted decile transitions:
**0.024%** in decile 1 (corrected: build report said 0.04%) rising to 6.6% in decile 10.
London sensitivity (draw composition only): London share of displaced 11.4% / 18.1% /
24.2% and London displacement rate 6.2% / 9.8% / 13.1% at mults 1.0 / 1.5 / 2.0 vs 7%
national.

**Assumptions:** London mult (paper qualitative only), low-wage floor 0.6, multiplicative
channel independence, and **junior = age < 25 (a proxy for KT's seniority concept;
under-25s are ~2% of the displaced)** are all author-imposed — the last item now added to
the JSON's `author_imposed` provenance block post-verification. Firm pay tiers proxied by
person earnings median. SSRN full text was paywall-blocked; estimates taken from the KCL
press release and secondary summaries.

---

## 4. Constituency geography (`results/geo/`)

**Script:** `analysis/geo_impact.py`
**Outputs:** `constituency_impacts.csv` (650 rows), `region_summary.csv`, `hexmap_income_change.png`, `hexmap_displacement.png`, `imputation_notes.json`

**Method:** the 650×53,508 constituency weight matrix indexes enhanced FRS 2023-24, on
which the SERNUM SOC join is invalid, so SOC major group was imputed from the plain-FRS
2024-25 weighted distribution within (age band × gender × region × earnings decile) cells,
seed 0. Imputation coverage: 99.4% (weighted) from the full 4-way cell.

**Headlines (central scenario, period 2025, enhanced-FRS dataset):**
- National: aggregate HBAI household income change −1.03% (an aggregate ratio, not a mean
  of household percentages; note this figure lives in script stdout, not the CSVs);
  displaced 71.5 per 1,000 workers; BHC poverty headcount +2.75m (**+3.9pp**, corrected
  from the build report's +4.0pp).
- Regions: worst income hit Northern Ireland (−2.29%) and Scotland (−1.79%); mildest
  Wales (−0.78%) and South East (−0.83%); displacement 61 (NI) to 79 (Wales) per 1,000.
- Worst constituencies: Lagan Valley −5.43%, Upper Bann −4.11%, Belfast East −4.03%,
  Stratford and Bow −3.90%, Islington South and Finsbury −3.86%. Largest net gainers:
  Kingston and Surbiton +2.05%, Dumfriesshire Clydesdale and Tweeddale +1.75%.

**Caveats:** occupation is imputed, not observed — constituency variation reflects
demographic/earnings composition plus one seed-0 draw. The +3.9pp national poverty
response is roughly twice the plain-FRS 2026 central run (+1.87pp): a dataset
(enhanced 2023-24 vs plain 2024-25) and period (2025 vs 2026) difference. The calibrated
weight rows imply a 70.8m population (vs ONS ~68.5m) — quote headcount levels with care.
Weights are calibrated to constituency targets, not occupation mix.

---

## 5. UC caseloads and index sensitivity (`results/caseloads/`, `results/robustness/`)

**Scripts:** `analysis/caseloads.py`, `analysis/index_sensitivity.py`
**Outputs:** `results/caseloads/{central,high,low,incidence_*}.json`, `summary.csv`, `caseloads.png`; `results/robustness/index_sensitivity_full.json`, `index_sensitivity.png`

**Part A — UC caseloads (2026, seed 0; baseline UC spend £41.36bn/yr):**

| Scenario | Net new UC benunits (k) | New UC households (k)* | New UC persons (k)* | ΔUC spend (£bn/yr) | ΔHousing element in paid awards (£bn/yr) |
|---|---|---|---|---|---|
| central | 376 | 415 | 1,014 | +3.30 | +1.08 |
| high | 695 | 717 | 1,755 | +6.61 | +1.82 |
| low | 58 | 75 | 149 | +0.36 | +0.07 |
| inc: junior | 302 | 330 | 817 | +2.47 | +0.52 |
| inc: compression | 342 | 374 | 943 | +3.21 | +0.88 |
| inc: uniform | 370 | 399 | 970 | +3.16 | +1.09 |

\* Household and person columns are **gross** new entrants (no exit netting); only the
benunit column nets off the 58k benunits exiting UC (central gross entries 434k). Do not
describe 415k/1,014k as "net". inc:exposure equals central by construction.

Central ΔUC of +£3.3bn ≈ 1.0% of DWP's ~£334bn 2025/26 welfare spend (cited context
figure, not a model output); high ≈ 2.0%. Council tax benefit +£9m (central); free school
meals change computed as exactly £0.0 (genuine null — baseline FSM aggregate is £1.41bn).

**Part B — exposure-index sensitivity (central scenario, five indices):** exchequer cost
£17.6–18.6bn, ΔGini +1.05 to +1.10pp, Δpoverty BHC +1.85 to +1.90pp; decile-10 transition
share is **8.9–11.3x** decile 1 (corrected from "9–11x"; c_aioe ratio is 8.88x). All three
qualitative conclusions (Gini rises; transition gradient rises with income; fiscal cost
within ±20% of central — max deviation −4.3%, dsit_llm) survive under every index, because
the quota mechanism fixes aggregate displacement at 7% and all five indices rank SOC major
groups similarly.

**Caveats:** exposure varies only at SOC 1-digit level; housing element measured within
paid awards (slight overstatement of housing-specific cash); single seed for point
estimates (Monte Carlo dispersion in `results/robustness/central_monte_carlo.json`).

---

## Fixes applied post-verification

1. **obr:** phi=1 delta sign corrected in prose (+£0.03bn surplus); deciles 6–7 recycling
   gains now listed; figure legend re-laid out (ncol 5→3) and re-rendered from the CSV.
2. **measured:** decile-1 transition share corrected to 0.024%; junior age<25 proxy added
   to the `author_imposed` provenance block in `measured.json` and
   `analysis/measured_incidence.py`.
3. **geo:** national poverty change corrected to +3.9pp; −1.03% relabelled as an aggregate
   ratio; population-artifact caveat added.
4. **caseloads:** household/person columns explicitly labelled gross; D10/D1 ratio range
   corrected to 8.9–11.3x.
5. **policy:** decile "benefit share" clarified as person-weighted household-gain share.

No simulations were rerun; all fixes were prose/documentation/figure-layout only, which no
computed output depended on.
