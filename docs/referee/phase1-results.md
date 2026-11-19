# Phase 1 results: exact expectations and sampling uncertainty (16 November 2026)

Method: option A of [assignment.md](assignment.md), chosen by the author. `prototypes/state_enumeration.py` runs one shocked simulation per displacement state of each household's employees, and `prototypes/state_contrasts.py` reweights the saved household outcomes. Both run on Microcosm UK 2024-25 at the 2027-28 central setting (7 per cent, +2.6 per cent survivor wages, capital shock). Nothing in `uk_ai_study/` or the attested results has changed.

## Cost and coverage

- 64 states (K = 6) cover every household in the data: the largest has six employees. Each state takes about 30 seconds; the full set took 38 minutes, against about 4 hours for the 50-draw central and family runs.
- K = 3 (8 states, 5 minutes) covers 98.7 per cent of expected displaced workers. It misses 26,000 fourth and later employees, worth about £0.4bn of cost.
- The state probabilities sum to one in every household (maximum error 7e-16).

## Acceptance test: exact central case against the current 50-draw means

| Measure | Exact (K = 6) | 50-draw mean | Gap in Monte Carlo SEs |
|---|---|---|---|
| Exchequer cost | £7.26bn | £7.34bn | −0.3 |
| BHC poverty change | +2.46pp | +2.40pp | +2.0 |
| AHC poverty change | +2.61pp | +2.57pp | +1.3 |
| Gini change | +1.78pp | +1.76pp | +1.4 |
| Expected displaced | 2.03m | 2.04m | — |

All within 2 Monte Carlo SEs: the method passes. Poverty sits slightly above the draw mean, consistent with independence within the household allowing a little more joint displacement than the systematic sampler.

## Uniform against exposure-proportional incidence, with survey-sampling uncertainty (referee M4)

Household bootstrap (B = 200, households resampled with replacement; Microcosm has no strata or PSU identifiers, so this approximates simple random sampling).

| Uniform minus exposure | Exact | Bootstrap SE | 95% interval |
|---|---|---|---|
| Cost | −£5.57bn | 0.29 | [−6.17, −5.01] |
| BHC poverty | +0.003pp | 0.028 | [−0.05, +0.06] |
| AHC poverty | −0.08pp | 0.031 | [−0.14, −0.03] |
| Gini | +0.07pp | 0.013 | [+0.05, +0.10] |

Sampling SEs of the levels (exposure-proportional): cost £1.07bn, BHC poverty 0.11pp, AHC poverty 0.18pp, Gini 0.07pp.

## Consequences for the paper

1. **The "cost and poverty do not move together" claim does not survive** (intro and results; currently uniform BHC +0.12pp [+0.04, +0.19] above exposure). Under exact expectations the two rules raise BHC poverty by the same amount, and uniform is *lower* on AHC. The finding was assignment noise. Rewrite it as: uniform incidence costs £5.6bn less with the same BHC poverty rise and a slightly larger Gini rise (+0.07pp).
2. The cost ordering across rules is robust (interval far from zero), as is the Gini difference, at about 60 per cent of the paper's current figure.
3. Report survey-sampling SEs beside every headline level and contrast; assignment noise is now zero.
4. The deterministic method makes the grid, presets, index table and closure exact and removes the OLS index adjustment (item 4).

## All five incidence rules (exact expectations, household bootstrap)

Each rule is a probability vector over the same 64 saved states, taken from the rule code itself (`draw_displaced`, the youth-tilted preset, the compression top-tier tilt, the uniform flat exposure, and the top-loaded composite tilt). Expected displaced: 2.03m under every rule.

| Rule | Cost (SE) | Paper 50-draw cost | BHC poverty (SE) | Paper BHC | AHC poverty | Gini |
|---|---|---|---|---|---|---|
| Exposure-proportional | £7.26bn (1.07) | £7.3bn | +2.46 (0.11) | +2.40 | +2.61 | +1.78 |
| Junior | £6.90bn (1.07) | £7.0bn | +2.44 (0.11) | +2.43 | +2.59 | +1.78 |
| Compression | £11.29bn (1.14) | £11.3bn | +2.63 (0.12) | +2.65 | +2.76 | +1.77 |
| Uniform | £1.70bn (1.00) | £1.9bn | +2.47 (0.11) | +2.52 | +2.53 | +1.86 |
| Top-loaded | £27.70bn (1.45) | £27.7bn | +2.96 (0.14) | +3.00 | +3.33 | +1.74 |

Differences from exposure-proportional (bootstrap 95% intervals):

| Rule | Cost | BHC poverty | AHC poverty | Gini |
|---|---|---|---|---|
| Junior | −£0.37bn [−0.42, −0.31] | −0.02 [−0.03, −0.01] | −0.02 [−0.03, −0.01] | −0.01 [−0.01, −0.00] |
| Compression | +£4.03bn [3.57, 4.46] | +0.16 [0.11, 0.21] | +0.14 [0.09, 0.19] | −0.02 [−0.04, +0.01] |
| Uniform | −£5.57bn [−6.17, −5.01] | +0.00 [−0.05, +0.06] | −0.08 [−0.14, −0.03] | +0.07 [0.05, 0.10] |
| Top-loaded | +£20.44bn [19.33, 21.45] | +0.49 [0.39, 0.61] | +0.71 [0.61, 0.84] | −0.04 [−0.09, +0.02] |

Further consequences for the paper:

5. **The junior tilt is distinguishable from exposure-proportional incidence** once assignment noise is removed: a small but precise £0.37bn lower cost. The paper's "interval includes zero" (results, junior tilt) was assignment noise; the substantive point, that a within-occupation age tilt changes little, stands.
6. Costs order as top-loaded > compression > exposure > junior > uniform, all distinguishable. BHC poverty orders top-loaded > compression > {exposure, junior, uniform}; the last three are within 0.03pp of each other.
7. The Gini moves little across rules (1.74 to 1.86). Only uniform differs clearly (+0.07pp); the high-earner tilts slightly *lower* the Gini rise relative to exposure, not significantly.
8. Cost per displaced worker now follows exactly from the rule's earnings profile, with no OLS adjustment.

## Validation against the paper's own draws (`prototypes/replicate_draws.py`)

For each seed, the sampler's real displaced mask (the pipeline's own code) is mapped to the matching saved household state and summed.

- **Seeds 0–49 reproduce `results/robustness/incidence_draws_five.csv` exactly** for all five rules: maximum absolute difference £0.00005bn in cost, under 1e-14 in poverty, Gini and displaced count. The state engine is exact, not an approximation.
- **Long run (2,000 seeds of the real sampler, which assumes nothing about dependence within households):**

| Rule | Cost | BHC | AHC | Gini |
|---|---|---|---|---|
| Exposure | £7.24bn | 2.462 | 2.614 | 1.786 |
| Junior | £6.93bn | 2.430 | 2.587 | 1.772 |
| Compression | £11.30bn | 2.623 | 2.760 | 1.767 |
| Uniform | £1.64bn | 2.462 | 2.525 | 1.856 |
| Top-loaded | £27.69bn | 2.961 | 3.329 | 1.742 |

  These equal the exact enumeration within Monte Carlo error. Independence within the household is not driving the results.
- **Uniform minus exposure, BHC poverty:**
  - +0.115pp (SE 0.040) over the paper's seeds 0–49;
  - +0.00002pp (SE 0.007) over 2,000 seeds;
  - +0.003pp exact.

  The published gap was sampling noise in the 50 draws (about 3 SE from the true value), not a modelling artefact.

## Next steps (Phase 1 remainder, then Phase 2)

- ~~Extend the reweighting to the other three rules~~ (done above).
- Move state enumeration into `uk_ai_study/` as part of the Phase 2 rerun, when every result is regenerated.
- Wage-margin and mixed-margin scenarios: use the expected survivor gradient bill (decision needed at implementation).
