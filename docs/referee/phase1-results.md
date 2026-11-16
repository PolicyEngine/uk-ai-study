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

## Next steps (Phase 1 remainder, then Phase 2)

- Extend the reweighting to the other three rules (junior, compression, top-loaded); they are probability vectors over the same states.
- Move state enumeration into `uk_ai_study/` as part of the Phase 2 rerun, when every result is regenerated.
- Wage-margin and mixed-margin scenarios: use the expected survivor gradient bill (decision needed at implementation).
