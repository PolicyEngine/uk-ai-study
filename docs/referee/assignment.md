# Phase 1 decision note: removing assignment noise (referee M3)

Status: **proposal for the author's decision**. Nothing in `uk_ai_study/` has changed.

## The problem

Each employee in an occupation group has the same displacement probability (the group's quota rate, optionally youth-tilted), and records are chosen by systematic sampling over a random permutation. Grossing weights are very uneven (Kish effective size of all employees: 5,810 of 72,504 records), so the weighted number displaced varies from draw to draw. Over 200 draws at the central rate, the realised displacement rate has a mean of 7.02 per cent, a standard deviation of 0.31 points and a range of 5.84 to 7.87 per cent. This drives the £1.9bn draw standard deviation of the central cost, the need for 50 draws, the seed-0 problem, and the OLS adjustment in the index table.

## Diagnostics (Microcosm UK 2024-25, 2027-28)

Employees per household, among households with at least one employee:

| Employees | Households | Share (records) | Share (weighted) |
|---|---|---|---|
| 1 | 23,107 | 50.2% | 46.1% |
| 2 | 20,059 | 43.6% | 43.1% |
| 3 | 2,283 | 5.0% | 8.5% |
| 4 | 479 | 1.0% | 1.7% |
| 5–6 | 101 | 0.2% | 0.5% |

Exact record splitting at household level needs one clone per subset of a household's employees: 155,898 household records, a new dataset.

Realised displacement rate over 200 draws at 7 per cent:

| Method | Mean | SD | Range |
|---|---|---|---|
| Current (random permutation) | 7.02% | 0.31 pt | 5.84–7.87% |
| Weight-sorted systematic sampling | 7.00% | 0.065 pt | 6.83–7.15% |

## Options

**A. State enumeration (recommended).** A household's shocked outcome depends only on which of its employees are displaced: the survivor uplift is normalised over *baseline* workers (`shocks.py`, `theta_bar`), and the capital shock does not depend on displacement. So run one simulation per bitmask over each household's first three employees (eight runs), save per-household outcomes, and compute any displacement rate or incidence rule as a reweighting, `w_h × P(state)`, with `P(state)` the product of the members' displacement probabilities (independence within household).

- Exact expectations for 98 per cent of employee households (weighted). For the 2.2 per cent with four or more employees, the fourth and later employees are either drawn (small residual noise) or handled by a 16-run extension.
- No draws, no seed 0, no OLS index adjustment. Every rule and rate is exactly comparable.
- Cheaper than now. Eight runs per (wage-uplift, capital) setting replace 50 draws per rule. The 66-cell grid needs 6 × 8 = 48 runs instead of 66 single draws, and is exact.
- Poverty and the cost are linear in the household outcomes, so they are exact expectations. The Gini is computed on the expected population: each household state carries weight `w_h × P(state)`. This differs slightly from the mean of per-draw Ginis; the note in the paper says so.
- Limits:
  - Scenarios whose shock depends on the realised set need care. The mixed-margin wage cut is normalised to the survivors' gradient bill; use its expectation. Ripple effects depend on aggregate displacement by group; those are deterministic given the rates.
  - Independence within the household replaces the mild negative dependence that systematic sampling induces.

**B. Weight-sorted systematic sampling (fallback).** Same inclusion probabilities; the realised-rate SD falls fivefold. Risk: household members share a weight and sit next to each other after sorting, so systematic sampling rarely picks two from one household. That suppresses joint displacement, which matters for poverty. Measure the co-displacement share against the independence benchmark before using it. It still needs draws, at least 20 for every number.

**C. Exact record splitting.** Needs a 156,000-household dataset with recalibrated weights. More work than A for the same result.

## Acceptance test

The deterministic central case must land within about 2–3 Monte Carlo standard errors of the current 50-draw means: cost £7.3bn (SE about £0.27bn), BHC poverty 2.40pp, Gini 1.76pp. If not, the independence assumption or the state coverage is wrong.

## Implementation plan, if A is chosen

1. Prototype in a new, unstamped `analysis/state_enumeration.py`: eight runs at the central setting; check the acceptance test. Nothing in `uk_ai_study/` changes, so the current paper still builds.
2. Move the method into `uk_ai_study/` only in the Phase 2 rerun, when every result is regenerated anyway.
3. Survey-sampling uncertainty (M4): bootstrap households on the saved per-household state outcomes, a cheap reweighting with no new simulations.
