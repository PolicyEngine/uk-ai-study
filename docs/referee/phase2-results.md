# Phase 2 prototype results: the new central case (17 November 2026)

Prototype only (`prototypes/phase2_*.py`), K = 3 household displacement states (98.7 per cent of expected displaced workers). The final numbers come from the K = 6 run when the method is ported into the package. Nothing in `uk_ai_study/` or the attested results has changed.

Author's decisions applied:
- JSA and observed take-up (q' = 0.75);
- calibrated spells (ONS UNEM01, mean 4.48 months out) with a 10 per cent re-employment penalty (Upward and Wright 2019);
- the closed capital account (25 per cent corporation tax, 50 per cent payout, 50 per cent to UK households);
- private-sector displacement only (7 per cent of all employees);
- the survivor uplift for private-sector employees only.

## Waterfall (K = 3, 7 per cent displacement)

| Step | Cost | BHC poverty | AHC poverty | Gini |
|---|---|---|---|---|
| Published method: open 39.8% capital, all employees, 12-month spells | £6.90bn | +2.46 | +2.61 | +1.79 |
| No capital uplift; private-sector displacement | £25.78bn | +2.75 | +2.93 | +1.06 |
| + new-style JSA | £29.40bn | +2.51 | +2.64 | +0.91 |
| + observed UC take-up (q' = 0.75) | £30.85bn | +2.40 | +2.54 | +0.81 |
| + calibrated spells and 10% penalty (bins at 3/6/9/12) | £9.96bn | +0.39 | +0.37 | +0.02 |
| + private-only survivor uplift; exact spell weights on 0/3/6/9/12 | £10.32bn | +0.36 | +0.33 | −0.02 |
| **+ closed capital account (new central case)** | **£2.54bn** | **+0.36** | **+0.32** | **+0.05** |

The first row reproduces the Phase 1 exact central case for K = 3. The 12-month case through the monthly benefit mix (£30.94bn, +2.38pp) matches the annual run (£30.85bn, +2.40pp).

## Closed capital account (new central case)

| Item | Value |
|---|---|
| Share of annual pay lost per displaced worker (spells + penalty) | 0.436 |
| Labour cost saved by private employers (pay, employer NI, employer pensions) | £53.4bn |
| Survivor uplift paid by private employers (with employer NI) | £29.0bn |
| Profit increase | £24.5bn |
| Corporation tax (25%) | £6.1bn |
| UK-household dividend payout (50% × 50% of post-tax profit) | £4.6bn |
| Dividend factor on baseline dividends (£95.9bn) | 1.048 |

The household outcomes at the dividend factor are interpolated between runs at ×1.00 and ×1.15. A direct run at ×1.061 matched the interpolation to £0.01bn (£7.861bn against £7.85bn).

## What changes in the paper

- The headline cost falls from £7.3bn to about £2.5bn. The job-loss poverty rise falls from +2.4 to about +0.4 points, and the Gini rise from +1.8 to about +0.05 points.
- Spells dominate the change: mean 4.5 months out of work against the published 12.
- The capital channel is now tied to the labour shock. It raises receipts through corporation tax and modest dividend income, instead of a 39.8 per cent uplift on all interest and dividends.
- The 12-month, no-take-up, open-capital case stays as a stress test.

## Remaining before the paper changes

1. Port state enumeration, the Phase 2 options and the closed account into `uk_ai_study/`, with unit tests.
2. Rewrite the stages per the stage inventory.
3. Run K = 6 overnight (grid and policy reforms at 12-month spells, per the agreed run budget).
4. Regenerate macros and re-attest.

Then Phases 3–7: margin contribution, exposure rules, validation, policy and text.

## Package run, K = 4 (19 November 2026): corrected central case

From `statesim/results.py` on the `central_f100` and `central_f115` state runs (448 simulations, 99.8 per cent of expected displaced workers covered; the residual is 4,600 of 2.03 million). This includes the fix for the prototype's averaging error: poverty and the Gini are now taken over the joint distribution of displacement state, take-up mode and spell length. Raw output: `docs/referee/data/central_k4_2026-11-19.json`. Bootstrap SEs are still to run.

| Rule / variant | Cost | BHC poverty | AHC poverty | Gini |
|---|---|---|---|---|
| **Exposure-proportional (new central)** | **£2.94bn** | **+0.52** | **+0.46** | **+0.18** |
| Age tilt within occupation | £2.85bn | +0.52 | +0.46 | +0.18 |
| High-earner tilt | £3.63bn | +0.55 | +0.49 | +0.16 |
| Uniform | £1.86bn | +0.54 | +0.47 | +0.24 |
| Top-loaded stress case | £6.63bn | +0.51 | +0.56 | +0.07 |
| Stress test: 12-month spells (all else central) | £5.50bn | +2.35 | +2.55 | +1.04 |
| Survey-flag take-up (q' = 0) | £2.26bn | +0.62 | +0.55 | +0.22 |
| No closed account (no capital channel, no corporation tax) | £10.72bn | +0.53 | +0.47 | +0.11 |

Closed account (exposure, calibrated spells): labour cost saved £53.4bn, survivor uplift paid £29.0bn, profit £24.5bn, corporation tax £6.1bn, UK-household payout £4.6bn (dividend factor 1.048).

Notes:
- The prototype's +0.36pp and +0.05pp were biased low, as flagged in round-4-decisions.md. The corrected central case is +0.52pp BHC and +0.18pp Gini.
- The top-loaded case now costs £6.6bn, not £27.7bn. Under the closed account, high earners' lost wages return largely as profit taxed at 25 per cent, so the rule mostly shifts the incidence of the tax base, not its size.
- The 12-month stress test recovers a poverty rise close to the published +2.4pp, while the cost (£5.5bn) stays below the published £7.3bn because the closed account adds corporation tax.
