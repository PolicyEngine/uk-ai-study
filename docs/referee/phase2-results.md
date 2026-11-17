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
