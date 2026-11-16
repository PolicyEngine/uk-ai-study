# Phase 2 calibration: spells and re-employment (16 November 2026)

Status: **proposal for the author's sign-off** before the overnight run.

## Spell distribution

**Source.** ONS, *Unemployment by age and duration (seasonally adjusted), UNEM01*, People, UK, published 20 October 2026 (`unem01saoct2026.xls`). Series YBWF (up to 6 months), YBWG (over 6 and up to 12), YBWH (over 12) and the over-24-months series.

**Method.** UNEM01 counts the *stock* of unemployed by the time they have already spent unemployed. That over-represents long spells relative to a cohort entering unemployment. In a steady state, the stock's elapsed-duration density is proportional to the inflow survival function S(t). So I fit a piecewise-constant monthly exit hazard (0–6, 6–12, 12+ months) to the four stock shares (<6, 6–12, 12–24, 24+) and read the inflow distribution of completed spells from S(t). The fit is exact (residuals 0).

| Basis | Stock shares (<6 / 6–12 / 12–24 / 24+) | Monthly exit hazard | Inflow spell bins: 3 / 6 / 9 / 12+ months | Mean months out in year 1 |
|---|---|---|---|---|
| 2025 average | 59.1 / 17.8 / 12.9 / 10.2% | 0.206 / 0.192 / 0.068 | 60.4 / 17.8 / 9.6 / 12.2% | 4.5 |
| Latest 12 months (to Jun–Aug 2026) | 56.0 / 18.6 / 14.6 / 10.9% | 0.196 / 0.167 / 0.071 | 58.5 / 17.4 / 9.5 / 14.6% | 4.7 |

Bins: 3 months covers spells of 0–4.5 months, 6 covers 4.5–7.5, 9 covers 7.5–10.5, and 12 covers 10.5 or more (capped at the year).

**Proposal.** The 2025-average distribution (60 / 18 / 10 / 12%) for the central case; the latest-12-month distribution as a sensitivity; 12 months for everyone as the stress test.

**Caveats to state in the paper.**
- The distribution is for all unemployed people. AI-displaced workers are higher-earning, which in UK evidence goes with shorter spells (Upward and Wright 2019, below), so this is conservative on spell length.
- An AI shock large enough to displace 7 per cent of employees would itself lengthen spells, which a steady-state distribution does not capture. The 12-month stress test bounds this.

## Re-employment wage penalty

**Source.** Upward, R. and Wright, P. W. (2019). Don't look down: the consequences of job loss in a flexible labour market. *Economica* 86(341), 166–200. doi:10.1111/ecca.12254. UK workers 1991–2007. Displacement causes an immediate income loss of nearly 40 per cent and a long-run loss of about 10 per cent, driven mainly by lower pay on re-employment rather than lower employment.

**Proposal.** Re-employed workers earn 90 per cent of their pre-displacement pay for the rest of the year (10 per cent penalty). Sensitivities: 0 and 20 per cent.

## Consequence for the headline

The current paper's break-even spell length for the central cost is about 9.5 months. A mean of 4.5 months out of work therefore implies that, under the new central case, the Exchequer **cost is likely to become a net gain**, before the take-up/JSA and closed-capital changes, which move it in opposite directions. The job-loss poverty rise will also fall substantially. This changes the paper's headline numbers and possibly its framing; the waterfall from the current £7.26bn will show each step.

## Take-up target

To be measured from Microcosm before the run: UC take-up among entitled benefit units containing an unemployed adult (the referee quotes about 88 per cent).
