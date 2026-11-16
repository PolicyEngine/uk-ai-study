# Phase 2 stage inventory (16 November 2026)

Under state enumeration, a stage costs nothing new if it only changes *who* is displaced (a probability vector over saved states). It needs new state runs if it changes *what happens* to a household in a given state (wage uplift, capital, wage cuts, reforms, baseline data). Each distinct setting costs 64 state runs (K = 6, about 30 s each, about 32 minutes). The Phase 2 central case multiplies that by 2 for take-up and by the spell lengths that need a tax run.

## Settings (each one is a set of state runs)

| Setting | Wage uplift | Capital | Other | Used by |
|---|---|---|---|---|
| S1 central | 2.6% (theta-graded) | closed account (dividend factor solved) | JSA, take-up mix, spells | presets (central, youth, high), incidence families, measured incidence, index sensitivity, wage tier, caseloads, gender, poverty decomposition, appendix deciles and decomposition, figures |
| S2 grid | 0, 1, …, 5% (6 runs of S1) | closed account | as S1 | jr16 grid, appendix grids |
| S3 no channels | 0 | none | as S1 | low preset (1%, no uplift), tax composition (job-loss absorption), OBR benchmark (1.67%, no uplift, no capital), OBR household deciles |
| S4 open capital | 2.6% | 39.8% uplift | as S1 | capital-variants sensitivity, comparison with the published central case |
| S5 SPI dividends | 2.6% | closed, on SPI-scaled dividends | as S1 | dividend-coverage sensitivity (or the new baseline, if adopted) |
| S6 policy reforms | 2.6% | closed | R1, R2, R3 each | policy counterfactuals (3 × 64) |
| S7 wage cuts | — | — | definition needed | wage cut, mixed wage adjustment (see open point 1) |
| S8 backcast | 2008–10 shock | — | — | recession backcast (or drop it, per referee M13) |

## Stages that become probability vectors only (no new runs)

incidence, measured_incidence, monte_carlo (families), index_sensitivity, wage_tier, the presets' rates (central, youth-tilted, high 13%, low 1% on S3), and the displacement rates of the grid. The OLS index adjustment is deleted.

## Outputs to save per state (beyond Phase 1)

To serve the appendix, caseloads, gender and decomposition stages without rerunning, each state run saves per household:
- the tax and benefit components of `gov_balance`: income tax, employee NI, employer NI, UC, other benefits;
- market income;
- the UC benefit-unit count and claim flag.

And per person: HBAI per-capita income (for the gender and age splits).

## Run budget

| Setting | States | × take-up | × spell tax runs | Runs | Hours (30 s) |
|---|---|---|---|---|---|
| S1 | 64 | 2 | 4 (3, 6, 9, 12 months) | 512 | 4.3 |
| S2 | 6 × 64 | 2 | 4 | 3,072 | 25.6 |
| S3 | 64 | 2 | 4 | 512 | 4.3 |
| S4 | 64 | 2 | 1 (12 months) | 128 | 1.1 |
| S5 | 64 | 2 | 4 | 512 | 4.3 |
| S6 | 3 × 64 | 2 | 4 | 1,536 | 12.8 |

All of it exceeds one night. Options in open point 2.

## Open points for the author

1. **Wage-cut arm under the new central case.** "Same gross loss" needs a definition when job losses are spell-weighted and public employees are not displaced. Proposal: the wage cut removes the same expected gross earnings as the spell-weighted private-sector job losses, applied to private employees only, graded by exposure as now.
2. **Run budget.**
   - Proposal: run the grid (S2) and the policy reforms (S6) at 12-month spells only (one tax run instead of four), and report spells for the central setting.
   - This brings S2 to 768 runs (6.4 h) and S6 to 384 runs (3.2 h). With S1, S3, S4 and S5, the total is about 20 hours: two nights, or one night plus a day, with per-state checkpoints.
   - Alternatively, K = 3 for the grid (8 states, covering 98.7 per cent of displaced workers) cuts S2 by 8×.
3. **Whether to keep the recession backcast** (referee M13: rerun with 2008–10 policy or relabel it an illustration).
4. **Spell shares, re-employment wage penalty and the take-up target:** sourced and measured, not assumed, and signed off before the run.
