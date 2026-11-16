# Phase 2 design: the new central case (16 November 2026)

Decisions taken by the author (all four as recommended):

| Item | Referee | Central case | Kept as variants |
|---|---|---|---|
| Benefits | M7 | New-style JSA where the contribution test holds; displaced UC take-up at the rate observed for existing unemployed benefit units | Survey flags, no JSA (lower bound); full take-up + JSA |
| Spells | M8 | Calibrated spell distribution: monthly benefit mix and annual tax on part-year earnings | 12-month spells (stress test) |
| Capital | M2 | Closed account: profit from the displaced private labour cost, 25% corporation tax, 50% payout, 50% to UK households | 39.8% open-economy uplift on interest and dividends |
| Public sector | M9 | Private-sector employees only displaced; same aggregate rate (7% of all employees) | Consolidated account; current treatment |

Everything builds on the Phase 1 state enumeration: no random draws anywhere. Every expectation is a reweighting of saved household outcomes.

## 1. Who can be displaced

- Universe: employees with `employment_sector != PUBLIC`.
- The quota stays at 7 per cent of *all* employees, allocated across SOC groups in proportion to private employment × mean shifted exposure. This raises the private-sector rate.
- State enumeration ranks private employees within the household (K = 6 still covers every household).
- Report the implied private rate and the share of the quota each group absorbs.

## 2. Benefits (M7)

UC take-up is linear in the household outcome, so it needs no draws. For each state, run twice:

1. with the survey `would_claim_uc` flags;
2. with `would_claim_uc` set to True for every benefit unit containing a displaced person (as the existing `takeup1` variant does).

For a displaced benefit unit whose survey flag is False, the expected outcome is `q' × Y(claim) + (1 − q') × Y(flag)`. Here `q' = (r − r₀)/(1 − r₀)`, where `r` is the take-up rate among entitled existing unemployed benefit units in the data (target ≈ 88%, to be measured exactly) and `r₀` is the rate implied by the survey flags among entitled displaced units.

New-style JSA is included in the displaced-state runs, as in the existing `jsa` variant: `jsa_contrib_reported` for displaced people aged 18–65 meeting the class 1 contribution condition, for min(spell, 26 weeks).

## 3. Spells (M8)

PolicyEngine UK computes UC and income tax annually, but UC is assessed monthly on that month's income. For a spell of `s` months:

- **Benefits** (UC and other means-tested awards): `(s/12) × B(out-of-work state) + ((12−s)/12) × B(in-work state)`. This is exact for monthly assessment when capital and household composition are unchanged within the year.
- **Income tax and NI:** annual, on part-year earnings `(12−s)/12 × earnings × (1 − wage penalty after re-employment)`, from one run per spell length with the displaced earnings scaled.
- **JSA:** `min(s, 6)` months.
- **Spell distribution:** discrete lengths {3, 6, 9, 12} months, with shares from UK evidence on the duration of unemployment after displacement (LFS unemployment duration by previous-job loss; Upward and Wright 2019 for re-employment wage penalties). The sources and exact shares get a short calibration note, verified before use, and need the author's sign-off.
- 12 months for everyone stays as the stress test reported beside the central case.

This needs `gov_balance` split into its tax and benefit components per household, which the runner already computes (`cost_component_totals`).

## 4. Capital: the closed account (M2)

- Labour cost saved by private employers: displaced private earnings forgone (spell-adjusted) plus employer NI and employer pension contributions on them, net of survivor wage gains paid by private employers.
- Profit rises by that amount (φ = 1 central; φ ∈ {0.5, 0.75} as sensitivities).
- Corporation tax at 25 per cent is added to the Exchequer balance as an accounting line (it is outside the household microsimulation).
- 50 per cent of post-tax profit is paid out, and 50 per cent of the payout reaches UK households.
- The household dividend increase is allocated in proportion to baseline dividend income: a dividend scale factor solved so the total equals the UK-household payout. Interest income is not scaled.
- Retained profit and the non-UK payout carry no household income. Their tax treatment (CGT, deferred tax) is a sensitivity row.
- The baseline is SPI-scaled dividends (referee M2) and the unscaled baseline is shown. *Decision deferred to implementation: confirm with the author before switching the baseline dividend level.*
- Report implied GDP and labour-share changes from the same accounting.
- The 39.8 per cent open-economy uplift stays as a sensitivity row.

## 5. Omitted fiscal lines (M10)

A signed bias table: VAT and indirect taxes (stated MPC and effective rate), corporation tax (now in the central case), CGT, debt interest, pension relief. Plus one incidence identity for £1 of gross labour cost removed that sums to 100: households (by line), Exchequer (by tax and benefit), retained profit, non-UK owners, pension saving.

## 6. Implementation and runs

1. `uk_ai_study/assignment.py` (new): state enumeration, rule probability vectors, reweighting, bootstrap. Ported from the Phase 1 prototypes, with unit tests: probabilities sum to one, expectations match brute-force enumeration on a toy household table, nesting.
2. `uk_ai_study/shocks.py`: private-sector universe; JSA and take-up options; spell scaling of displaced earnings; dividend allocation for the closed account.
3. `uk_ai_study/runner.py`: per-household tax/benefit components per state; spell mixing; closed-account lines.
4. Rewrite the analysis stages to consume state outputs instead of seeds: monte_carlo_families, grid, presets, wage margin, closure, index, ripple, caseloads, duration, take-up, OBR. Most become reweightings, and the manifest shrinks.
5. **Run budget** (about 30 s per state run):
   - central setting: 64 states × 2 take-up × 4 spell lengths ≈ 512 runs, about 4–5 hours;
   - grid wage uplifts: 6 × 64 states;
   - wage-margin and other settings: a few more.
   This fits one overnight run, with checkpointing per state.
6. Acceptance:
   - With the Phase 2 options switched off, the new code reproduces the Phase 1 exact central case (£7.26bn, +2.46pp, +1.78pp).
   - With them on, every number regenerates and attests, the paper builds with no pending macros, and the tests pass.

## Open points for the author before the overnight run

- Spell distribution shares and the re-employment wage penalty, with sources.
- Whether to switch the baseline to SPI-scaled dividends.
- The exact take-up target `r`, once measured.
