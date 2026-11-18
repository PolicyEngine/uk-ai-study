# Round 4: decisions taken while the author was unavailable (18 November 2026)

The author delegated the open revision questions to Fable 5.2 ("ask fable 5.2 your q, I'm not available"). Fable's answers follow. Decisions marked **AUTHOR TO CONFIRM** have not been acted on beyond what keeps options open.

## Taken (acted on)

1. **Baseline dividends (M2):** keep Microcosm's dividends; show band-by-band SPI scaling as a sensitivity (K = 3 reweighting). Under the closed account the UK payout is fixed by the accounting, so the dividend level only sets who receives it.
2. **Low and high presets (M1):** drop them as named presets. Replace them with one "scaling in the displacement rate" table at 1, 7 and 13 per cent, in two rows:
   - channel-free;
   - channels on, with the survivor uplift proportional to the rate (about 0.4, 2.6 and 4.8 per cent) and the dividend factor re-solved.

   Served by interpolation between the `uplift_u0`, `central` and `uplift_u5` state runs. The youth-tilted preset survives only as the age-tilt rule.
3. **Headline framing (M1, M6):**
   - Lead with the two channel-free contrasts: first the margin (job loss against wage cuts), then who loses the job.
   - Give levels second, as a bracket between the flow reading (calibrated spells) and the stock reading (12 months).
   - Break-evens leave the abstract.
   - "Seven per cent displaced" is explained as 7 per cent losing a job during the year, about 2.6 per cent of employee-years.
   - The stress test changes spell length only.
   - The published method becomes a reconciliation row.
   - The central Gini change is reported as indistinguishable from zero if its SE covers zero.
   - Write the margin sentence only once the per-pound pair is known at both 12-month and calibrated spells: part-year spells may narrow or flip the 42p against 55p gap.
4. **Wage-cut arm (M6):**
   - Headline and the 2×2 with channels off.
   - One secondary row with the closed capital account in both arms and no survivor uplift.
   - The gross loss matched to the expected private loss, including the 10 per cent penalty.
   - Four deterministic wage-cut runs: {uniform, exposure-graded} × {12-month, calibrated-spell matching}, private employees only, with the National Living Wage floor reported.
5. **Incidence rules (M12):** rename by mechanism:

   | Current | New |
   |---|---|
   | Exposure-proportional | unchanged |
   | Uniform | unchanged |
   | Junior-concentrated | Age tilt within occupation (under-25s × 1.29) |
   | Expertise compression | High-earner tilt (top earnings third of above-median-exposure occupations × 2) |
   | Top-loaded stress test | Top-loaded stress case |

   - "Four rules and one stress case" everywhere; hypotheses as motivation only, citing papers, not press releases.
   - Keep the age tilt as a row (its small effect is itself a result) but drop it from the abstract and introduction.
   - Add a one-parameter earnings-gradient family (probability ∝ exposure × earnings^γ), all reweighting.
6. **Design (item 7):**
   - Keep the 10 per cent penalty (0 and 20 per cent as sensitivities), q' = 0.75 (recomputed on the central set; flags-only and full take-up as bounds), the private-only treatment and the 2025-average spells.
   - 7b: the grid runs at calibrated spells, not 12 months; implemented as uplift points 0, 2.6 and 5 per cent at K = 3.
   - 7c: the stress test is spell length only.
   - 7d: add a sensitivity with survey-flag take-up in the shortest spell bin.
   - 7e: source the 50 per cent payout and the UK ownership share (ONS share-ownership data); add a 19 per cent corporation tax row; state that same-year payout is an assumption.
   - 7f: report the implied private-sector rate, the employment shortfall in employee-years, and K = 4 against K = 3.

Run changes made on 18 November:
- Settings are now `central_f100`, `central_f115` (K = 4), `uplift_u0` and `uplift_u5` (K = 3, calibrated spells), and `opencap` (published-method reconciliation, 12 months).
- The 12-month grid, the separate no-channels setting and the policy-reform runs are dropped.

## AUTHOR TO CONFIRM

- **7a. The survivor uplift is not scaled to the shorter spells.**
  - The central case pairs a spell-scaled labour loss (£53.4bn) with a full-year 2.6 per cent uplift (£29.0bn) calibrated to a permanent 7 per cent fall, so £2.54bn is again a residual of mismatched channels (the referee's M1 objection).
  - Scaling the uplift to effective labour lost (about 1.1 per cent) would put the cost near £5bn (Fable's estimate).
  - **Recommendation:** keep 2.6 per cent as the stated central case (approved, traceable to Doorley et al.), but report the level as a range over uplift 0 to 2.6 per cent, with the scaled value as a named row, and quote that range in the abstract. The `uplift_u0` runs make this possible without deciding now.
- **3. The policy section (M14).**
  - **Recommendation:** cut it to a paragraph in the discussion. With a poverty rise of +0.36pp (SE about 0.11), "share of the shock-induced rise undone" has a denominator about three SEs from zero, and cost per poverty point for general UC reforms mostly measures the reform in the baseline.
  - Keep an insurance metric for the existing system (net replacement rate of displaced households by spell and decile), which needs no reform runs.
  - If kept instead: an appendix with R1 (renamed "earnings replacement", plus a taxed variant) and R2 only, under the 12-month stress test.
  - The reform runs were dropped from tonight's plan, so either choice remains open.

## Correction found while porting

The Phase 2 prototype averaged household income across spell lengths and take-up modes before testing it against the poverty line, which understates poverty and the Gini. `statesim/results.py` treats spell and take-up mode as state dimensions (one spell per displaced household) and scores take-up in levels against the survey-flag baseline. The prototype figures of +0.36pp and +0.05pp are therefore biased low; the regenerated numbers replace them.
