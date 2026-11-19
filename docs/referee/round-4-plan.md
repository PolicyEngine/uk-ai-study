# Referee round 4: revision plan (16 November 2026)

The report is in [round-4-report.md](round-4-report.md): major revision at a field journal (JPubE / AEJ: Policy), with no arithmetic or build errors. This plan covers every major comment (M1–M16) and the minor comments. Nothing here changes results until the phase it names; each phase lands as its own PR, re-attested, with the paper rebuilt from generated macros.

Labels: **T** text only, **R** re-basing existing runs or cheap new runs (minutes to an hour), **A** new analysis needing code changes and a rerun (hours, overnight).

## Principles

- Every number stays a generated `\gen` macro; no hand-typed values.
- One full overnight rerun at the end of Phase 2 replaces the current results; later phases add stages to the manifest rather than rerunning everything.
- Each phase ends with `rebuild.py --check`, `pytest`, a Fable review of the changed sections, and a PR left for the author to merge.

## Phase 1: design decisions and assignment (M3, M4) — A

The single change that most improves the paper: remove assignment noise so that no result rests on seed 0.

1. **Choose the assignment method** (half a day, decision note in `docs/referee/assignment.md`).
   - *Record splitting* (referee's preference): each employee record becomes a displaced copy with weight p·w and a retained copy with weight (1−p)·w. PolicyEngine weights are household-level, so exact splitting needs 2^k household clones for k employees. Feasible for k ≤ 3 (check the share of households), with an independence approximation above that.
   - *Fallback:* weight-aware systematic sampling with a fixed weighted quota, displacement sets nested across rates, and at least 20 draws behind every reported number, plus antithetic or control-variate variance reduction.
   - Decide on run time, exactness, and how far it changes `shocks.py` and the runner.
2. **Implement** in `uk_ai_study/shocks.py` (`prescribed_systematic_sample`, `apply_*`), the runner and checkpoint fingerprints; unit tests for expectation and nesting.
3. **Survey-sampling uncertainty (M4):** bootstrap over households (or replicate weights, if Microcosm supplies them) for the main contrasts: rule differences, margin differences, pence per pound. New stage `analysis/bootstrap_contrasts.py`.
4. Remove the OLS "adjust to the mean displaced count" in the index table (minor 16) once assignment is deterministic.

## Phase 2: baseline and accounting choices (M2, M7, M8, M9, M10) — A, then the overnight rerun

5. **Benefit baseline (M7):** new-style JSA on for eligible displaced workers; take-up among the displaced at the observed rate for the unemployed in the data (~88% of entitled), with the current setting kept as a lower-bound variant. Check whether council tax reduction can respond; sign redundancy and notice pay.
6. **Part-year spells (M8):** simulate spells properly (monthly UC assessment, annual tax on part-year earnings), with a spell distribution and re-employment wage penalty calibrated to UK displacement evidence (Upward and Wright, 2019). Report the impact-year stress test beside a steady-state flow (about 0.7% a year with turnover). If PolicyEngine UK cannot take a spell input, use monthly periods for the displaced and document the method.
7. **Capital channel and closure (M2):** make the closed account central. Tie the profit increase to the displaced *private* wage bill (labour cost saved, including employer NI and pension contributions), apply corporation tax and a payout ratio, then a UK-household ownership share. Report implied GDP and labour-share changes. Keep the 39.8% open-economy uplift as a sensitivity. Use SPI-scaled dividends as the baseline, or show both throughout. Add a CGT / deferred-tax sensitivity on retained profit.
8. **Public sector (M9):** three variants: public-sector employees excluded with the quota reallocated; a consolidated account netting the pay bill saved and intra-government employer NI; the current treatment. Restrict φ to the private wage bill.
9. **Omitted fiscal lines (M10):** signed bias table (VAT and indirect taxes at a stated MPC and effective rate, CT, CGT, debt interest, pension relief). Add one incidence identity for £1 of gross labour cost removed that sums to 100: households, Exchequer by line, employers/profit, pension saving.
10. **Overnight rerun** of every stage under Phases 1–2 (the same driver and finalize scripts as the 2027-28 rerun), then attest.

## Phase 3: the margin result as the contribution (M5, M6, M15) — R/A

11. **2×2 design:** {uniform, exposure-graded} × {job loss, wage cut}, without the wage and capital channels. New stage in `analysis/wage_cut_household.py`.
12. **Dolls-style replication:** a proportional 5% income shock and a reweighting-based unemployment shock with full take-up, so the UK coefficient sits beside Dolls et al.'s 41.5%.
13. **PTR/EMTR:** distributions for the displaced and the exposed, checked against published UK figures.
14. **Wage-cut feasibility:** report the implied cut by group; handle the National Living Wage floor; add an hours-cut variant.
15. **Abstract, intro, conclusion (M6, M15):** lead with the channel-free comparison (pence per pound; poverty 0.92 vs 2.88pp) and a one-sentence contribution with a number, positioned against Dolls et al.

## Phase 4: exposure and incidence rules (M11, M12) — R/A

16. **Normalisation:** quota rates and headlines under percentile rank, the positive raw scale, and a non-zero floor for the least-exposed group. Confirm the scale Pizzinelli et al. apply complementarity to.
17. **SPI / imputed-occupation records:** their share of cost and displaced earnings; a run excluding or capping them.
18. **Within-group variation:** LFS/APS 4-digit distributions within major group, or a bound on their effect.
19. **Rules (M12):** rename by what they do ("high-earner tilt", "age tilt within occupation"); cite estimates, not press releases; implement expertise compression as within-occupation wage compression, and the junior margin as non-entry of a cohort; or present incidence as a continuous earnings-gradient family.

## Phase 5: validation (M13, M16) — R/A

20. **Baseline validation table:** 2027-28 baseline against official aggregates (Gini, absolute and relative poverty, income tax, NI, UC spending and caseload, wage bill, dividends). Explain the poverty-line choice (minor 2).
21. **Backcast:** rerun with 2008-10 policy parameters if available, otherwise relabel it an illustration and use it to discount poverty levels.
22. **OBR on a common basis:** single year at the full gap, add VAT and CT from the closure, JSA and observed take-up, uniform incidence.
23. **Ireland:** quantitative comparison on Doorley et al.'s metrics, or drop the table.
24. **Decile-1 artefact (M16):** fix or exclude the 333 high-pension-contribution households throughout, or make the pre-contribution ranking the default. Add a replicator access route and a data appendix.

## Phase 6: policy section (M14) — R/T

25. Report each reform in the unshocked baseline and the increment due to the shock; add an insurance metric (replacement rate for the displaced; share of the shock-induced poverty rise undone); rename R1 (earnings replacement) and add a taxed variant; consider a more generous contributory benefit; or cut the section to a paragraph. Rerun or drop the 5-draw duration ranking.

## Phase 7: text and presentation (M1, M12 wording, minor 1–28) — T

26. **Headline (M1):** "I simulate", not "I estimate"; levels only as ranges over a stated parameter box; rebuild or drop the low/high presets so the capital and wage channels scale with displacement.
27. **Minor comments 1–28:** units (Gini 30.3), rounding harmonised or footnoted, one name for the incidence rules, seed/draw labels on every table, Table 4 header and pension caveat, figure conventions and weighting, "assignment only" intervals, by-construction claims, JSA rate uprated or explained, conjecture cut, ripple appendix rewritten or cut, capital-only cell qualified, realised wage-bill growth, mean baseline earnings, Dolls UK figure moved to the literature, a regional note, forward definition of the OBR-sized run, self-employed share and direction of bias.
28. **Literature (M15):** EUROMOD/UKMOD stress tests, displaced-worker earnings losses, UK PTR/EMTR measurement, wage insurance, the self-employed and freelancers, UK AI job scenarios with fiscal estimates (each verified before citing).

## Order and effort

| Phase | Type | Effort | Depends on |
|---|---|---|---|
| 1 Assignment, bootstrap | A | 2–3 days | — |
| 2 Baseline and accounting, then overnight rerun | A | 3–4 days + 1 night | 1 |
| 3 Margin contribution | R/A | 1–2 days | 2 |
| 4 Exposure and rules | R/A | 1–2 days | 2 |
| 5 Validation | R/A | 1–2 days | 2 |
| 6 Policy | R/T | 1 day | 2 |
| 7 Text and minor | T | 1–2 days | 3–6 |

Phases 3–6 are independent of each other once Phase 2 lands and can run in parallel. The paper on main stays as published until the revision merges.

## Tracking

Issue: PolicyEngine/uk-ai-study#25 (one checkbox per item above). Each phase PR references its items.
