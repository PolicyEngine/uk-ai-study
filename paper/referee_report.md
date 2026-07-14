# Referee Report (simulated, Claude agent, 2026-07-14)

**Manuscript:** "Who Bears the AI Shock? Displacement, Poverty and the Public Finances in the UK"
**Journal fit assessed for:** Fiscal Studies / Oxford Review of Economic Policy type outlet

## 1. Summary

The paper builds a scenario-based microsimulation of an AI labour-market shock for the UK. Occupation-level AI exposure (Felten et al.'s AIOE, complementarity-adjusted following Pizzinelli et al., crosswalked to SOC2020 one-digit major groups) is attached to FRS 2024-25 workers, and a stylised shock — centrally 7 per cent of employees displaced, a 2.6 per cent aggregate wage uplift for survivors, and a 0.4pp capital-return uplift — is passed through PolicyEngine UK to recover fiscal, poverty and inequality effects. The distinctive move is to treat the *incidence* of displacement, not just its size, as the scenario axis: the same 1.61 million displaced workers are reallocated across five incidence families (exposure-proportional, junior-concentrated, expertise-compression, uniform, and a "measured" family loosely calibrated to one UK firm-level study). Headline findings: the Exchequer cost of an identical aggregate shock varies from £14.2bn to £32.5bn depending on who bears it; BHC poverty rises 1.8-2.3pp with much less cross-family variation; and the Gini rises in every one of 66 shock-size cells and all five families. Extensions cover three simulated fiscal responses (UC uplift dominates wage insurance in poverty averted per pound), a corporate-tax "recomposition" accounting exercise, and a constituency-level mapping.

## 2. Overall assessment and recommendation

**Recommendation: Major revision.**

The paper's central conceptual point — that the incidence of an AI shock is a live empirical disagreement, and that running the competing incidence views through a full tax-benefit system quantifies what is at stake in that disagreement — is genuinely useful and, to my knowledge, novel. The £14-£32bn fiscal spread on a fixed aggregate shock, and the "cheap incidence is not benign incidence" decoupling of the fiscal and poverty margins, are policy-relevant findings that a fiscal-policy readership will value. The open-source, replicable implementation is a real merit, and the paper is unusually candid about its assumptions: nearly every limitation I intended to raise is disclosed somewhere.

However, candour is not a substitute for resolution, and several disclosed limitations are not merely caveats — they materially weaken headline claims. The "measured" incidence family, which delivers the paper's most-quoted number (£32.5bn) and appears in the abstract, is built on parameters taken from a *press release* of a single unrefereed SSRN working paper whose full text the authors admit they have not read, combined with several author-imposed multipliers. That number should not headline the paper in its current form. The one-digit SOC exposure basis, the static no-re-employment design, the absence of any behavioural response, the uncalibrated plain-FRS weights, and the constituency exercise built on imputed occupations on a *different* dataset with roughly double the national poverty response are individually defensible as first-pass simplifications, but their combined effect is that the paper's quantitative headline numbers (£18.4bn, +1.87pp poverty) carry far more apparent precision than the framework can support. The qualitative architecture is sound; the quantitative claims need to be either hardened or demoted. Hence major revision rather than minor: the required changes concern which results the paper is allowed to claim, not just how it says them.

## 3. Major comments

**M1. Provenance of the "measured" incidence family is inadequate for a headline result.** The parameters (junior multiplier 1.29, London multiplier 1.5, wage-tier multipliers 9.6/0.6) come from a press release and secondary summaries of Klein Teeselink (2025), "pending verification against the SSRN full text". The London magnitude, the low-wage floor, the multiplicative independence of the three tilts, and the under-25 proxy are all author-imposed. Yet the £32.5bn / +2.29pp figures appear in the abstract and executive summary as the "costliest" family, and Section 4.5 says the central preset "understates both the fiscal and the poverty cost" to the extent this family is a better guide. A number built on an unread source cannot bear that weight. Either verify the parameters against the full paper and report sensitivity to each imposed multiplier — the 9.6/0.6 wage-tier ratio in particular looks like the dominant driver of the £32.5bn — or move the family to an appendix clearly labelled illustrative and strip it from the abstract.

**M2. Static, full-year, no-re-employment displacement drives the fiscal magnitudes and is asymmetrically disclosed.** Every displaced worker loses a full year of earnings with zero re-employment, zero replacement hiring, and no labour-supply or wage-bargaining response by anyone else. With any plausible re-employment hazard, the £18.4bn is an upper bound on the first-year cost, and the cross-family *spread* — the paper's central object — would also compress, since top-loaded incidence is expensive precisely because high earners lose a full year of highly taxed income; if high earners re-employ faster, the £14.2-£24.5bn spread shrinks. State explicitly that all fiscal figures are annualised upper bounds conditional on zero re-employment, and add at least a crude sensitivity (e.g. scale earnings losses by an assumed 6-month mean out-of-work duration, overall and differentially by family). If the incidence spread is not robust to that, the central claim needs re-stating.

**M3. One-digit SOC exposure is a coarser constraint than the paper concedes.** With nine major groups, the entire incidence machinery operates on nine exposure numbers. This matters most for the decile transition gradient, the poverty response, and especially the "expertise-compression" and "measured" families, whose within-occupation logic is fundamentally sub-major-group and is proxied by person-level earnings tilts within one-digit cells: two of the five incidence families are conceptually defined at a resolution the data cannot represent. The LFS-based imputation of finer codes flagged as "planned" should be executed for at least the central scenario before publication. The exposure-index sensitivity in A.7 does not address resolution, only index choice, and all five indices are aggregated to the same nine groups — the quota mechanism plus identical group rankings nearly guarantees the reported stability.

**M4. Single seed-0 draws for headline tables.** Tables 1, 2, 3, 4, 6, the grid figures and the decomposition are single draws at seed 0; Monte Carlo evidence is confined to the central scenario (20 draws, SD £1.5bn). A £1.5bn SD is not negligible against cross-family gaps of £1.0bn (exposure vs junior) — Table 2's ordering could plausibly flip across draws, yet the text narrates the ordering as a finding. Report Monte Carlo means and SDs for all five families (and ideally the reform table), and only narrate orderings that survive the draw noise.

**M5. The expertise-compression family is an author-designed construct presented alongside "schools of thought".** Its numbers are direct functions of the arbitrary multiplier (2 toward the top earnings tertile). Add sensitivity in the multiplier (1.5, 2, 3) to show whether "compression is the most expensive stylised family" is a property of the mechanism or of the chosen tilt strength. Same for the junior multiplier.

**M6. The constituency geography is a different exercise wearing the same paper's clothes.** Different dataset (enhanced FRS 2023-24), different year, imputed occupations, and a national BHC poverty response of +3.9pp — roughly double the paper's central +1.87pp. A factor-of-two discrepancy on the headline poverty statistic between the paper's own two datasets is not a footnote-level issue. Either reconcile the gap (decompose dataset vs period vs imputation) or soften the abstract's constituency claims to ordinal statements. Naming specific constituencies to one decimal place from imputed occupations and a single seed-0 draw invites misuse.

**M7. Plain-FRS, uncalibrated weights and the Gini level.** Baseline Gini 0.309 vs official ~0.34 (no SPI adjustment); capital income heavily under-covered. Under-coverage of top incomes attenuates precisely the top-loaded mechanisms (fiscal cost of displacing top earners, capital-channel Gini, the recomposition exercise). A run on the enhanced/calibrated dataset for the central scenario would show whether the headline fiscal cost is robust to weighting; its absence is conspicuous given the machinery exists.

**M8. No behavioural response anywhere.** No labour supply, no take-up modelling (UC take-up among newly eligible middle-class households is empirically well below 100 per cent — this inflates both the fiscal cost and the measured cushioning), no wage adjustment, no GE effects. The cushioning "finding" is conditional on full take-up by displaced professionals with historically low take-up propensity. Run a partial take-up sensitivity; PolicyEngine supports this.

**M9. The 7 per cent / 2.6 per cent central calibration.** Compressing what Briggs-Kodnani frame as a decade-scale transition into a single simulation year quietly drives every £bn/yr figure, and the conversion convention is never actually shown. Spell out the arithmetic, and consider re-labelling the £bn figures as "annualised cost at peak displacement stock".

**M10. The tax-composition (phi) exercise borders on tautology.** Because the effective labour-tax rate on displaced earnings (25.0 per cent) approximately equals the CT main rate, "phi=1 is roughly revenue-neutral" is arithmetic, not analysis. The abstract sentence presents a free-parameter midpoint (phi=0.5) as if it were a result. State in the abstract that phi is unidentified.

## 4. Minor comments

1. Abstract far too long; the executive summary duplicates it and the introduction — for a journal, cut the executive summary.
2. Equation (1): state theta's range and the shift explicitly near the equation.
3. Table 1 note vs the low scenario's capital revenue: clarify sign conventions.
4. Fig. 4 colour legend unreadable at print size; residual bar invisible.
5. Gini units: pick one convention (x100 points vs level).
6. Table 2: uniform family 1.62m vs 1.61m elsewhere — explain or fix.
7. Doorley et al. (2026): give report number/URL.
8. Henseke/Williamson/Rockall: complete author lists ("et al." in a reference list is not acceptable).
9. Flag SSRN working-paper status of Hosseini-Lichtinger and Klein Teeselink on first use in the text.
10. Figure 8 right panel duplicates a table — consider dropping.
11. "£10.4bn per pp of poverty averted": clarify person-weighted AHC and average (not marginal) cost-effectiveness.
12. Quote/page the OBR (2026) claim; OBR (2025) and OBR (2026) easily conflated.
13. Hex maps: no labels; two panels use different colour conventions without note.
14. Gender subsection: cut or reference the "financial autonomy" speculation.
15. Check alphabetisation of "Klein Teeselink".

## 5. Questions for the authors

1. Have you obtained the Klein Teeselink full text, and do the press-release parameters survive? What does £32.5bn become under alternative wage-tier ratios?
2. What is the Monte Carlo SD of the cross-family differences? Does exposure vs junior (£18.4 vs £19.4bn) survive 20 draws?
3. Why not run the central scenario on the enhanced dataset with your own occupation imputation, so the +3.9pp/+1.87pp poverty discrepancy can be decomposed into dataset, period, and imputation components?
4. What exactly is the arithmetic converting Briggs-Kodnani task exposure into 7 per cent single-year displacement?
5. How would partial UC take-up (60-80 per cent among new entrants) change the cushioning result and the reform ranking?
6. Under a 6-month mean re-employment duration, does the ordering of families by fiscal cost change?
7. Does the R1 vs R2 ranking survive making wage insurance taxable (the realistic implementation)?
8. Elementary occupations receive exactly zero displacement by construction of the min-zero shift. Is a literal zero defensible, and how sensitive are bottom-decile results to small positive exposure for that group?

---

In sum: a well-executed, transparently documented scenario framework with a genuinely good organising idea, currently over-claiming on numbers its own disclosed limitations cannot support. The revision path is clear — verify or demote the measured family, add Monte Carlo and behavioural/duration/take-up sensitivities, reconcile the two-dataset poverty gap, and re-scope the abstract to the claims that survive.
