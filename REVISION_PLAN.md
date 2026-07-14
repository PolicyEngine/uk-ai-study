# Revision plan (referee + code audit + literature scan, 14 July 2026)

Verdict: **Major revision, but the paper is worth saving.** The gap (UK AI tax-benefit
incidence) is real and still open, but PolicyEngine's own `ai-inequality` project lists
PolicyEngine-UK as its next extension — scoop risk within months. Move fast; prioritise
correctness fixes and Monte Carlo, then the framing upgrade.

## Phase 0 — Correctness (blockers, ~1 week)

1. **Grid vs preset inconsistency (referee M1).** 7%/0%-wage grid cell = £21.5bn, 3% wage
   cell ≈ £10.2bn, yet central preset (7%, 2.6%) = £18.4bn — not on the same surface.
   Likely cause: grid uses "% of exposed employment", presets use "% of employees".
   Reconcile the denominator everywhere (abstract, methodology.tex:23, results.tex:9/83/92).
2. **Wage-shock equation (M2).** Eq. (3) with employment-weighted θ̄ does not conserve the
   aggregate wage bill; code apparently renormalises. Write the actual formula in the paper.
   Also: conservation only tested at displacement=0 — survivor-composition drift (survivors'
   mean θ > θ̄) is untested; quantify it and add a test (code audit #6).
3. **Capital shock baseline (M3).** ×1.398 scaling comes from +0.4pp on an undocumented
   1.005% baseline return. Justify or re-parameterise as direct proportional scaling +
   sensitivity.
4. **`shocks.py:210` silent `employment_status` failure.** Make it a hard error or assert
   post-hoc that displaced persons are UNEMPLOYED; add a test.
5. Fix stale `shocks.py` docstring (uniform, not exposure-proportional, within-group draws).

## Phase 1 — Statistical validity (~1–2 weeks)

6. **Monte Carlo everywhere (M5).** 20–50 draws with mean ± SD for all five incidence
   families, the reform table, and decile/age breakdowns. Stop writing single-draw
   `decile_income_change` to preset JSONs (results/low.json flips sign across deciles from
   seed noise). Only narrate orderings that survive the SD.
7. **Klein Teeselink provenance (M4).** Get the SSRN full text or demote the "measured"
   family to appendix and strip £32.5bn from the abstract; add wage-tier-ratio sensitivity
   (3/0.8, 6/0.7, 9.6/0.6).
8. **Duration + take-up sensitivities (M6).** Scale earnings loss by expected out-of-work
   duration (e.g. 6 months) and apply 60–80% UC take-up; report whether the £14–32bn spread
   and the "means-testing dominates wage insurance" ranking survive.
9. **Poverty-line variants.** Report `in_relative_poverty_bhc` alongside the absolute line
   (JR16 comparability); state AHC line treatment; check relative-poverty mechanics.
10. **Reconcile +3.9pp vs +1.87pp poverty (M7).** Run the central scenario on the enhanced
    dataset with the paper's occupation imputation; decompose dataset/period/imputation.

## Phase 2 — Robustness & reproducibility (~1 week)

11. Multiplier sensitivities: compression 1.5/2/3; junior tilts; zero-displacement floor for
    elementary occupations (code audit #7).
12. Employment-weighted (not unweighted) mean imputation for no-SOC employees (runner.py:105).
13. Pin `frs_2024_25.h5` revision/sha256; ship a lockfile; align Python version (pyproject
    says ≥3.13, machine runs 3.10); vendor the populace PR #325 crosswalk derivation.
14. Integration test (synthetic dataset through `run_scenario`), `gini()` unit test,
    join-coverage test.
15. Fix relative paths in analysis scripts; note VAT/consumption not shocked in gov_balance.

## Phase 3 — Framing upgrade (the enhancement that raises the ceiling, ~2 weeks)

16. **Anchor scenarios in observed 2023–25 evidence**: junior/entry-level displacement
    (Brynjolfsson et al. "Canaries"; GOV.UK firm-level −4.5%, junior −5.8%), null aggregate
    wage effects as the low scenario. The by-age incidence becomes genuinely novel.
17. **Tie fiscal results to OBR's ±£90bn AI scenarios** (July 2026 FRS) — household-level
    incidence underneath OBR macro numbers is the clean hook nobody has done.
18. Add a 2026-vintage exposure robustness: Anthropic Economic Index observed exposure and/or
    the UK SOC-coded GenAI index (arXiv 2507.22748) alongside C-AIOE.
19. Optional differentiator: policy-response counterfactuals (uprating, UBI variant, capital
    tax / NI on capital income) — preempts PolicyEngine's framing.
20. Cite and differentiate from PolicyEngine `ai-inequality` (or reach out to collaborate).

## Phase 4 — Polish

21. Referee minors: five-family count in intro; 1.62m vs 1.61m; 13% preset outside the grid;
    Gini units; "et al." in reference list; SSRN WP flags; absorbed-share 30% vs "half"
    (results.tex:41 vs policy.tex:4); transition-share definitions (results.tex:9/78 vs
    Table A); £48.4k mean lost earnings sanity note; taxable wage-insurance variant.

## Target

Fiscal Studies (current, good fit); fallbacks: International Journal of Microsimulation,
Oxford Review of Economic Policy (if reframed around policy responses).
