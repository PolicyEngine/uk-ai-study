# Research agenda: scoring the Anthropic scenarios, and what comes after

Status: **proposal, not results.** Every number in the results table below is a
*pre-registered expectation*, derived by hand from published aggregates. Nothing
here has been run. The point of writing them down before running anything is that
the interesting outcome is whether they survive.

Motivating release: Korinek, Jones, Sacher, Cotter & McCrory (2026),
*Economic Scenarios for Transformative AI*, Anthropic Institute WP 2026-02
([PDF](https://www-cdn.anthropic.com/files/4zrzovbb/website/cf58f84d46a4a76bf5a5b039ac695fba6b80041c.pdf)),
with the scenario explorer at <https://www.anthropic.com/institute/econ-scenarios>.

## Where the gap is

Four recent treatments of AI and the public finances, and the same hole in each:

| Source | Covers | Missing |
|---|---|---|
| Korinek et al. (2026), Anthropic WP 2026-02 | macro paths to 2030, US | no tax-benefit system; reviewers note it "does not follow individual workers" |
| Dynan, Elmendorf & Sheiner, [NBER w35437](https://sites.harvard.edu/doug-elmendorf/files/2026/07/6.26.26-How-Might-Fiscal-Policy-Respond-to-the-Rise-of-Artificial-Intelligence.pdf) | US federal debt scenarios | aggregate; no household microsimulation |
| Iselin & Nunn (2026), [Budget Lab](https://budgetlab.yale.edu/research/how-potential-ai-futures-would-play-out-current-tax-system) | US federal income tax + payroll + corporate wedge | no transfers, no poverty, no states |
| Brookings, *Can AI restore fiscal sustainability?* | five offsetting fiscal forces | names displaced-worker support as a force, does not quantify it |

Every one of them identifies the transfer system as important and then does not
model it. No AI shock has been run through a full tax-benefit microsimulation in
any country. That is the opening this repo is already built for.

## Paper 1 — Scoring the three scenarios for the UK

Adopt the Korinek et al. calibration verbatim, so that what differs is model
coverage rather than assumptions (the approach PolicyEngine took to the Budget
Lab in `PolicyEngine/ai-inequality`). Run their modest / substantial / extreme
through the existing pipeline.

Their published 2030 targets:

| | knowledge unemp. | other unemp. | knowledge wages | other wages | labour share |
|---|---|---|---|---|---|
| Modest | 2.9% | 5.4% | +0.4% | +1.1% | 59.4c |
| Substantial | 4.5% | 4.6% | −0.3% | +5.9% | 56.1c |
| Extreme | 17.9% | 3.9% | −11.5% | +33.6% | 45.2c |

Three things have to be built or decided before this runs:

1. **Negative survivor wages.** `shocks.py` eq 3.5 gives every survivor a positive
   uplift proportional to theta. Anthropic's exposed workers take −11.5%. A
   negative-shock channel for exposed survivors is new structure, not a parameter
   change, and it is the channel that interacts hardest with the UC taper and
   income-tax progressivity.
2. **Stock vs flow.** Their 17.9% is a 2030 unemployment *stock after
   reallocation*; `displacement_rate` is a *flow*. Their re-employment parameter
   (`l = 0.17 x 3/months`, Appendix B) gives the search duration needed to do the
   translation defensibly. Plausible mappings span a wide range and the
   sensitivity is itself a result.
3. **Knowledge-work boundary.** Their "knowledge work = 62% of the wage bill" has
   to map onto SOC2020 major groups. Groups 1–3 is the natural candidate; the
   boundary choice needs a sensitivity.

### Pre-registered expected results

> **These are hand-derived hypotheses, not model output.** Nothing in this table
> has been through `apply_shocks` or PolicyEngine. They are arithmetic on
> published aggregates, written down before the run so that the comparison
> against real output means something. The derivation is shown below so it can
> be checked and disagreed with.

**Derivation.** From `results/central.json`: 7% displacement = 1.557m displaced,
so the employee base is ~22.2m. Knowledge work is ~45% of UK employment (~10.0m)
against Anthropic's 62% of the *wage bill* — knowledge workers are better paid,
so the employment share is lower than the wage-bill share. Excess knowledge
unemployment over their near-baseline modest case (2.9%) is +1.6pp under
substantial and +15.0pp under extreme, giving stocks of ~160k and ~1.50m, i.e.
0.7% and 6.7% of employees. Exchequer and poverty ranges are then scaled off the
central case (7% displacement, +2.6% uplift -> GBP 18.2bn, +1.81pp) with a
judgement adjustment for the wage-divergence channel, which the central case does
not contain. That adjustment is the weakest link and is why these are ranges.

| Scenario | Displacement equiv. | Exchequer (GBP bn, + = cost) | BHC poverty (pp) | Gini change | Labour-share shift |
|---|---|---|---|---|---|
| Modest | ~0 (below baseline churn) | −1 to −3 (net gain) | −0.05 to 0.00 | +0.001 | −0.6pp |
| Substantial | ~0.7% (160k) | −3 to −6 (net gain) | −0.10 to +0.10 | +0.003 | −3.9pp |
| Extreme | ~6.7% (1.50m) | **+30 to +55** | **−0.5 to +0.5** | **+0.020 to +0.035** | −14.8pp |
| *memo:* existing central (**actual model output**) | 7.0% (1.56m) | +18.2 | +1.81 | +0.010 | n/a |

Only the memo row is a real result. The three above it are predictions.

The two worth pre-registering, because they are the ones that could be wrong in
an interesting way:

- **Extreme has roughly the same headline job loss as the existing central case
  (~1.5m) but a much larger Exchequer cost.** If that holds, the entire
  difference is incidence and wage divergence, not the size of the job loss —
  which is the thesis of this repo, tested against an external calibration.
- **Extreme may leave UK poverty flat or falling while the Gini rises sharply.**
  The losers are middle-to-upper earners with weak means-tested entitlement; the
  +33.6% to non-exposed workers lifts low earners. A scenario that costs tens of
  billions and barely moves the poverty rate would mirror the sign flip Max found
  in the US income-shift work, and would be the paper's headline.

If poverty moves sharply *up* instead, the prediction is wrong and that is the
more publishable outcome — it would mean UK exposure reaches further down the
distribution than the US case implies.

### What the code already supports, and what it does not

Checked against `uk_ai_study/shocks.py` (738 lines):

- **Already there.** `WageMarginScenario` applies a C-AIOE-graded *cut* with the
  eq 3.5 uplift on top, so net change is `uplift_i - cut_i` and can already be
  negative for high-exposure, low-complementarity workers.
  `MixedMarginScenario` mixes the displacement and wage-cut margins at fixed
  gross loss. The negative-survivor-wage channel is therefore closer to existing
  than first assumed.
- **Not there.** Nothing targets *group-level* wage changes. Anthropic specifies
  two numbers (−11.5% knowledge, +33.6% other) and the existing scenarios take a
  single aggregate plus a gradient. A scenario type that solves for the gradient
  parameters hitting two group targets is the actual new code required.
- **Also not there.** A stock-to-flow translation, and the knowledge-work to
  SOC2020 boundary with its sensitivity.

### Where this stops

This is a static impact-year analysis, and the questions are path questions.
The dynamic track is OG-UK, with two blockers found in
`ogmodels/OG-UK/oguk/oguk_default_parameters.json`:

- `epsilon = 1.0` — Cobb-Douglas, so factor shares are **fixed by construction**.
  The 60c -> 45c labour-share move, the single most important outcome in every
  source above, is identically zero in the current calibration. Requires
  `epsilon > 1` (gross substitutes), and that value then drives everything.
- `alpha_T = 0.06` — transfers are a **fixed share of GDP**, so the welfare state
  grows with the economy and does not respond to need. That is the opposite of an
  automatic stabiliser, and it means the stabilisation question below cannot be
  asked in OG-UK at all.

OG-UK's comparative advantage is cohorts: none of the four sources above has an
age dimension, and "AI displaces mid-career knowledge workers, enriches capital
owners who are mostly old, and leaves the young with the productivity and the
debt" is a pure OLG question nobody has run.

## Paper 2 — Belief heterogeneity through the tax-benefit system

Appendix B of the Anthropic paper codes five parameters per survey respondent:
capability `kappa_2030`, adoption `alpha`, automation `k`, productivity gain
`theta`, re-employment discount `l`. Table B.1 prints every answer-to-value
mapping. They then state: *"We run the model on each respondent's five values and
report the quantiles of each outcome across respondents"* — using the **3,259
respondents who answered all five items** (not the headline 10,980).

So they have built 3,259 individual macro-outcome vectors and stopped one step
short of a household. The proposal is to push each one through PolicyEngine UK
and recover a distribution over Exchequer cost, poverty and inequality induced by
the distribution of *beliefs*, rather than by three round numbers.

Why this is worth doing:

1. **New object.** The subjective-expectations literature elicits beliefs and
   studies individual behaviour. Mapping an elicited belief distribution onto a
   fiscal aggregate through a structural microsimulation has not been done.
2. **Non-linearity is the finding.** Quantiles of a non-linear transform are not
   the transform of the quantiles. If the median belief implies a small cost but
   the mean over beliefs implies a much larger one, the gap is pure convexity in
   the tax-benefit system — UC taper kinks, the personal allowance, contributory
   exhaustion. Their reported quantiles cannot see this, which means their own
   headline may understate the expected cost, shown using their own data.
3. **It fixes the Monte Carlo.** `REVISION_PLAN.md` item 6 notes that
   `results/low.json` flips sign across deciles from seed noise — the draws
   currently sample a nuisance parameter. Re-anchoring draws to belief
   heterogeneity makes every draw a real respondent and the SDs interpretable.

**Dependency.** The respondent microdata is not released, and the joint
distribution of the five answers is exactly what matters and exactly what is
missing — correlation between high capability and pessimistic re-employment
cannot be recovered from marginals. Three routes, in order of preference:

- (a) request the microdata from the authors;
- (b) field a UK version (n ~ 2,000), which is better anyway since UK beliefs are
  the right input to a UK model;
- (c) simulate the joint under a copula matching the published marginals and
  report sensitivity to the dependence parameter — weakest, publishable as a bound.

Route (a) or (b). Paper 1 should be published first: it costs nothing, and it is
a better basis for the request.

## Sequencing

1. Close `REVISION_PLAN.md` Phase 0 (M1 denominator, M2 wage-bill conservation).
   Publishing against an external calibration on numbers with a known internal
   inconsistency is the fastest way to lose this audience.
2. Paper 1, static, as a write-up while the release is current.
3. Paper 2 data request in parallel.
4. OG-UK `epsilon` re-specification as the dynamic track.
