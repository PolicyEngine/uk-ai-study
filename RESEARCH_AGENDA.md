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

### Results

Run on the pinned June 2026 FRS build (`frs_2024_25.h5`, sha256 `623802aa...`),
period 2026, seed 0, via `analysis/anthropic_scenarios.py`. The `central`
preset reproduces `results/central.json` bit-exactly on this build, so the
environment is validated.

| Scenario | Displaced | Exchequer (GBP bn, + = cost) | BHC poverty (pp) | Gini change |
|---|---|---|---|---|
| Modest | 0 | -3.3 | -0.08 | +0.0001 |
| Substantial | 0.39m | +1.8 | +0.23 | +0.0012 |
| Extreme | 3.57m | **+67.3** | **+4.18** | +0.0129 |
| *memo:* JR16 central | 1.56m | +18.2 | +1.81 | +0.0104 |

**The pre-registered prediction was wrong, in the way that was flagged as the
more publishable outcome.** The prediction was that extreme would leave UK
poverty "flat or falling" while the Gini rose sharply, on the reasoning that
the losers are middle-to-upper earners with weak means-tested entitlement.
Poverty instead rises +4.18pp — more than twice the JR16 central case — while
the Gini rises LESS than predicted (+0.0129 against a predicted +0.020 to
+0.035). UK exposure reaches further down the distribution than the US framing
implies.

The decomposition (`analysis/anthropic_decomposition.py`) shows why, and the
channels are close to additive (they sum to +4.25pp against +4.18pp for the
full scenario):

| Channel | Exchequer (GBP bn) | BHC poverty (pp) | Gini change |
|---|---|---|---|
| Displacement only | +92.1 | +5.49 | +0.0170 |
| Wage divergence only | -9.7 | -1.14 | -0.0095 |
| Capital shock only | -1.7 | -0.10 | +0.0005 |
| **Full** | **+67.3** | **+4.18** | **+0.0129** |

The reasoning behind the prediction was right, and was simply outweighed. Wage
divergence on its own IS strongly poverty-reducing (-1.14pp) and
inequality-reducing (-0.0095), because the +33.6% accruing to non-knowledge
workers reaches lower-paid households while the -11.5% falls on households
with little means-tested entitlement. That effect is real but is swamped by
displacement, which alone would raise poverty +5.49pp. The capital shock is
close to irrelevant to poverty (-0.10pp) because capital income is thin in the
FRS at the bottom of the distribution.

**The Exchequer number is dominated by the same channel.** Displacement alone
costs GBP 92bn; wage divergence returns GBP 10bn of that, because the
non-knowledge wage gain is taxed. The net GBP 67bn is roughly 3.7x the JR16
central case for roughly 2.3x the job loss.

#### The stock-to-flow sensitivity dominates everything

Their 17.9% is a 2030 unemployment STOCK; the model needs an annual FLOW.
Little's law (`u = f x d`) converts one to the other, and the assumed duration
drives the result more than the choice of scenario does:

| Expected duration | Implied knowledge flow | Exchequer (GBP bn) | BHC poverty (pp) |
|---|---|---|---|
| 6 months (survey median) | 30.0% | +67.3 | +4.18 |
| 1 year | 15.0% | +28.1 | +1.54 |
| 2 years | 7.5% | +6.2 | -0.06 |

At two years the extreme scenario's poverty effect turns slightly NEGATIVE
(-0.06pp) and the Exchequer cost falls by an order of magnitude. **Any scoring
of these scenarios is really a statement about assumed unemployment duration**,
and a paper that does not say so is not reporting a result. This is the single
most important methodological finding of the exercise.

#### Translation decisions, each a live sensitivity

1. **Knowledge work = SOC2020 major groups 1-3.** That is 52.3% of UK
   employees, against Korinek et al.'s 62% of the WAGE BILL — consistent,
   since knowledge workers are better paid. Group 4 (administrative and
   secretarial) is the boundary case: it has the HIGHEST C-AIOE of any group
   (0.744) but is not conventionally called knowledge work. `knowledge_groups`
   is a scenario field so this can be varied; it has not been yet.
2. **Modest is the counterfactual, not a scenario.** Korinek et al. describe it
   as near-business-as-usual, so its 2030 rates are the no-AI baseline and it
   displaces nobody by construction. Its nonzero rows above are the wage and
   capital channels alone. Using a single economy-wide baseline instead
   produces the perverse result that modest displaces MORE non-knowledge
   workers than substantial.
3. **Capital from the labour share.** Rather than JR16's +0.4pp, capital income
   is scaled by the labour-share move their scenarios state (60c -> 45.2c in
   extreme, a factor of 1.37).

#### What has not been done

- No Monte Carlo: every figure above is a single seed-0 draw. Given
  `REVISION_PLAN.md` item 6 (seed noise flips decile signs), the poverty and
  Exchequer headlines need 20-50 paired draws before they go in a paper.
- No knowledge-boundary sensitivity (groups 1-4 vs 1-3).
- No decile or age breakdown of the Anthropic scenarios.
- Their US calibration is applied to UK microdata unchanged; whether US
  occupational exposure transfers to the UK is assumed, not tested.

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
