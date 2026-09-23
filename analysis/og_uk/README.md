# UK with and without AI: OG-UK runs

Dynamic general-equilibrium counterpart to the static PolicyEngine work in #12.
PolicyEngine answers *who bears an AI shock*; OG-UK answers *what it does to growth,
factor shares and the fiscal path over time*. The two are complementary and are
deliberately not merged — see #13.

## What is being run

One paired transition path, 1-sector, 2026-2030 (Anthropic's horizon), baseline against AI.

| parameter | baseline | with AI | shape | source of the assumption |
|---|---|---|---|---|
| `epsilon` | 1.0 | 1.0 | fixed | OG-UK's own single-sector default. At Cobb-Douglas `gamma` **is** the capital share, so the labour-share target is hit exactly rather than by calibration. |
| `gamma` | 0.35 | **0.389** | **step** | Anthropic Table 3: labour share falls 3.9pp (60c -> 56.1c). Applied to the UK's 0.65 gives 0.611, so `gamma = 1 - 0.611`. |
| `Z` | 1.000 | **-> 1.031** | **ramp** | Anthropic Table 3 panel d: measured TFP **+3.1%** above the no-AI path under *substantial*. |
| `g_y_annual` | 0.011 | 0.011 | — | **Mis-sourced — see `docs/OG_UK_OBR_COMPARISON.md`.** Should be 0.010 (OBR productivity growth, not potential output growth). All results below inherit this. |

Why `gamma` steps and `Z` ramps: this mirrors Anthropic's own design. Their `k` (automation
share) is **"held constant"** at 0.50/0.75/0.90, while `kappa`, `alpha` and `theta` **build**
from a mid-2026 anchor to their 2030 values. So automation intensity is a level and
productivity is a path.

## Results

`results/ai_scenario_1sector.json`. Labour shares land exactly on target: **0.6500 -> 0.6110**.

| | 2026 | 2030 |
|---|---|---|
| GDP | +12.19% | **+16.06%** |
| Consumption | +12.88% | +13.08% |
| Tax revenue | +10.95% | +14.99% |
| Debt | +12.19% | +11.85% |
| Government | +9.61% | +20.73% |

GDP growth **1.62% -> 2.55%**. Capital **+22.0%**, labour **-2.7%** — machines substituting
for workers. Resource-constraint error 6.7e-3 / 7.9e-3.

Because `epsilon = 1`, capital deepening cannot move factor shares — only `gamma` can. So the
labour-share result is **purely the automation channel**, with nothing confounding it.

### Against Anthropic's substantial scenario

| outcome, 2030 | Anthropic (US) | OG-UK | |
|---|---|---|---|
| labour-share fall | -3.9pp | -3.9pp | matched by construction |
| measured TFP | +3.1% | +3.1% | input, matched |
| **GDP vs no-AI** | **+8.3%** | **+16.1%** | we overshoot |
| **GDP growth** | **5.4%** | **2.55%** | we undershoot |
| capital stock | +13.8% | +22.0% | |
| unemployment | 4.6% | — | not modelled |

**We overshoot the level and undershoot the growth**, and both have the same cause: `gamma`
steps once and is fully in force from 2026, whereas their automation builds to 2030. The whole
level effect arrives immediately, leaving no growth to deliver afterwards. Fixing it needs
time-varying `gamma`, which OG-Core does not support (patch in progress; see below).

## How the three models map onto each other

| channel | Anthropic | Moll & Imas | OG-UK parameter |
|---|---|---|---|
| **Automation** (tasks labour -> capital) | `k` 0.50/0.75/0.90, held constant; affected mass `kappa` ramps 0.14 -> 0.53 | **the core** — `alpha` from 1/3 -> 1 by 2045 | **`gamma`** — step only, cannot ramp |
| **Augmentation** (productivity) | the `1-k` share | none — pure automation | **`Z`** — time-varying |
| **Adoption / diffusion** | `alpha` 0.10 -> 0.20/0.40/0.6 | — | **none** |
| **Task reinstatement** | `d` 0.50/0.25/0 | — | **none** |
| **Unemployment / search** | matching, `l` search discount, posting speed | — (Solow) | **none** |
| **Capital accumulation** | supply elasticity 3 | Solow, saving rate | **OLG endogenous saving** |
| **Factor shares** | endogenous, 60c -> 45.2c | -> 0 in the limit | via `gamma`; impossible at `epsilon = 1` without it |
| **Worker heterogeneity** | 2 groups, one wage each | representative agent | **`e[t,s,j]`**, 7 types x 80 ages |
| **Demand composition** | one good | their key critique, not in their model | **8 ONS sectors**, `p_m` prices |
| **Fiscal / tax** | none | none | **full PolicyEngine UK** |
| **Cohorts** | none | none | **OLG, S=80** |
| **Ideas / R&D** | ideas stock | `A_dot = A^phi K` | none |
| **Elasticity** | 0.5 assumed | argues **0.2** | 1.0 here; 0.4-1.3 per sector in the 8-sector build |

**We collapse their six AI channels into two.** `Z` absorbs capability, adoption and
productivity together — there are no tasks to count separately — and `gamma` is their
automation share. Reinstatement and search have no home at all.

That is why this model **cannot speak to displacement**. Anthropic's 4.6% unemployment has no
counterpart and never will without a labour-market block. The honest boundary: OG-UK owns
growth and factor shares, PolicyEngine owns who bears it.

## Known limitations

1. **The shock is a step, not a ramp** for `gamma`. Causes both discrepancies against Anthropic
   above. `p.gamma` is indexed `[m]` in ~14 sites in `ogcore/firm.py` with no time dimension.
   A backward-compatible patch is in progress (`Z`'s `[-1]` / `[: p.T]` convention); the steady
   state already accepts a `gamma` path, the transition still needs a broadcasting fix in
   `get_MPx`.
2. **`g_y_annual` is mis-sourced** (see above). Every re-trended figure inherits it.
3. **No common origin.** The `gamma` step puts a +12.2% GDP wedge in at 2026. Automation and a
   shared starting point cannot coexist until (1) is fixed.
4. **Consumption is front-loaded** — +12.9% already in 2026. Households have perfect foresight
   and consume against a permanently richer future from period one. This is real economics, not
   an artefact; a truly common origin would need an *unanticipated* shock, which TPI cannot do.
5. **Investment is unreliable.** It is the residual in the resource constraint and absorbs the
   numerical error.
6. **8-sector transition diverges past t~7.** Usable over 2026-2030 only — which covers
   Anthropic's horizon, but gives no long-run answer. Six hypotheses eliminated; see
   PolicyEngine/og-model-dashboard#2.

## Replicating every number

```bash
# 1. the paired baseline-vs-AI transition (~14 min, needs an OG-UK checkout)
cd /path/to/OG-UK
HDF5_USE_FILE_LOCKING=FALSE uv run python /path/to/uk-ai-study/analysis/og_uk/run_ai_scenario.py

# 2. the result tables — % gaps, growth, labour share, vs Anthropic (instant)
python analysis/og_uk/report_results.py

# 3. the OBR baseline comparison and component decomposition (instant)
python analysis/og_uk/compare_obr_baseline.py
```

Steps 2 and 3 read `results/ai_scenario_1sector.json`, which is committed, so they run
without re-solving. Both take `--oguk-dir` if OG-UK is not at `~/ogmodels/OG-UK`.

| script | reproduces |
|---|---|
| `run_ai_scenario.py` | the paired transition; writes `results/ai_scenario_1sector.json` |
| `report_results.py` | the % gap table, GDP growth, the labour-share path, and the Anthropic comparison |
| `compare_obr_baseline.py` | 1.67% vs OBR's 1.60%, and the productivity / labour-supply / potential-output decomposition |
| `run_gamma_ramp.py` | the time-varying-`gamma` test — **requires the `ogcore/firm.py` patch** |

Roughly 14 minutes for the pair. Two environment gotchas: the default Dask client fails
(`Nanny failed to start`) so a `LocalCluster(processes=False)` is passed explicitly, and
`_build_specs` makes a live network call for demographics on every run.

`run_gamma_ramp.py` is the time-varying-`gamma` test and requires the `firm.py` patch.
