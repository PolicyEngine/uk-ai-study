# OG-UK against OBR: what is numerically comparable, and what does not match

Answers a question from Max: *"on the OLG model — what OBR result is this intending
to replicate?"*

Short answer: **nothing, currently.** The `og-model-dashboard` "OBR comparison" tab is a
**structural** comparison against OBR Working Paper No. 22, *A new UK overlapping generations
model* (April 2025) — Production, Households, Bequests, Tax system, Government, Open economy,
Solution and calibration. It compares model architecture, not numbers. OG-UK takes calibration
inputs from the EFO but does not target any published OBR result.

This note sets out what a numerical comparison would look like, reports the one that can be
made today, and records two findings that came out of making it.

## What can be compared today

Run: OG-UK 1-sector baseline, `start_year=2026`, no reform, `epsilon=1.0`, `gamma=0.35`.
Real GDP growth re-trended at `g_y + g_n`.

| year | OG-UK | OBR (March 2026 EFO) |
|---|---|---|
| 2027 | 1.82% | |
| 2028 | 1.60% | |
| 2029 | 1.65% | |
| 2030 | 1.62% | |
| **2027-30 mean** | **1.67%** | **1.60%** |

Source for the OBR figure: March 2026 EFO para 1.10 — *"We expect GDP growth to pick up to
average 1.6 per cent a year from 2027 to 2030."*

**The headline matches to +0.07pp.** That is a genuine validation of the baseline and worth
stating.

## But the components do not match

| | OBR (March 2026 EFO) | OG-UK | gap |
|---|---|---|---|
| Productivity growth, medium term | **1.0%** (para 1.2) | 1.1% (`g_y_annual`) | +0.1pp |
| Labour supply growth by 2030 | **0.5%** (para 1.2) | 0.63% (`g_n`) | +0.13pp |
| **Potential output growth, 2030** | **1.5%** (paras 1.10, 2.10) | **1.73%** | **+0.23pp** |
| Potential output growth, 2026 | 1.2% (para 2.10) | 1.85% | +0.65pp |

OG-UK runs hot on **both** components. Realised GDP growth only lines up because the
transition dynamics pull it back down. That is a coincidence, not an agreement, and it should
not be presented as a validation without this table beside it.

## Finding 1: `g_y_annual` is mis-sourced

`oguk/api.py` documents the parameter as:

> `g_y_annual = 0.011` — OBR "Economic and Fiscal Outlook" (Nov 2025), Table 1.1:
> **potential output growth** ~1.1% per year.

But OG-Core defines `g_y_annual` as *"Annual growth rate of **labor augmenting technological
change**"* — productivity growth, not potential output growth. OBR separates the two
explicitly (March 2026 EFO, para 1.2):

> "we assume that **productivity growth** will pick up to **1 per cent** in the medium term,
> while **labour supply growth** declines ... to **½ a per cent** by 2030"

with potential output growth of 1.5% being the sum. Since OG-UK already carries labour-force
growth separately in `g_n`, setting `g_y` to potential output growth **double-counts labour
supply**.

**Fix:** `g_y_annual = 0.010`. That brings implied potential output growth from 1.73% to
1.63%, close to OBR's 1.5%.

Every re-trended figure produced so far inherits this, including the AI scenario results in
#12 and the OG-UK dashboards.

## Finding 2: the OBR has published its own AI scenarios

The March 2026 EFO contains four scenarios, two of which are structural. The one directly
relevant here:

> "**technological displacement** substitute for labour, increasing capital deepening and
> raising productivity for workers who remain employed. We assume that this higher trend
> productivity fully offsets lower employment to leave the level of GDP unchanged but that
> this increase in average labour productivity is not reflected in higher real earnings. This
> implies a **lower labour share and a higher corporate profit share** relative to our central
> forecast. As labour income faces a higher effective tax rate, this **reduces the
> tax-richness of economic activity**."

with equilibrium unemployment rising to **5.5%** in both structural scenarios, and method in
OBR *Briefing Paper No. 9: Forecasting productivity* (Nov 2025), Annex B.

This is the official UK benchmark for exactly the question this project is asking — same
mechanism (capital deepening, falling labour share) and the same fiscal consequence. It is a
better anchor for UK AI work than Anthropic's US scenarios, and unlike Anthropic it is
directly replicable:

- GDP level **unchanged** versus central forecast
- labour share **down**, profit share **up**
- structural unemployment **5.5%**
- revenue **falls** despite unchanged GDP, because labour income is taxed more heavily

The last point is the interesting one and is squarely a PolicyEngine question: OBR asserts
reduced "tax-richness" but does not decompose it across households.

## What a full numerical comparison would need

1. **Baseline validation against the EFO path**, not one growth rate. OBR publishes borrowing
   (5.2% of GDP in 2024-25 falling to 4.3%), debt and receipts year by year; OG-UK produces all
   three. Data: <https://obr.uk/data/> (historical forecast database) and Annex A of the EFO.
2. **Replicate the technological-displacement scenario** and check whether OG-UK reproduces
   "GDP unchanged, labour share down, revenue down".
3. **OBR WP No. 22 simulations**, if that paper publishes reform results that can be re-run.

(1) should come first: it is cheap, and it is what makes any scenario result credible.

## Sources

- OBR, *Economic and fiscal outlook, March 2026* —
  [PDF](https://assets.publishing.service.gov.uk/media/69a6d7b62e1f4fbda4252208/economic-and-fiscal-outlook-march-2026-web-accessible.pdf)
  (paras 1.2, 1.10, 2.10, and the scenarios box)
- OBR, *Briefing Paper No. 9: Forecasting productivity* (Nov 2025), Annex B — AI productivity scenarios
- OBR, *Working Paper No. 22: A new UK overlapping generations model* (April 2025)
- OBR data downloads — <https://obr.uk/data/>. Note `obr.uk` rejects automated requests;
  the CRAN package [`obr`](https://rdrr.io/cran/obr/man/get_efo_economy.html) wraps the data
  API if a scripted pull is wanted.
