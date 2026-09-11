"""Score the Korinek et al. (2026) scenarios through the UK tax-benefit system.

Korinek, Jones, Sacher, Cotter & McCrory (2026), *Economic Scenarios for
Transformative AI*, Anthropic Institute Working Paper 2026-02.

Their model is a two-occupation task-based macro model with a frictional
labour market, solved to 2030 for the US. It has no tax-benefit system and,
as their own reviewers note, "does not follow individual workers". This
module adopts their published 2030 targets verbatim and pushes them through
PolicyEngine UK, so that what differs from their results is model coverage
rather than assumptions.

Three translation decisions, each a sensitivity rather than a fact:

1. KNOWLEDGE WORK. They define knowledge work as 62% of the US wage bill and
   split the economy in two. We map it to SOC2020 major groups 1-3 (managers,
   professional, associate professional) — the three most AI-exposed groups on
   C-AIOE apart from group 4 (administrative), which is the boundary case. See
   ``KNOWLEDGE_GROUPS`` and the ``knowledge_groups`` scenario field.

2. STOCK TO FLOW. Their unemployment numbers are 2030 STOCKS after
   reallocation; ``displacement_rate`` is a FLOW of workers out of jobs in the
   impact year. A stock u sustained with expected duration d years implies an
   annual separation flow of roughly u/d (steady-state Little's law). Their
   re-employment parameter (Appendix B, ``l = 0.17 x 3/months``) gives the
   duration. ``flow_from_stock`` implements this and it is the single largest
   source of uncertainty in the exercise.

3. WAGE DIVERGENCE. They report two group-level wage changes (knowledge and
   other). The existing ``ShockScenario`` takes one aggregate uplift and a
   theta gradient, which cannot hit two targets. ``apply_anthropic_shock``
   applies the two group rates directly to survivors.

The capital shock is NOT taken from JR16's +0.4pp here: their scenarios state
a labour share, and we scale capital income to deliver the implied rise in the
capital share of factor income.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from uk_ai_study.shocks import prescribed_systematic_sample

#: SOC2020 major groups treated as "knowledge work". Groups 1-3 are managers,
#: professional and associate professional occupations. Group 4
#: (administrative and secretarial) has the HIGHEST C-AIOE of any group
#: (0.744) but is not usually called knowledge work; it is the boundary case
#: and ``knowledge_groups`` exists so the choice can be varied.
KNOWLEDGE_GROUPS: tuple[int, ...] = (1, 2, 3)

#: Group-specific counterfactual unemployment rates. Korinek et al.'s MODEST
#: scenario is their near-business-as-usual case — they describe its impact as
#: "roughly the same kind of impact as the internet did", "hard to see in
#: macroeconomic data". Its 2030 rates are therefore the right no-AI
#: counterfactual against which to measure excess unemployment, and using them
#: makes the modest scenario a zero-displacement case BY CONSTRUCTION, which
#: is the honest reading rather than a defect.
#:
#: The alternative — a single economy-wide baseline (the UK rate is ~4.4%) —
#: is wrong because their two groups start from different rates, and it
#: produces the perverse result that the modest scenario displaces MORE
#: non-knowledge workers than the substantial one. ``BASELINE_UNEMPLOYMENT``
#: is retained for that sensitivity.
BASELINE_KNOWLEDGE_UNEMPLOYMENT = 0.029
BASELINE_OTHER_UNEMPLOYMENT = 0.054
BASELINE_UNEMPLOYMENT = 0.044


def flow_from_stock(
    stock: float,
    baseline_stock: float,
    duration_years: float,
) -> float:
    """Annual separation flow implied by an excess unemployment stock.

    In a steady state with separation flow ``f`` and expected unemployment
    duration ``d``, the unemployment stock is ``u = f * d`` (Little's law).
    Excess stock over baseline therefore implies ``f = (u - u0) / d``.

    Parameters
    ----------
    stock, baseline_stock:
        2030 and baseline unemployment RATES for the group.
    duration_years:
        Expected time to re-employment, in years.

    Returns the annual flow as a share of the group's employment. The result
    is capped at 1.0; a stock large enough to imply a flow above 1 means the
    steady-state reading has broken down, which is itself worth reporting.
    """
    if duration_years <= 0:
        raise ValueError("duration_years must be positive")
    excess = max(0.0, float(stock) - float(baseline_stock))
    return min(1.0, excess / float(duration_years))


@dataclass(frozen=True)
class AnthropicScenario:
    """One Korinek et al. (2026) scenario, as UK-applicable shock parameters.

    All rates are 2030 targets read off their published scenario table.

    knowledge_unemployment, other_unemployment:
        2030 unemployment STOCKS by group (not flows).
    knowledge_wage_change, other_wage_change:
        2030 wage changes relative to the no-AI path, applied to SURVIVORS.
    labour_share:
        2030 labour share of factor income. The baseline is 0.60.
    duration_years:
        Expected unemployment duration used for the stock-to-flow conversion.
    """

    name: str
    knowledge_unemployment: float
    other_unemployment: float
    knowledge_wage_change: float
    other_wage_change: float
    labour_share: float
    duration_years: float = 0.5
    knowledge_groups: tuple[int, ...] = KNOWLEDGE_GROUPS
    baseline_labour_share: float = 0.60
    baseline_knowledge_unemployment: float = BASELINE_KNOWLEDGE_UNEMPLOYMENT
    baseline_other_unemployment: float = BASELINE_OTHER_UNEMPLOYMENT

    @property
    def knowledge_displacement_flow(self) -> float:
        return flow_from_stock(
            self.knowledge_unemployment,
            self.baseline_knowledge_unemployment,
            self.duration_years,
        )

    @property
    def other_displacement_flow(self) -> float:
        return flow_from_stock(
            self.other_unemployment,
            self.baseline_other_unemployment,
            self.duration_years,
        )

    @property
    def capital_income_factor(self) -> float:
        """Scaling on capital income implied by the labour-share move.

        A fall in the labour share from ``s0`` to ``s1`` raises the capital
        share from ``1 - s0`` to ``1 - s1``. Holding total factor income
        fixed, capital income scales by ``(1 - s1) / (1 - s0)``.
        """
        return (1.0 - self.labour_share) / (1.0 - self.baseline_labour_share)


#: The three published scenarios. Unemployment, wage and labour-share figures
#: are read directly from the scenario explorer's 2030 outcomes.
ANTHROPIC_PRESETS = {
    "anthropic_modest": AnthropicScenario(
        "anthropic_modest",
        knowledge_unemployment=0.029,
        other_unemployment=0.054,
        knowledge_wage_change=0.004,
        other_wage_change=0.011,
        labour_share=0.594,
    ),
    "anthropic_substantial": AnthropicScenario(
        "anthropic_substantial",
        knowledge_unemployment=0.045,
        other_unemployment=0.046,
        knowledge_wage_change=-0.003,
        other_wage_change=0.059,
        labour_share=0.561,
    ),
    "anthropic_extreme": AnthropicScenario(
        "anthropic_extreme",
        knowledge_unemployment=0.179,
        other_unemployment=0.039,
        knowledge_wage_change=-0.115,
        other_wage_change=0.336,
        labour_share=0.452,
    ),
}


def knowledge_mask(persons: pd.DataFrame, groups: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of persons in knowledge-work SOC2020 major groups.

    Persons without an observed SOC code are NOT counted as knowledge
    workers; they fall into the "other" group, matching the treatment of
    unmatched records elsewhere in the pipeline (they receive mean exposure
    but never a group-specific shock).

    Accepts both the raw FRS coding (1000-9000) and plain major groups (1-9),
    matching ``exposure.exposure_for_major_group``. The FRS ``adult.tab``
    column stores 1000-9000.
    """
    codes = pd.to_numeric(persons["soc_major_group"], errors="coerce").to_numpy(
        dtype=float
    )
    codes = np.where(codes >= 10, codes / 1000.0, codes)
    return np.isin(codes, np.asarray(groups, dtype=float))


def draw_group_displaced(
    persons: pd.DataFrame,
    scenario: AnthropicScenario,
    seed: int = 0,
) -> np.ndarray:
    """Displace a quota of workers within each of the two occupation groups.

    Unlike ``shocks.draw_displaced``, which allocates one economy-wide quota
    across groups in proportion to exposure, this draws a SEPARATE quota per
    group, because Korinek et al. specify a separate unemployment target for
    each. Within a group, inclusion probabilities are uniform: their model has
    no within-group exposure gradient, so imposing one would add structure
    they do not have.
    """
    earnings = persons["employment_income"].to_numpy(dtype=float)
    weight = persons["weight"].to_numpy(dtype=float)
    employed = earnings > 0
    knowledge = knowledge_mask(persons, scenario.knowledge_groups)

    displaced = np.zeros(len(persons), dtype=bool)
    rng = np.random.default_rng(seed)
    for label, mask, rate in (
        ("knowledge", employed & knowledge, scenario.knowledge_displacement_flow),
        ("other", employed & ~knowledge, scenario.other_displacement_flow),
    ):
        idx = np.flatnonzero(mask)
        if idx.size == 0 or rate <= 0:
            continue
        quota = rate * weight[idx].sum()
        tilts = np.ones(idx.size, dtype=float)
        chosen = prescribed_systematic_sample(weight[idx], quota, tilts, rng=rng)
        displaced[idx[chosen]] = True
    return displaced


def apply_anthropic_shock(
    persons: pd.DataFrame,
    scenario: AnthropicScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Shocked person table under one Korinek et al. scenario.

    Displacement is drawn per group; survivors receive their GROUP's wage
    change (not a theta gradient, since the source specifies two group-level
    rates); capital income is scaled by the labour-share-implied factor.
    """
    shocked = persons.copy()
    displaced = draw_group_displaced(persons, scenario, seed=seed)
    shocked["displaced"] = displaced

    earnings = shocked["employment_income"].to_numpy(dtype=float)
    knowledge = knowledge_mask(persons, scenario.knowledge_groups)
    survivors = (earnings > 0) & ~displaced

    rate = np.where(knowledge, scenario.knowledge_wage_change, scenario.other_wage_change)
    new_earnings = np.where(displaced, 0.0, earnings * (1.0 + np.where(survivors, rate, 0.0)))
    shocked["employment_income"] = new_earnings

    factor = scenario.capital_income_factor
    for column in ("savings_interest_income", "dividend_income"):
        shocked[column] = shocked[column].to_numpy(dtype=float) * factor
    return shocked
