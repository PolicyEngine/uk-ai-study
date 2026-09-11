"""Tests for the Korinek et al. (2026) scenario translation."""

import numpy as np
import pandas as pd
import pytest

from uk_ai_study.anthropic import (
    ANTHROPIC_PRESETS,
    AnthropicScenario,
    apply_anthropic_shock,
    draw_group_displaced,
    flow_from_stock,
    knowledge_mask,
)


def _persons(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    # half knowledge (SOC 1000-3000), half other (SOC 5000-9000), FRS coding
    soc = np.where(
        rng.random(n) < 0.5,
        rng.choice([1000.0, 2000.0, 3000.0], n),
        rng.choice([5000.0, 6000.0, 8000.0, 9000.0], n),
    )
    return pd.DataFrame(
        {
            "employment_income": rng.uniform(5_000, 80_000, n),
            "savings_interest_income": rng.uniform(0, 2_000, n),
            "dividend_income": rng.uniform(0, 3_000, n),
            "soc_major_group": soc,
            "weight": rng.uniform(50, 500, n),
            "age": rng.integers(20, 64, n),
            "exposure": rng.uniform(0, 1, n),
            "complementarity": rng.uniform(0.4, 0.7, n),
        }
    )


def test_flow_from_stock_littles_law():
    # a 13.5pp excess stock held with 6-month duration needs a 27% annual flow
    assert flow_from_stock(0.179, 0.044, 0.5) == pytest.approx(0.27)
    # halving the flow doubles the duration needed for the same stock
    assert flow_from_stock(0.179, 0.044, 1.0) == pytest.approx(0.135)
    # a stock at or below baseline implies no excess separations
    assert flow_from_stock(0.029, 0.044, 0.5) == 0.0
    # capped at 1.0 rather than returning an impossible flow
    assert flow_from_stock(0.9, 0.0, 0.1) == 1.0
    with pytest.raises(ValueError):
        flow_from_stock(0.1, 0.0, 0.0)


def test_knowledge_mask_accepts_both_socs():
    """The FRS stores 1000-9000; the exposure CSV is keyed 1-9."""
    frs = pd.DataFrame({"soc_major_group": [1000.0, 3000.0, 5000.0, np.nan]})
    plain = pd.DataFrame({"soc_major_group": [1.0, 3.0, 5.0, np.nan]})
    expected = [True, True, False, False]
    assert list(knowledge_mask(frs, (1, 2, 3))) == expected
    assert list(knowledge_mask(plain, (1, 2, 3))) == expected


def test_modest_is_the_counterfactual():
    """Modest is their near-BAU case, so it displaces nobody by construction."""
    s = ANTHROPIC_PRESETS["anthropic_modest"]
    assert s.knowledge_displacement_flow == 0.0
    assert s.other_displacement_flow == 0.0


def test_capital_factor_from_labour_share():
    s = ANTHROPIC_PRESETS["anthropic_extreme"]
    # labour 60% -> 45.2% means capital 40% -> 54.8%
    assert s.capital_income_factor == pytest.approx(0.548 / 0.40)


def test_displacement_quota_is_group_specific():
    """Each group's displaced weight must hit its own target, not a pooled one."""
    persons = _persons()
    s = ANTHROPIC_PRESETS["anthropic_extreme"]
    displaced = draw_group_displaced(persons, s, seed=0)
    w = persons["weight"].to_numpy()
    km = knowledge_mask(persons, s.knowledge_groups)
    emp = persons["employment_income"].to_numpy() > 0
    realised = w[displaced & km].sum() / w[emp & km].sum()
    assert realised == pytest.approx(s.knowledge_displacement_flow, rel=0.02)
    # extreme puts non-knowledge unemployment BELOW its counterfactual, so no
    # non-knowledge worker is displaced
    assert not displaced[~km].any()


def test_survivors_get_their_own_group_wage_rate():
    persons = _persons()
    s = ANTHROPIC_PRESETS["anthropic_extreme"]
    shocked = apply_anthropic_shock(persons, s, seed=0)
    km = knowledge_mask(persons, s.knowledge_groups)
    d = shocked["displaced"].to_numpy()
    ratio = (
        shocked["employment_income"].to_numpy() / persons["employment_income"].to_numpy()
    )
    assert np.allclose(ratio[km & ~d], 1 + s.knowledge_wage_change)
    assert np.allclose(ratio[~km & ~d], 1 + s.other_wage_change)
    assert np.all(shocked["employment_income"].to_numpy()[d] == 0.0)


def test_capital_income_scaled_for_everyone():
    persons = _persons()
    s = ANTHROPIC_PRESETS["anthropic_extreme"]
    shocked = apply_anthropic_shock(persons, s, seed=0)
    for col in ("savings_interest_income", "dividend_income"):
        assert np.allclose(
            shocked[col].to_numpy(),
            persons[col].to_numpy() * s.capital_income_factor,
        )


def test_longer_duration_means_smaller_flow():
    """The stock-to-flow conversion is the headline sensitivity."""
    base = ANTHROPIC_PRESETS["anthropic_extreme"]
    slow = AnthropicScenario(**{**base.__dict__, "duration_years": 2.0})
    assert slow.knowledge_displacement_flow < base.knowledge_displacement_flow
