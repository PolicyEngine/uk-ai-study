"""Unit tests for the state-enumeration engine (statesim/engine.py)."""
import itertools

import numpy as np
import pytest

from statesim import engine


def toy():
    # 3 households: 1, 2 and 3 eligible employees; one ineligible person
    hh_index = np.array([0, 1, 1, 2, 2, 2, 2])
    eligible = np.array([1, 1, 1, 1, 1, 1, 0], bool)
    p = np.array([0.1, 0.2, 0.3, 0.05, 0.5, 0.9, 0.0])
    return hh_index, eligible, p


def test_rank_and_masks():
    hh, el, _ = toy()
    rank = engine.within_household_rank(hh, el)
    assert rank.tolist() == [0, 0, 1, 0, 1, 2, -1]
    m = engine.state_mask(rank, 0b101, 3)
    assert m.tolist() == [True, True, False, True, False, True, False]


def test_state_probabilities_sum_to_one_and_match_brute_force():
    hh, el, p = toy()
    rank = engine.within_household_rank(hh, el)
    K, n_hh = 3, 3
    probs = engine.state_probabilities(p, hh, rank, K, n_hh)
    assert np.allclose(probs.sum(0), 1.0)
    # household 2: members with p = 0.05, 0.5, 0.9 -> P(b=0b011) = .05*.5*(1-.9)
    assert probs[0b011, 2] == pytest.approx(0.05 * 0.5 * 0.1)
    # household 0 has one member: states using bit 1 or 2 are impossible
    assert probs[0b010, 0] == 0 and probs[0b001, 0] == pytest.approx(0.1)


def test_expectation_equals_brute_force_enumeration():
    rng = np.random.default_rng(0)
    hh, el, p = toy()
    rank = engine.within_household_rank(hh, el)
    K, n_hh = 3, 3
    # an outcome that depends nonlinearly on who in the household is displaced
    y = rng.normal(size=(2 ** K, n_hh))
    probs = engine.state_probabilities(p, hh, rank, K, n_hh)
    got = engine.expectation(probs, y)
    for h in range(n_hh):
        members = np.flatnonzero((hh == h) & el)
        brute = 0.0
        for bits in itertools.product([0, 1], repeat=len(members)):
            pr = np.prod([p[i] if bt else 1 - p[i] for i, bt in zip(members, bits)])
            b = sum(bt << rank[i] for i, bt in zip(members, bits))
            brute += pr * y[b, h]
        assert got[h] == pytest.approx(brute)


def test_coverage_gap():
    hh, el, p = toy()
    rank = engine.within_household_rank(hh, el)
    w = np.ones(len(p))
    assert engine.coverage_gap(p, w, rank, 2) == pytest.approx(0.9)
    assert engine.coverage_gap(p, w, rank, 3) == 0.0


def test_spell_weights_and_lost_pay():
    assert sum(engine.SPELL_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-5)
    mean = sum(s * w for s, w in engine.SPELL_WEIGHTS.items())
    assert mean == pytest.approx(4.48, abs=0.01)
    assert engine.lost_pay_share({12: 1.0}, 0.1) == 1.0
    assert engine.lost_pay_share({0: 1.0}, 0.1) == pytest.approx(0.1)


def test_benefit_mix_endpoints():
    assert engine.benefit_mix(12, 10.0, 4.0, 1.0) == pytest.approx(0.5 * 10 + 0.5 * 4)
    assert engine.benefit_mix(0, 10.0, 4.0, 1.0) == pytest.approx(1.0)
    assert engine.benefit_mix(3, 12.0, 0.0, 0.0) == pytest.approx(3.0)


def test_takeup_mix_endpoints():
    assert engine.takeup_mix(0, 1.0, 3.0, 2.0, 9.0) == 3.0
    assert engine.takeup_mix(1, 1.0, 3.0, 2.0, 9.0) == pytest.approx(1.0 + 9.0 - 2.0)


def test_closed_account_arithmetic():
    acc = engine.closed_account(100.0, 40.0)
    assert acc["profit"] == 60.0
    assert acc["corporation_tax"] == pytest.approx(15.0)
    assert acc["uk_household_payout"] == pytest.approx(60 * 0.75 * 0.25)


def test_interpolate():
    assert engine.interpolate(1.0, 4.0, 1.0, 1.15, 1.05) == pytest.approx(1.0 + 3.0 / 3)


def test_gini_matches_runner_formula():
    from uk_ai_study.runner import gini as runner_gini
    rng = np.random.default_rng(1)
    v, w = rng.lognormal(size=500), rng.uniform(0.5, 2, size=500)
    v[:5] = -10.0
    assert engine.gini(v, w) == pytest.approx(runner_gini(v, w))
