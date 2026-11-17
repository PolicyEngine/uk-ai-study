"""Pure functions for state enumeration, reweighting and the closed account.

Everything here is testable without Microcosm; the simulation runs live in
statesim/run.py. Validated in prototypes/ (docs/referee/phase1-results.md,
phase2-results.md): exact replication of the paper's draws, and the Phase 2
waterfall.
"""
from __future__ import annotations

import numpy as np

#: corporation tax, payout ratio and UK-household ownership share of the payout
CT_RATE, PAYOUT, UK_SHARE = 0.25, 0.5, 0.5
#: employer NI rate on pay above the secondary threshold (closed account)
EMPLOYER_NI_MARGINAL = 0.15
#: spell-length weights on run points 0/3/6/9/12 months: tent-function weights
#: of the inflow spell density fitted to ONS UNEM01 (2025 average), mean 4.48
#: months (docs/referee/phase2-calibration.md)
SPELL_WEIGHTS = {0: 0.254087, 3: 0.343849, 6: 0.181202, 9: 0.096705, 12: 0.124156}
#: re-employment pay penalty (Upward and Wright 2019)
REEMPLOYMENT_PENALTY = 0.10
#: UC take-up mixing weight for households with a displaced benefit unit:
#: (0.88 - 0.53) / (1 - 0.53), docs/referee/phase2-calibration.md
TAKEUP_Q = 0.75


def within_household_rank(hh_index: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Rank of each eligible person among their household's eligible people (-1 if not)."""
    rank = np.full(len(hh_index), -1)
    seen: dict = {}
    for i in np.flatnonzero(eligible):
        h = hh_index[i]
        rank[i] = seen.get(h, 0)
        seen[h] = rank[i] + 1
    return rank


def state_mask(rank: np.ndarray, b: int, K: int) -> np.ndarray:
    """Displaced mask of state b: rank r < K is displaced iff bit r of b is set."""
    ok = (rank >= 0) & (rank < K)
    return ok & (((b >> np.clip(rank, 0, None)) & 1) == 1)


def state_probabilities(p: np.ndarray, hh_index: np.ndarray, rank: np.ndarray, K: int, n_hh: int) -> np.ndarray:
    """P_h(b) for every state b and household h (independence within household).

    Returns an array of shape (2**K, n_hh); each column sums to one.
    """
    pr = np.zeros((n_hh, K))
    for r in range(K):
        sel = rank == r
        pr[hh_index[sel], r] = p[sel]
    probs = np.ones((2 ** K, n_hh))
    for b in range(2 ** K):
        for r in range(K):
            probs[b] *= pr[:, r] if (b >> r) & 1 else (1.0 - pr[:, r])
    return probs


def coverage_gap(p: np.ndarray, weight: np.ndarray, rank: np.ndarray, K: int) -> float:
    """Expected displaced weight held by people ranked K or beyond (not enumerated)."""
    return float((p * weight)[rank >= K].sum())


def expectation(probs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """E_h[Y] = sum_b P_h(b) Y_h(b); values has shape (2**K, n_hh)."""
    return (probs * values).sum(axis=0)


def benefit_mix(s: int, b_out_jsa, b_out, b_re):
    """Monthly UC assessment for a spell of s months (JSA for at most 6)."""
    j = min(s, 6)
    return (j / 12) * b_out_jsa + ((s - j) / 12) * b_out + ((12 - s) / 12) * b_re


def takeup_mix(q, base_flag, y_flag, base_claim, y_claim):
    """Level under take-up mixing, each mode against its paired baseline."""
    return base_flag + q * (y_claim - base_claim) + (1 - q) * (y_flag - base_flag)


def lost_pay_share(spells: dict = SPELL_WEIGHTS, penalty: float = REEMPLOYMENT_PENALTY) -> float:
    """Expected share of annual pay a displaced worker loses in year one."""
    return float(sum(w * (s / 12 + (12 - s) / 12 * penalty) for s, w in spells.items()))


def closed_account(saved_cost: float, survivor_cost: float) -> dict:
    """Profit, corporation tax and UK-household payout of the closed account (£)."""
    profit = saved_cost - survivor_cost
    return {"profit": profit, "corporation_tax": CT_RATE * profit,
            "uk_household_payout": (1 - CT_RATE) * PAYOUT * UK_SHARE * profit}


def interpolate(y_a, y_b, f_a: float, f_b: float, f: float):
    """Linear interpolation of household outcomes in the dividend factor."""
    return y_a + (f - f_a) / (f_b - f_a) * (y_b - y_a)


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Gini, identical to uk_ai_study.runner.gini (negatives bottom-coded)."""
    values = np.clip(np.asarray(values, float), 0.0, None)
    order = np.argsort(values)
    v, w = values[order], np.asarray(weights, float)[order]
    cw = np.cumsum(w)
    cv = np.cumsum(v * w)
    if cv[-1] == 0:
        return 0.0
    return float(1 - 2 * np.sum((cv - v * w / 2) * w) / (cv[-1] * cw[-1]))
