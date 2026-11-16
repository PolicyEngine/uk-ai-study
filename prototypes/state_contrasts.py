"""Prototype: rule contrasts and household-bootstrap SEs from saved states.

Reads the per-household state outcomes written by state_enumeration.py and
computes, for exposure-proportional and uniform incidence at the central rate,
the exact expected cost, BHC/AHC poverty change and Gini change, plus a
household bootstrap (resampling households with replacement; Microcosm carries
no strata or PSU identifiers, so this is a simple-random-sample approximation
to survey sampling error). No new simulations: each rule is a probability
vector over the same states.

Usage: python prototypes/state_contrasts.py STATES_DIR [B]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))

from uk_ai_study import shocks  # noqa: E402
from uk_ai_study.data import PERIOD, microsimulation  # noqa: E402
from uk_ai_study.runner import build_person_table, gini  # noqa: E402
from state_enumeration import person_probabilities  # noqa: E402


def uniform_probabilities(persons, rate):
    employed, weight, _, _ = shocks._employee_groups(persons)
    p = np.zeros(len(persons))
    idx = np.flatnonzero(employed)
    p[idx] = shocks.prescribed_inclusion_probabilities(
        weight[idx], rate * float(weight[idx].sum()), np.ones(len(idx)))
    return p


def state_probs(p, hh_index, rank, K, n_hh):
    p_rank = np.zeros((n_hh, K))
    for r in range(K):
        sel = rank == r
        p_rank[hh_index[sel], r] = p[sel]
    probs = np.ones((2 ** K, n_hh))
    for b in range(2 ** K):
        for r in range(K):
            probs[b] *= p_rank[:, r] if (b >> r) & 1 else (1 - p_rank[:, r])
    return probs


def metrics(mult, base, states, probs):
    """Exact expectations with household multiplicities ``mult`` (bootstrap)."""
    w = base["hh_w"] * mult
    n = base["hh_n"]
    S = len(states)
    e_gov = sum(probs[b] * states[b]["gov"] for b in range(S))
    e_pb = sum(probs[b] * states[b]["pov_bhc"] for b in range(S))
    e_pa = sum(probs[b] * states[b]["pov_ahc"] for b in range(S))
    pw = (w * n).sum()
    keep = w > 0
    vals = np.concatenate([states[b]["equiv"][keep] for b in range(S)])
    wts = np.concatenate([(w * n * probs[b])[keep] for b in range(S)])
    nz = wts > 0
    vals, wts = vals[nz], wts[nz]
    return np.array([
        (w * (base["gov"] - e_gov)).sum() / 1e9,
        100 * ((w * e_pb).sum() - (w * base["pov_bhc"]).sum()) / pw,
        100 * ((w * e_pa).sum() - (w * base["pov_ahc"]).sum()) / pw,
        100 * (gini(vals, wts) - gini(base["equiv"][keep], (w * n)[keep])),
    ])


def main(states_dir: str, B: int = 200) -> None:
    d = Path(states_dir)
    K = json.loads((d / "meta.json").read_text())["K"]
    base = dict(np.load(d / "baseline.npz"))
    pp = dict(np.load(d / "persons.npz"))
    states = [dict(np.load(d / f"state_{b:02d}.npz")) for b in range(2 ** K)]
    n_hh = len(base["hh_w"])
    persons = build_person_table(microsimulation(), PERIOD)
    scen = shocks.PRESETS["central"]
    rules = {
        "exposure": person_probabilities(persons, scen),
        "uniform": uniform_probabilities(persons, scen.displacement_rate),
    }
    probs = {k: state_probs(p, pp["hh_index"], pp["rank"], K, n_hh) for k, p in rules.items()}
    ones = np.ones(n_hh)
    point = {k: metrics(ones, base, states, probs[k]) for k in rules}
    rng = np.random.default_rng(20261116)
    boot = {k: [] for k in rules}
    for _ in range(B):
        mult = np.bincount(rng.integers(0, n_hh, n_hh), minlength=n_hh).astype(float)
        for k in rules:
            boot[k].append(metrics(mult, base, states, probs[k]))
    names = ["cost_bn", "pov_bhc_pp", "pov_ahc_pp", "gini_change_pp"]
    out = {"K": K, "B": B, "rules": {}}
    for k in rules:
        bk = np.array(boot[k])
        out["rules"][k] = {nm: {"point": float(point[k][i]), "boot_se": float(bk[:, i].std(ddof=1))}
                           for i, nm in enumerate(names)}
    diff = np.array(boot["uniform"]) - np.array(boot["exposure"])
    out["uniform_minus_exposure"] = {
        nm: {"point": float(point["uniform"][i] - point["exposure"][i]),
             "boot_se": float(diff[:, i].std(ddof=1)),
             "ci95": [float(np.percentile(diff[:, i], 2.5)), float(np.percentile(diff[:, i], 97.5))]}
        for i, nm in enumerate(names)}
    (d / "contrasts.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
