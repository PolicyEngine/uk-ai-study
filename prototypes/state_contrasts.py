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
sys.path.insert(0, str(ROOT / "analysis"))

from uk_ai_study import shocks  # noqa: E402
from uk_ai_study.data import PERIOD, microsimulation  # noqa: E402
from uk_ai_study.runner import build_person_table, gini  # noqa: E402
from state_enumeration import person_probabilities  # noqa: E402


def all_employee_probabilities(persons, rate, tilt):
    """Prescribed probabilities for an all-employee weighted quota with a tilt."""
    employed, weight, _, _ = shocks._employee_groups(persons)
    p = np.zeros(len(persons))
    idx = np.flatnonzero(employed)
    p[idx] = shocks.prescribed_inclusion_probabilities(
        weight[idx], rate * float(weight[idx].sum()), np.asarray(tilt, float)[idx])
    return p


def rule_probabilities(persons, scen):
    """Each default incidence rule's inclusion probabilities, from the rule code."""
    import incidence_scenarios as inc
    import measured_incidence as mi

    flat = persons.copy()
    flat["exposure"] = 1.0
    w = persons["weight"].to_numpy()
    earnings = persons["employment_income"].to_numpy(dtype=float)
    exposure = persons["exposure"].to_numpy()
    employed = earnings > 0
    med_exp = inc._weighted_tertile_threshold(exposure[employed], w[employed], 0.5)
    top_earn = inc._weighted_tertile_threshold(earnings[employed], w[employed], 2 / 3)
    top_tier = employed & (exposure > med_exp) & (earnings >= top_earn)
    return {
        "exposure": person_probabilities(persons, scen),
        "junior": person_probabilities(persons, shocks.PRESETS["central_youth_tilted"]),
        "compression": all_employee_probabilities(
            persons, scen.displacement_rate,
            np.where(top_tier, inc.COMPRESSION_TOP_TERTILE_MULTIPLIER, 1.0)),
        "uniform": person_probabilities(flat, scen),
        "klein_top_loaded": all_employee_probabilities(
            persons, scen.displacement_rate, mi.composite_tilt(persons, mi.LONDON_MULT_CENTRAL)),
    }


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
    rules = rule_probabilities(persons, scen)
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
    out["minus_exposure"] = {}
    for k in rules:
        if k == "exposure":
            continue
        diff = np.array(boot[k]) - np.array(boot["exposure"])
        out["minus_exposure"][k] = {
            nm: {"point": float(point[k][i] - point["exposure"][i]),
                 "boot_se": float(diff[:, i].std(ddof=1)),
                 "ci95": [float(np.percentile(diff[:, i], 2.5)), float(np.percentile(diff[:, i], 97.5))]}
            for i, nm in enumerate(names)}
    out["expected_displaced_m"] = {k: float(p @ shocks._employee_groups(persons)[1] / 1e6)
                                   for k, p in rules.items()}
    (d / "contrasts_all_rules.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
