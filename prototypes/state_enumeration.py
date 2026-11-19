"""Prototype: exact expectations by enumerating household displacement states.

Phase 1 of the round-4 revision (docs/referee/assignment.md, option A). Not a
pipeline stage: it lives outside analysis/ so the attested build is untouched.

A household's shocked outcome depends only on which of its employees are
displaced (the survivor uplift is normalised over baseline workers and the
capital shock does not depend on displacement). For K employee ranks per
household, run one shocked simulation per bitmask b in 0..2^K-1 (employee of
within-household rank r is displaced iff bit r of b is set), save household
outcomes, then reweight: E[Y] = sum_h w_h sum_b P_h(b) Y_h(b), with P_h(b) the
product of the members' displacement probabilities (independent within the
household), using the sampler's own prescribed inclusion probabilities.

Employees of rank >= K are never displaced in the runs; their expected
displaced weight is reported as the coverage gap.

Usage: python prototypes/state_enumeration.py OUT_DIR [K]
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uk_ai_study import shocks  # noqa: E402
from uk_ai_study.data import PERIOD, microsimulation  # noqa: E402
from uk_ai_study.runner import build_person_table, gini  # noqa: E402


def household_outcomes(sim, hh_index, n_hh):
    """Per-household gov balance, poor-person counts, equivalised income."""
    c = lambda v, m: sim.calculate(v, period=PERIOD, map_to=m).values  # noqa: E731
    pov_b = np.bincount(hh_index, weights=c("in_poverty_bhc", "person"), minlength=n_hh)
    pov_a = np.bincount(hh_index, weights=c("in_poverty_ahc", "person"), minlength=n_hh)
    return {
        "gov": c("gov_balance", "household"),
        "pov_bhc": pov_b,
        "pov_ahc": pov_a,
        "equiv": c("equiv_hbai_household_net_income", "household"),
    }


def main(out_dir: str, K: int = 3) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base_sim = microsimulation()
    persons = build_person_table(base_sim, PERIOD)
    c = lambda v, m: base_sim.calculate(v, period=PERIOD, map_to=m).values  # noqa: E731
    hh_ids = c("household_id", "household")
    p_hh = c("person_household_id", "person")
    pos = {h: i for i, h in enumerate(hh_ids)}
    hh_index = np.array([pos[h] for h in p_hh])
    n_hh = len(hh_ids)
    hh_w = c("household_weight", "household")
    hh_n = c("household_count_people", "household")

    employed, weight, _, group = shocks._employee_groups(persons)
    # within-household employee rank (stable by person order)
    rank = np.full(len(persons), -1)
    seen = {}
    for i in np.flatnonzero(employed):
        h = hh_index[i]
        rank[i] = seen.get(h, 0)
        seen[h] = rank[i] + 1

    base = household_outcomes(base_sim, hh_index, n_hh)
    np.savez_compressed(out / "baseline.npz", hh_w=hh_w, hh_n=hh_n, **base)
    meta = {"K": K, "n_households": int(n_hh), "seconds": {}}

    scen = shocks.PRESETS["central"]
    original_draw = shocks.draw_displaced
    for b in range(2 ** K):
        t = time.time()
        mask = employed & (rank >= 0) & (rank < K) & (((b >> np.clip(rank, 0, None)) & 1) == 1)
        shocks.draw_displaced = lambda persons_, scenario_, seed=0, m=mask: m
        try:
            table = shocks.apply_shocks(persons, scen, seed=0)
        finally:
            shocks.draw_displaced = original_draw
        sim = shocks.build_shocked_simulation(base_sim, table, PERIOD)
        res = household_outcomes(sim, hh_index, n_hh)
        np.savez_compressed(out / f"state_{b:02d}.npz", **res)
        del sim, table
        gc.collect()
        meta["seconds"][b] = round(time.time() - t, 1)
        print(f"state {b} done in {meta['seconds'][b]}s", flush=True)

    np.savez_compressed(out / "persons.npz", hh_index=hh_index, rank=rank, employed=employed,
                        weight=weight, group=group, age=persons["age"].to_numpy())
    meta["total_seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(reweight(out, persons, scen))


def person_probabilities(persons, scenario):
    """The sampler's prescribed inclusion probability for every employee."""
    employed, weight, _, group = shocks._employee_groups(persons)
    quotas = shocks.group_quotas(persons, scenario.displacement_rate)
    p = np.zeros(len(persons))
    age = persons["age"].to_numpy()
    for g, q in quotas.items():
        members = np.flatnonzero(employed & (group == g))
        if q["quota"] <= 0:
            continue
        m = np.ones(len(members))
        if scenario.youth_displacement_multiplier != 1.0:
            m = m * np.where(age[members] < 25, scenario.youth_displacement_multiplier, 1.0)
        p[members] = shocks.prescribed_inclusion_probabilities(weight[members], q["quota"], m)
    return p


def reweight(out: Path, persons, scenario) -> dict:
    meta = json.loads((out / "meta.json").read_text())
    K = meta["K"]
    base = dict(np.load(out / "baseline.npz"))
    pp = dict(np.load(out / "persons.npz"))
    hh_index, rank, weight = pp["hh_index"], pp["rank"], pp["weight"]
    n_hh = len(base["hh_w"])
    p = person_probabilities(persons, scenario)
    # P_h(b) = prod over ranks r<K of p_r^{bit} (1-p_r)^{1-bit}; missing ranks have p=0
    p_rank = np.zeros((n_hh, K))
    for r in range(K):
        sel = rank == r
        p_rank[hh_index[sel], r] = p[sel]
    probs = np.ones((2 ** K, n_hh))
    for b in range(2 ** K):
        for r in range(K):
            bit = (b >> r) & 1
            probs[b] *= p_rank[:, r] if bit else (1 - p_rank[:, r])
    w, n = base["hh_w"], base["hh_n"]
    states = [dict(np.load(out / f"state_{b:02d}.npz")) for b in range(2 ** K)]
    e_gov = sum(probs[b] * states[b]["gov"] for b in range(2 ** K))
    e_pb = sum(probs[b] * states[b]["pov_bhc"] for b in range(2 ** K))
    e_pa = sum(probs[b] * states[b]["pov_ahc"] for b in range(2 ** K))
    persons_w = float((w * n).sum())
    vals = np.concatenate([states[b]["equiv"] for b in range(2 ** K)])
    wts = np.concatenate([w * n * probs[b] for b in range(2 ** K)])
    covered = weight[(rank >= 0) & (rank < K)] @ p[(rank >= 0) & (rank < K)]
    missed = weight[rank >= K] @ p[rank >= K]
    res = {
        "cost_bn": float((w * (base["gov"] - e_gov)).sum() / 1e9),
        "pov_bhc_pp": float(100 * ((w * e_pb).sum() - (w * base["pov_bhc"]).sum()) / persons_w),
        "pov_ahc_pp": float(100 * ((w * e_pa).sum() - (w * base["pov_ahc"]).sum()) / persons_w),
        "gini_change_pp": float(100 * (gini(vals, wts) - gini(base["equiv"], w * n))),
        "expected_displaced_m_covered": float(covered / 1e6),
        "expected_displaced_m_missed_rank_ge_K": float(missed / 1e6),
        "state_probability_sum_check": float(np.abs(probs.sum(0) - 1).max()),
    }
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
