"""Check the state enumeration against the paper's actual draws.

For each seed, draw the sampler's real displaced mask for a rule (the same code
the pipeline uses), map each household to the saved state matching its
displaced employees, and sum outcomes. Seeds 0-49 must reproduce
results/robustness/incidence_draws_five.csv; many more seeds give the long-run
mean of the sampler (which, unlike the enumeration, does not assume
independence within the household).

Usage: python prototypes/replicate_draws.py STATES_DIR N_SEEDS
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from uk_ai_study.data import PERIOD, microsimulation  # noqa: E402
from uk_ai_study.runner import build_person_table, gini  # noqa: E402


def main(states_dir: str, n_seeds: int) -> None:
    import incidence_scenarios as inc
    import measured_incidence as mi
    from uk_ai_study.shocks import PRESETS

    d = Path(states_dir)
    K = json.loads((d / "meta.json").read_text())["K"]
    base = dict(np.load(d / "baseline.npz"))
    pp = dict(np.load(d / "persons.npz"))
    states = [dict(np.load(d / f"state_{b:02d}.npz")) for b in range(2 ** K)]
    gov = np.stack([s["gov"] for s in states])
    pb = np.stack([s["pov_bhc"] for s in states])
    pa = np.stack([s["pov_ahc"] for s in states])
    eq = np.stack([s["equiv"] for s in states])
    hh_index, rank = pp["hh_index"], pp["rank"]
    n_hh = len(base["hh_w"])
    w, n = base["hh_w"], base["hh_n"]
    pw = (w * n).sum()
    persons = build_person_table(microsimulation(), PERIOD)
    weight = persons["weight"].to_numpy()
    scen = PRESETS["central"]

    def mask(rule, seed):
        if rule == "klein_top_loaded":
            return mi.draw_measured(persons, scen, mi.LONDON_MULT_CENTRAL, seed=seed)
        return inc.displacement_mask_for(rule, persons, seed=seed)

    hh_ar = np.arange(n_hh)
    rows = []
    for rule in ["exposure", "uniform", "junior", "compression", "klein_top_loaded"]:
        for seed in range(n_seeds):
            m = mask(rule, seed)
            sel = m & (rank >= 0)
            state = np.zeros(n_hh, dtype=np.int64)
            np.add.at(state, hh_index[sel], (1 << rank[sel]).astype(np.int64))
            g = gov[state, hh_ar]
            row = {"rule": rule, "seed": seed,
                   "cost_bn": float((w * (base["gov"] - g)).sum() / 1e9),
                   "pov_bhc_pp": float(100 * (w * (pb[state, hh_ar] - base["pov_bhc"])).sum() / pw),
                   "pov_ahc_pp": float(100 * (w * (pa[state, hh_ar] - base["pov_ahc"])).sum() / pw),
                   "gini_change_pp": float(100 * (gini(eq[state, hh_ar], w * n) - gini(base["equiv"], w * n))),
                   "displaced_m": float(weight[m].sum() / 1e6)}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(d / f"replicated_draws_{n_seeds}.csv", index=False)

    paper = pd.read_csv(ROOT / "results/robustness/incidence_draws_five.csv")
    paper = paper[paper.wage_axis == "central"]
    cmp = df.merge(paper, on=["seed"], suffixes=("", "_paper"), how="inner")
    cmp = cmp[cmp.rule == cmp.family]
    out = {}
    for col, pc in [("cost_bn", "exchequer_cost_bn"), ("pov_bhc_pp", "poverty_change_bhc_pp"),
                    ("gini_change_pp", "gini_change_pp"), ("displaced_m", "displaced_weighted_m")]:
        diff = (cmp[col] - cmp[pc]).abs()
        out[f"max_abs_diff_{col}"] = float(diff.max())
    piv = df.pivot(index="seed", columns="rule")
    gap = piv["pov_bhc_pp"]["uniform"] - piv["pov_bhc_pp"]["exposure"]
    out["n_seeds"] = n_seeds
    out["uniform_minus_exposure_bhc_mean"] = float(gap.mean())
    out["uniform_minus_exposure_bhc_se"] = float(gap.std(ddof=1) / np.sqrt(len(gap)))
    out["means"] = df.groupby("rule")[["cost_bn", "pov_bhc_pp", "pov_ahc_pp", "gini_change_pp", "displaced_m"]].mean().round(3).to_dict()
    (d / f"replication_check_{n_seeds}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
