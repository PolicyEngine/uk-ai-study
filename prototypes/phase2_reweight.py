"""Reweight Phase 2 state runs into expected outcomes.

Usage: python prototypes/phase2_reweight.py RUN_DIR SPEC_JSON
SPEC keys:
  q          take-up mixing weight for households with a displaced benefit
             unit (0 = survey flags, 1 = everyone claims; 0.75 central)
  spells     {"3": w3, "6": w6, ...} spell weights (sum to 1)
  mix        true: monthly benefit mix from out_jsa/out/re runs
  rule       "exposure" (sampler probabilities on the run's universe)
  source     run name to read directly when mix is false (e.g. "out", "tax12")
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uk_ai_study import shocks  # noqa: E402
from uk_ai_study.runner import gini  # noqa: E402


def probabilities(pp, rate, quota_universe_weight, exposure_rows):
    """Exposure-proportional prescribed probabilities on the eligible universe.

    The aggregate quota is ``rate`` x all employees; groups get quotas in
    proportion to eligible employment x mean shifted exposure.
    """
    import pandas as pd
    elig, w, g = pp["eligible"], pp["weight"], pp["group"]
    persons = pd.DataFrame({"employment_income": np.where(elig, 1.0, 0.0), "weight": w,
                            "exposure": exposure_rows, "soc_major_group": np.where(g < 0, np.nan, g)})
    quotas = shocks.group_quotas(persons, rate * quota_universe_weight / float(w[elig].sum()))
    p = np.zeros(len(w))
    for grp, q in quotas.items():
        members = np.flatnonzero(elig & (g == grp))
        if q["quota"] > 0:
            p[members] = shocks.prescribed_inclusion_probabilities(w[members], q["quota"], np.ones(len(members)))
    return p


def state_probs(p, hh_index, rank, K, n_hh):
    pr = np.zeros((n_hh, K))
    for r in range(K):
        sel = rank == r
        pr[hh_index[sel], r] = p[sel]
    probs = np.ones((2 ** K, n_hh))
    for b in range(2 ** K):
        for r in range(K):
            probs[b] *= pr[:, r] if (b >> r) & 1 else (1 - pr[:, r])
    return probs


def household_level(d, key):
    return dict(np.load(d / f"{key}.npz"))


def outcome(d, b, m, spec, spells):
    """Household outcomes (gov, hbai, equiv, equiv_ahc) for state b, mode m."""
    acc = None
    for s, ws in spells.items():
        s = int(s)
        if spec.get("mix") and s == 0:
            # re-employed for the whole year: the re run is both tax and benefit source
            t = household_level(d, f"s{b:02d}_{m}_re")
            gov, hbai = t["gov_balance"], t["hbai_household_net_income"]
        elif spec.get("mix"):
            t = household_level(d, f"s{b:02d}_{m}_tax{s}")
            bj, bo, br = (household_level(d, f"s{b:02d}_{m}_{k}")["household_benefits"]
                          for k in ("out_jsa", "out", "re"))
            j = min(s, 6)
            bmix = (j / 12) * bj + ((s - j) / 12) * bo + ((12 - s) / 12) * br
            gov = t["gov_balance"] + t["household_benefits"] - bmix
            hbai = np.maximum(0, t["hbai_household_net_income"] - t["household_benefits"] + bmix)
        else:
            t = household_level(d, f"s{b:02d}_{m}_{spec['source']}")
            gov, hbai = t["gov_balance"], t["hbai_household_net_income"]
        fac = np.divide(t["equiv_hbai_household_net_income"], t["hbai_household_net_income"],
                        out=np.zeros_like(hbai), where=t["hbai_household_net_income"] > 0)
        housing = t["hbai_household_net_income"] - t["hbai_household_net_income_ahc"]
        fac_a = np.divide(t["equiv_hbai_household_net_income_ahc"], t["hbai_household_net_income_ahc"],
                          out=fac.copy(), where=t["hbai_household_net_income_ahc"] > 0)
        o = {"gov": gov, "equiv": hbai * fac, "equiv_ahc": (hbai - housing) * fac_a,
             "thr": t["poverty_threshold_bhc"], "thr_a": t["poverty_threshold_ahc"]}
        acc = {k: ws * v for k, v in o.items()} if acc is None else {k: acc[k] + ws * v for k, v in o.items()}
    return acc


def main(run_dir: str, spec: dict) -> dict:
    d = Path(run_dir)
    cfg = json.loads((d / "config.json").read_text())
    K = cfg["K"]
    pp = dict(np.load(d / "persons.npz"))
    bf = dict(np.load(d / "baseline_flag.npz"))
    bc = dict(np.load(d / "baseline_claim.npz")) if (d / "baseline_claim.npz").exists() else bf
    n_hh = len(bf["hh_w"])
    w, n = bf["hh_w"], bf["hh_n"]
    if not (d / "exposure.npy").exists():
        from uk_ai_study.data import PERIOD, microsimulation
        from uk_ai_study.runner import build_person_table
        persons = build_person_table(microsimulation(), PERIOD)
        np.save(d / "exposure.npy", persons["exposure"].to_numpy(float))
        np.save(d / "employed.npy", persons["employment_income"].to_numpy(float) > 0)
    expo = np.load(d / "exposure.npy")
    all_emp_w = float(pp["weight"][np.load(d / "employed.npy")].sum())
    p = probabilities(pp, shocks.CENTRAL_DISPLACEMENT_RATE, all_emp_w, expo)
    probs = state_probs(p, pp["hh_index"], pp["rank"], K, n_hh)
    q = float(spec.get("q", 0.0))
    spells = spec.get("spells", {"12": 1.0})
    modes = cfg["modes"]
    # does state b displace someone in household h? (then take-up mixing applies)
    bu_of = {}
    E = {k: np.zeros(n_hh) for k in ("gov",)}
    pov_b = np.zeros(n_hh)
    pov_a = np.zeros(n_hh)
    vals, wts, vals_a = [], [], []
    for b in range(2 ** K):
        has_disp = np.zeros(n_hh, bool)
        sel = (pp["rank"] >= 0) & (pp["rank"] < K) & (((b >> np.clip(pp["rank"], 0, None)) & 1) == 1)
        has_disp[np.unique(pp["hh_index"][sel])] = True
        of = outcome(d, b, "flag", spec, spells)
        if q > 0 and "claim" in modes:
            oc = outcome(d, b, "claim", spec, spells)
            qh = np.where(has_disp, q, 0.0)
            # level = base_flag + q (Y_claim - base_claim) + (1-q) (Y_flag - base_flag)
            gov = bf["gov_balance"] + qh * (oc["gov"] - bc["gov_balance"]) + (1 - qh) * (of["gov"] - bf["gov_balance"])
            eq = bf["equiv_hbai_household_net_income"] + qh * (oc["equiv"] - bc["equiv_hbai_household_net_income"]) \
                + (1 - qh) * (of["equiv"] - bf["equiv_hbai_household_net_income"])
            eqa = bf["equiv_hbai_household_net_income_ahc"] + qh * (oc["equiv_ahc"] - bc["equiv_hbai_household_net_income_ahc"]) \
                + (1 - qh) * (of["equiv_ahc"] - bf["equiv_hbai_household_net_income_ahc"])
        else:
            gov, eq, eqa = of["gov"], of["equiv"], of["equiv_ahc"]
        E["gov"] += probs[b] * gov
        pov_b += probs[b] * (eq < of["thr"])
        pov_a += probs[b] * (eqa < of["thr_a"])
        keep = probs[b] > 0
        vals.append(eq[keep]); vals_a.append(eqa[keep]); wts.append((w * n * probs[b])[keep])
    pw = (w * n).sum()
    base_pb = (bf["equiv_hbai_household_net_income"] < bf["poverty_threshold_bhc"])
    base_pa = (bf["equiv_hbai_household_net_income_ahc"] < bf["poverty_threshold_ahc"])
    res = {
        "spec": spec,
        "cost_bn": float((w * (bf["gov_balance"] - E["gov"])).sum() / 1e9),
        "pov_bhc_pp": float(100 * ((w * n * pov_b).sum() - (w * n * base_pb).sum()) / pw),
        "pov_ahc_pp": float(100 * ((w * n * pov_a).sum() - (w * n * base_pa).sum()) / pw),
        "gini_change_pp": float(100 * (gini(np.concatenate(vals), np.concatenate(wts))
                                       - gini(bf["equiv_hbai_household_net_income"], w * n))),
        "expected_displaced_m": float(p @ pp["weight"] / 1e6),
    }
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {})
