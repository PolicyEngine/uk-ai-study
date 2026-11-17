"""Closed capital account for the Phase 2 central case, and the interpolation.

Expected profit change for private employers = labour cost saved on the
displaced (spell- and penalty-weighted earnings, employer NI and employer
pension contributions) minus the survivor uplift they pay (with employer NI).
Corporation tax 25%, payout 50%, UK-household share 50%; the UK payout is
allocated in proportion to baseline dividends via a factor f*. Household
outcomes at f* are interpolated linearly between the f = 1 and f = 1.15 runs,
written to OUT_DIR, which phase2_reweight.py then reads.

Usage: python prototypes/phase2_closure.py RUN_F1 RUN_F115 OUT_DIR SPELLS_JSON PENALTY
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))

from uk_ai_study import shocks  # noqa: E402
from uk_ai_study.data import PERIOD, microsimulation  # noqa: E402
from uk_ai_study.runner import build_person_table  # noqa: E402
from phase2_reweight import probabilities  # noqa: E402

CT, PAYOUT, UK_SHARE = shocks.CORPORATION_TAX_RATE, 0.5, 0.5
EMPLOYER_NI_MARGINAL = 0.15  # employer NI rate on pay above the secondary threshold


def main(f1: str, f2: str, out_dir: str, spells: dict, penalty: float) -> None:
    d1, d2, out = Path(f1), Path(f2), Path(out_dir)
    cfg1 = json.loads((d1 / "config.json").read_text())
    cfg2 = json.loads((d2 / "config.json").read_text())
    fa, fb = cfg1["dividend_factor"], cfg2["dividend_factor"]
    sim = microsimulation()
    persons = build_person_table(sim, PERIOD)
    c = lambda v: np.asarray(sim.calculate(v, period=PERIOD, map_to="person").values, float)  # noqa: E731
    pp = dict(np.load(d1 / "persons.npz"))
    w = persons["weight"].to_numpy(float)
    e = persons["employment_income"].to_numpy(float)
    employed = e > 0
    private = pp["eligible"]
    public = employed & ~private
    er_ni, er_pen = c("ni_employer"), c("employer_pension_contributions")
    div = persons["dividend_income"].to_numpy(float)
    p = probabilities(pp, shocks.CENTRAL_DISPLACEMENT_RATE, float(w[employed].sum()), np.load(d1 / "exposure.npy"))

    lost_share = sum(float(ws) * (int(s) / 12 + (12 - int(s)) / 12 * penalty) for s, ws in spells.items())
    saved = float((w * p * lost_share * (e + er_ni + er_pen))[private].sum())
    theta = persons["complementarity"].to_numpy(float)
    theta_bar = float((theta * w)[employed].sum() / w[employed].sum())
    uplift = 0.026 * theta / theta_bar * e
    paid_private = float((w * (1 - p) * uplift * (1 + EMPLOYER_NI_MARGINAL))[private].sum())
    paid_public = float((w * uplift * (1 + EMPLOYER_NI_MARGINAL))[public].sum())
    profit = saved - paid_private
    ct = CT * profit
    uk_payout = (1 - CT) * PAYOUT * UK_SHARE * profit
    base_div = float((w * div).sum())
    f_star = 1 + uk_payout / base_div
    alpha = (f_star - fa) / (fb - fa)

    out.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "persons.npz", "exposure.npy", "employed.npy", "baseline_flag.npz", "baseline_claim.npz"):
        if (d1 / fname).exists():
            shutil.copy(d1 / fname, out / fname)
    n = 0
    for f in sorted(d1.glob("s*_*.npz")):
        a, b = dict(np.load(f)), dict(np.load(d2 / f.name))
        np.savez_compressed(out / f.name, **{k: a[k] + alpha * (b[k] - a[k]) for k in a})
        n += 1
    res = {"lost_share_of_annual_pay": lost_share, "labour_cost_saved_bn": saved / 1e9,
           "survivor_uplift_paid_private_bn": paid_private / 1e9, "profit_bn": profit / 1e9,
           "corporation_tax_bn": ct / 1e9, "uk_household_payout_bn": uk_payout / 1e9,
           "baseline_dividends_bn": base_div / 1e9, "dividend_factor": f_star, "alpha": alpha,
           "public_survivor_uplift_cost_bn": paid_public / 1e9, "interpolated_files": n}
    (out / "closure.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4]), float(sys.argv[5]))
