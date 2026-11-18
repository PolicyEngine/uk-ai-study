"""Expected outcomes from saved state runs: rules, closure, variants, bootstrap.

Reads OUT_ROOT/<setting>/ written by statesim/run.py; no simulation calls.

Corrections to the Phase 2 prototype (prototypes/phase2_reweight.py):
- Spell length and take-up mode are extra state dimensions, not averages of
  income: each displaced household realises one spell s (weight w_s, one
  common spell per household) and one take-up mode (claim with probability
  q', else survey flag). Poverty and the Gini are therefore taken over the
  joint (state, mode, spell) distribution, so expected poverty is the mean of
  the poverty indicators (the prototype thresholded mean incomes, which
  understates poverty and inequality).
- Take-up is scored in levels against the survey-flag baseline: a displaced
  benefit unit that did not claim before claims with probability q' after
  displacement, and all UC it then receives counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from statesim import engine  # noqa: E402
from statesim.run import OUT_ROOT  # noqa: E402
from uk_ai_study import shocks  # noqa: E402

JUNIOR_MULT = 5.8 / 4.5
COMPRESSION_MULT = 2.0
HIGH_WAGE_MULT, LOW_WAGE_MULT = 9.6, 0.6
RATE = shocks.CENTRAL_DISPLACEMENT_RATE


class Setting:
    def __init__(self, name: str, root: Path = OUT_ROOT):
        self.dir = root / name
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.K = self.cfg["K"]
        self.pp = dict(np.load(self.dir / "persons.npz"))
        self.exposure = np.load(self.dir / "exposure.npy")
        self.employed = np.load(self.dir / "employed.npy")
        self.base = dict(np.load(self.dir / "baseline_flag.npz"))
        self.n_hh = len(self.base["hh_w"])
        self._cache: dict = {}

    def run(self, key: str) -> dict:
        if key not in self._cache:
            self._cache[key] = dict(np.load(self.dir / f"{key}.npz"))
        return self._cache[key]

    def has(self, key: str) -> bool:
        return (self.dir / f"{key}.npz").exists()


# ---------------------------------------------------------------- rules
def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return values[order][np.searchsorted(cw, q * cw[-1])]


def rule_probabilities(st: Setting, rule: str) -> np.ndarray:
    """Inclusion probabilities on the setting's eligible universe.

    The aggregate quota is 7 per cent of ALL employees in every rule.
    """
    pp = st.pp
    el, w, g = pp["eligible"].astype(bool), pp["weight"], pp["group"]
    quota = RATE * float(w[st.employed].sum())
    expo = st.exposure - st.exposure[st.employed].min()
    p = np.zeros(len(w))
    if rule in ("exposure", "junior", "uniform"):
        e = np.ones_like(expo) if rule == "uniform" else expo
        groups = np.unique(g[el])
        emp = {k: float(w[el & (g == k)].sum()) for k in groups}
        mean_e = {k: float(np.average(e[el & (g == k)], weights=w[el & (g == k)])) for k in groups}
        raw = {k: emp[k] * mean_e[k] for k in groups}
        if sum(raw.values()) <= 0:  # degenerate exposure: allocate by employment (as group_quotas)
            raw = dict(emp)
        scale = quota / sum(raw.values())
        tilt = np.where(pp["age"] < 25, JUNIOR_MULT, 1.0) if rule == "junior" else np.ones(len(w))
        for k in groups:
            m = np.flatnonzero(el & (g == k))
            if raw[k] > 0:
                p[m] = shocks.prescribed_inclusion_probabilities(w[m], raw[k] * scale, tilt[m])
        return p
    earn = pp["earnings"]
    idx = np.flatnonzero(el)
    if rule == "compression":
        med_e = _weighted_quantile(st.exposure[el], w[el], 0.5)
        top = _weighted_quantile(earn[el], w[el], 2 / 3)
        tilt = np.where((st.exposure > med_e) & (earn >= top), COMPRESSION_MULT, 1.0)
    elif rule == "top_loaded":
        med = _weighted_quantile(earn[el], w[el], 0.5)
        tilt = np.where(expo > 0, expo, 1e-9) * np.where(pp["age"] < 25, JUNIOR_MULT, 1.0) \
            * np.where(earn >= med, HIGH_WAGE_MULT, LOW_WAGE_MULT)
    else:
        raise ValueError(rule)
    p[idx] = shocks.prescribed_inclusion_probabilities(w[idx], quota, tilt[idx])
    return p


RULES = ("exposure", "junior", "compression", "uniform", "top_loaded")


# ---------------------------------------------------------------- closure
def closed_account(st: Setting, p: np.ndarray, spells: dict, penalty: float) -> dict:
    pp = st.pp
    w, e, el = pp["weight"], pp["earnings"], pp["eligible"].astype(bool)
    lost = engine.lost_pay_share(spells, penalty)
    saved = float((w * p * lost * (e + pp["employer_ni"] + pp["employer_pension"]))[el].sum())
    th = pp["complementarity"]
    th_bar = float((th * w)[el].sum() / w[el].sum())
    upl = st.cfg["wage_uplift"] * th / th_bar * e
    paid = float((w * (1 - p) * upl * (1 + engine.EMPLOYER_NI_MARGINAL))[el].sum())
    acc = engine.closed_account(saved, paid)
    base_div = float((w * pp["dividends"]).sum())
    acc.update(labour_cost_saved=saved, survivor_uplift_paid=paid, lost_pay_share=lost,
               baseline_dividends=base_div, dividend_factor=1 + acc["uk_household_payout"] / base_div)
    return acc


# ---------------------------------------------------------------- outcomes
def _household(st: Setting, b: int, m: str, s: int, mix: bool) -> dict:
    """gov, equivalised BHC/AHC income and thresholds for state b, mode m, spell s."""
    if s == 0:
        t = st.run(f"s{b:02d}_{m}_re")
        gov, hbai = t["gov_balance"], t["hbai_household_net_income"]
    elif mix and st.has(f"s{b:02d}_{m}_out_jsa"):
        t = st.run(f"s{b:02d}_{m}_tax{s}")
        bmix = engine.benefit_mix(s, st.run(f"s{b:02d}_{m}_out_jsa")["household_benefits"],
                                  st.run(f"s{b:02d}_{m}_out")["household_benefits"],
                                  st.run(f"s{b:02d}_{m}_re")["household_benefits"])
        gov = t["gov_balance"] + t["household_benefits"] - bmix
        hbai = np.maximum(0, t["hbai_household_net_income"] - t["household_benefits"] + bmix)
    else:
        t = st.run(f"s{b:02d}_{m}_tax{s}")
        gov, hbai = t["gov_balance"], t["hbai_household_net_income"]
    h0 = t["hbai_household_net_income"]
    fac = np.divide(t["equiv_hbai_household_net_income"], h0, out=np.zeros_like(h0), where=h0 > 0)
    housing = h0 - t["hbai_household_net_income_ahc"]
    ha = t["hbai_household_net_income_ahc"]
    fac_a = np.divide(t["equiv_hbai_household_net_income_ahc"], ha, out=fac.copy(), where=ha > 0)
    return {"gov": gov, "eq": hbai * fac, "eqa": (hbai - housing) * fac_a,
            "thr": t["poverty_threshold_bhc"], "thra": t["poverty_threshold_ahc"]}


def _blend(a: dict, b: dict, alpha: float) -> dict:
    return {k: a[k] + alpha * (b[k] - a[k]) for k in a}


def combos(st: Setting, p: np.ndarray, spells: dict, q: float, mix: bool = True,
           other: Setting | None = None, alpha: float = 0.0):
    """Yield (weight per household, outcome dict) over (state, mode, spell)."""
    pp = st.pp
    probs = engine.state_probabilities(p, pp["hh_index"], pp["rank"], st.K, st.n_hh)
    modes = st.cfg["modes"] if q > 0 else ["flag"]
    spell_items = [(int(s), float(ws)) for s, ws in spells.items()]
    for b in range(2 ** st.K):
        mask = engine.state_mask(pp["rank"], b, st.K)
        has = np.zeros(st.n_hh, bool)
        has[np.unique(pp["hh_index"][mask])] = True
        for m in modes:
            mw = np.where(has, q if m == "claim" else 1 - q, 1.0 if m == "flag" else 0.0)
            for i, (s, ws) in enumerate(spell_items):
                sw = np.where(has, ws, 1.0 if i == 0 else 0.0)
                wt = probs[b] * mw * sw
                if not wt.any():
                    continue
                o = _household(st, b, m, s, mix)
                if other is not None and alpha:
                    o = _blend(o, _household(other, b, m, s, mix), alpha)
                yield wt, o


def summarise(st: Setting, gen, mult: np.ndarray | None = None) -> dict:
    w = st.base["hh_w"] * (1.0 if mult is None else mult)
    n = st.base["hh_n"]
    gov = np.zeros(st.n_hh)
    pb = np.zeros(st.n_hh)
    pa = np.zeros(st.n_hh)
    vals, wts = [], []
    for wt, o in gen:
        gov += wt * o["gov"]
        pb += wt * (o["eq"] < o["thr"])
        pa += wt * (o["eqa"] < o["thra"])
        keep = wt * w > 0
        vals.append(o["eq"][keep])
        wts.append((w * n * wt)[keep])
    b = st.base
    pw = (w * n).sum()
    base_pb = b["equiv_hbai_household_net_income"] < b["poverty_threshold_bhc"]
    base_pa = b["equiv_hbai_household_net_income_ahc"] < b["poverty_threshold_ahc"]
    return {
        "cost_bn": float((w * (b["gov_balance"] - gov)).sum() / 1e9),
        "pov_bhc_pp": float(100 * ((w * n * pb).sum() - (w * n * base_pb).sum()) / pw),
        "pov_ahc_pp": float(100 * ((w * n * pa).sum() - (w * n * base_pa).sum()) / pw),
        "gini_change_pp": float(100 * (engine.gini(np.concatenate(vals), np.concatenate(wts))
                                       - engine.gini(b["equiv_hbai_household_net_income"], w * n))),
    }


def evaluate(f100: Setting, f115: Setting | None, rule: str, spells: dict, q: float,
             closure: bool = True, mult=None) -> dict:
    p = rule_probabilities(f100, rule)
    out = {"rule": rule, "q": q, "spells": spells,
           "expected_displaced_m": float(p @ f100.pp["weight"] / 1e6),
           "coverage_gap_m": engine.coverage_gap(p, f100.pp["weight"], f100.pp["rank"], f100.K) / 1e6}
    alpha, acc = 0.0, None
    if closure and f115 is not None:
        acc = closed_account(f100, p, spells, f100.cfg["penalty"])
        fa, fb = f100.cfg["dividend_factor"], f115.cfg["dividend_factor"]
        alpha = (acc["dividend_factor"] - fa) / (fb - fa)
    res = summarise(f100, combos(f100, p, spells, q, other=f115, alpha=alpha), mult)
    if acc is not None:
        # corporation tax is an accounting line on population totals; under the
        # household bootstrap it is held at its full-sample value
        res["cost_bn"] -= acc["corporation_tax"] / 1e9
        out["closed_account_bn"] = {k: v / 1e9 if k not in ("lost_pay_share", "dividend_factor") else v
                                    for k, v in acc.items()}
    out.update(res)
    return out


def main(out_path: str, boot: int = 100) -> None:  # pragma: no cover - needs the state runs
    f100, f115 = Setting("central_f100"), Setting("central_f115")
    sw = {str(k): v for k, v in engine.SPELL_WEIGHTS.items()}
    q = engine.TAKEUP_Q
    res = {"central": {}, "variants": {}, "bootstrap": {}}
    for r in RULES:
        res["central"][r] = evaluate(f100, f115, r, sw, q)
        print(r, {k: round(v, 3) for k, v in res["central"][r].items() if isinstance(v, float)}, flush=True)
    res["variants"]["stress_12m"] = evaluate(f100, f115, "exposure", {"12": 1.0}, q)
    res["variants"]["survey_takeup"] = evaluate(f100, f115, "exposure", sw, 0.0)
    res["variants"]["no_closure"] = evaluate(f100, None, "exposure", sw, q, closure=False)
    rng = np.random.default_rng(20261118)
    draws = {r: [] for r in ("exposure", "uniform")}
    for _ in range(boot):
        mult = np.bincount(rng.integers(0, f100.n_hh, f100.n_hh), minlength=f100.n_hh).astype(float)
        for r in draws:
            e = evaluate(f100, f115, r, sw, q, mult=mult)
            draws[r].append([e["cost_bn"], e["pov_bhc_pp"], e["pov_ahc_pp"], e["gini_change_pp"]])
    names = ["cost_bn", "pov_bhc_pp", "pov_ahc_pp", "gini_change_pp"]
    for r, d in draws.items():
        a = np.array(d)
        res["bootstrap"][r] = {nm: float(a[:, i].std(ddof=1)) for i, nm in enumerate(names)}
    diff = np.array(draws["uniform"]) - np.array(draws["exposure"])
    res["bootstrap"]["uniform_minus_exposure_ci95"] = {
        nm: [float(np.percentile(diff[:, i], 2.5)), float(np.percentile(diff[:, i], 97.5))]
        for i, nm in enumerate(names)}
    Path(out_path).write_text(json.dumps(res, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100)
