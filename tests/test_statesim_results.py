"""Tests for statesim/results.py on a synthetic two-household setting."""
import json

import numpy as np
import pytest

from statesim import results


def _write(d, name, **arrays):
    np.savez_compressed(d / f"{name}.npz", **arrays)


def fake_setting(tmp_path, gov, eq, thr=10.0):
    """K=1; household 0 has one eligible employee (p from the rule), household 1 none.

    gov[b], eq[b]: household arrays for state b (b=0 not displaced, b=1 displaced).
    """
    d = tmp_path / "fake"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"K": 1, "modes": ["flag"], "wage_uplift": 0.0,
                                               "penalty": 0.0, "dividend_factor": 1.0}))
    np.savez_compressed(d / "persons.npz", hh_index=np.array([0, 1]), rank=np.array([0, -1]),
                        eligible=np.array([True, False]), weight=np.array([1.0, 1.0]),
                        group=np.array([1.0, -1.0]), age=np.array([40.0, 40.0]),
                        earnings=np.array([30000.0, 0.0]))
    np.save(d / "exposure.npy", np.array([1.0, 0.0]))
    np.save(d / "employed.npy", np.array([True, False]))
    hh = dict(hh_w=np.array([1.0, 1.0]), hh_n=np.array([1.0, 1.0]))
    def outc(g, e):
        return dict(gov_balance=np.array(g), household_benefits=np.zeros(2),
                    hbai_household_net_income=np.array(e), equiv_hbai_household_net_income=np.array(e),
                    hbai_household_net_income_ahc=np.array(e), equiv_hbai_household_net_income_ahc=np.array(e),
                    poverty_threshold_bhc=np.full(2, thr), poverty_threshold_ahc=np.full(2, thr))
    _write(d, "baseline_flag", **hh, **outc(gov[0], eq[0]))
    for b in (0, 1):
        _write(d, f"s{b:02d}_flag_tax12", **outc(gov[b], eq[b]))
    return results.Setting("fake", root=tmp_path)


def test_expectation_uses_poverty_indicators_not_mean_income(tmp_path):
    # displaced: income 5 (poor); not displaced: income 20. With p, expected
    # poverty in household 0 is p, not 1(mean income < 10).
    st = fake_setting(tmp_path, gov=[[100.0, 50.0], [60.0, 50.0]], eq=[[20.0, 20.0], [5.0, 20.0]])
    p = np.array([0.3, 0.0])
    res = results.summarise(st, results.combos(st, p, {"12": 1.0}, q=0.0, mix=False))
    assert res["pov_bhc_pp"] == pytest.approx(100 * 0.3 / 2)  # 2 people, household 0 poor w.p. 0.3
    assert res["cost_bn"] == pytest.approx(0.3 * 40.0 / 1e9)


def test_rule_probabilities_hit_the_quota(tmp_path):
    st = fake_setting(tmp_path, gov=[[0.0, 0.0], [0.0, 0.0]], eq=[[1.0, 1.0], [1.0, 1.0]])
    p = results.rule_probabilities(st, "exposure")
    assert (p * st.pp["weight"]).sum() == pytest.approx(results.RATE * 1.0)
