"""Take-up and new-style JSA runs (analysis/displaced_takeup_runs.py):
pure helpers on fake data. No simulation is run."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import displaced_takeup_runs as tr  # noqa: E402


def test_design_is_123_simulations_and_82_rows():
    pairs = sum(len(s) for s in tr.SCENARIO_SEEDS.values())
    assert pairs == 41
    # a paired baseline and one shock per variant for every (scenario, seed)
    assert pairs * (1 + len(tr.VARIANTS)) == 123
    assert tr.EXPECTED_ROWS == 82
    assert tr.SCENARIO_SEEDS["disp_only"] == (0,)


def test_displaced_benunits_and_flags():
    person_bu = np.array([10, 10, 11, 12, 12])
    bu = np.array([10, 11, 12])
    displaced = np.array([False, True, False, False, False])
    d = tr.displaced_benunits(person_bu, bu, displaced)
    assert d.tolist() == [True, False, False]
    flags = tr.takeup_flags(np.array([False, False, True]), d)
    assert flags.tolist() == [True, False, True]


def test_jsa_eligibility_ages_and_earnings_condition():
    lel = 129.0
    age = np.array([17, 18, 24, 25, 65, 66, 40, 40])
    earn = np.array([20e3, 20e3, 20e3, 20e3, 20e3, 20e3, 50 * lel - 1, 50 * lel])
    displaced = np.array([True] * 7 + [True])
    e = tr.jsa_eligible(age, earn, displaced, lel)
    assert e.tolist() == [False, True, True, True, True, False, False, True]
    assert not tr.jsa_eligible(age, earn, np.zeros(8, bool), lel).any()
    award = tr.jsa_award(age, e, 95.55, 75.65)
    assert award[1] == pytest.approx(26 * 75.65)
    assert award[3] == pytest.approx(26 * 95.55)
    assert award[0] == 0.0 and award[5] == 0.0


def test_jsa_input_replaces_only_eligible():
    base = np.array([0.0, 500.0, 300.0])
    award = np.array([2484.3, 0.0, 2484.3])
    elig = np.array([True, False, True])
    assert tr.jsa_reported_input(base, award, elig).tolist() == [2484.3, 500.0, 2484.3]


def test_jsa_taxed_reads_the_tax_base():
    v = {"social_security_income": SimpleNamespace(adds=["state_pension", "jsa_contrib_reported"]),
         "taxable_social_security_income": SimpleNamespace(adds=["social_security_income"])}
    assert tr.jsa_taxed(v)
    v["social_security_income"] = SimpleNamespace(adds=["state_pension"])
    assert not tr.jsa_taxed(v)
    v["social_security_income"] = SimpleNamespace(adds=None)
    assert not tr.jsa_taxed(v)


def test_obr_table_zeroes_only_displaced_earnings():
    p = pd.DataFrame({"employment_income": [30e3, 50e3], "dividend_income": [1.0, 2.0]})
    s = p.assign(displaced=[False, True], employment_income=[31e3, 0.0], dividend_income=[9.0, 9.0])
    t = tr.obr_table(p, s)
    assert t["employment_income"].tolist() == [30e3, 0.0]
    assert t["dividend_income"].tolist() == [1.0, 2.0]
    assert t["displaced"].tolist() == [False, True]


def test_check_pairing():
    ref = pd.Series({0: 1.95, 1: 2.0})
    tr.check_pairing(1.95, ref, 0, "central")
    with pytest.raises(AssertionError):
        tr.check_pairing(1.96, ref, 0, "central")
    with pytest.raises(AssertionError):
        tr.check_pairing(1.95, ref, 5, "central")


def metrics(**kw):
    base = {v: 0.0 for v in list(tr.BN_CHANGES.values()) + list(tr.PP_CHANGES.values())}
    base.update(kw)
    return base


def test_draw_changes_signs_and_bases():
    b = metrics(gov=100e9, income_tax=50e9, ni_employee=10e9, national_insurance=11e9, benefits=40e9,
                cash_benefits=35e9, pov_bhc=0.20, gini=0.30)
    s = metrics(gov=90e9, income_tax=46e9, ni_employee=9e9, national_insurance=9.5e9, benefits=41e9,
                cash_benefits=37e9, pov_bhc=0.21, gini=0.305)
    c = tr.draw_changes(b, s)
    assert c["exchequer_cost_bn"] == pytest.approx(10.0)
    assert c["household_basis_cost_bn"] == pytest.approx(1.0 + 5.0)
    assert c["household_basis_cash_cost_bn"] == pytest.approx(2.0 + 5.0)
    assert c["narrow_cash_change_bn"] == pytest.approx(-4.0 - 1.5 - 1.0)
    assert c["poverty_change_bhc_pp"] == pytest.approx(1.0)
    assert c["gini_change_pp"] == pytest.approx(0.5)


def test_decile_flows_weighted_sums():
    out = tr.decile_flows([1, 1, 2], [2.0, 1.0, 3.0], {"hh_uc": [10.0, 5.0, 1.0]}).set_index("decile")
    assert out.loc[1, "hh_uc"] == pytest.approx(25.0)
    assert out.loc[2, "hh_uc"] == pytest.approx(3.0)


def test_displaced_diagnostics():
    # four benefit units, three displaced (weights 1, 2, 1), one not
    d = tr.displaced_diagnostics(
        bu_weight=[1.0, 2.0, 1.0, 5.0], displaced_bu=[True, True, True, False],
        flag_base=[True, False, True, True], entitled_before=[False, False, True, False],
        entitled_after=[True, True, True, False], capital=[0.0, 20e3, 1e3, 1e6], capital_limit=16e3,
        other_earner=[False, True, True, False], owner=[True, True, False, True],
        person_weight=[1.0, 3.0, 1.0], displaced=[True, True, False], baseline_earnings=[40e3, 60e3, 10e3])
    assert d["flag_true_share"] == pytest.approx(2 / 4)
    assert d["entitled_after_share"] == pytest.approx(1.0)
    assert d["flag_true_share_entitled_after"] == pytest.approx(2 / 4)
    # newly entitled: units 0 and 1 (unit 2 was entitled before)
    assert d["flag_true_share_newly_entitled"] == pytest.approx(1 / 3)
    assert d["over_capital_limit_share"] == pytest.approx(2 / 4)
    assert d["other_earner_share"] == pytest.approx(3 / 4)
    assert d["owner_share"] == pytest.approx(3 / 4)
    assert d["mean_lost_earnings_gbp"] == pytest.approx((40e3 + 3 * 60e3) / 4)
    assert d["displaced_benunits_m"] == pytest.approx(4e-6)


def test_existing_unemployed_reference():
    r = tr.existing_unemployed_reference(
        bu_weight=[1.0, 1.0, 2.0, 10.0], n_unemployed=[1, 2, 1, 0], flag=[True, False, True, True],
        entitled=[True, True, False, True], uc=[6000.0, 9000.0, 0.0, 50_000.0])
    assert r["persons_m"] == pytest.approx(5e-6)
    assert r["uc_per_person_gbp"] == pytest.approx((6000 + 9000) / 5)
    assert r["entitled_share"] == pytest.approx(2 / 4)
    assert r["flag_true_share_entitled"] == pytest.approx(1 / 2)


def test_summarise_single_draw_has_nan_spread():
    s = tr.summarise(pd.DataFrame({"exchequer_cost_bn": [7.0]}))
    assert s["exchequer_cost_bn"]["mean"] == 7.0 and np.isnan(s["exchequer_cost_bn"]["se"])
    s = tr.summarise(pd.DataFrame({"exchequer_cost_bn": [7.0, 9.0]}))
    assert s["exchequer_cost_bn"]["se"] == pytest.approx(1.0)
