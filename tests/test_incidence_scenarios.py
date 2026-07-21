"""Unit tests for the factorial incidence-family design (R2-3, R2-6a)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from uk_ai_study.shocks import PRESETS, apply_shocks

from incidence_scenarios import (
    COMPRESSION_TOP_TERTILE_MULTIPLIER,
    DEFAULT_FAMILIES,
    compression_mask,
    displacement_mask_for,
    shocked_table_for,
)


def make_persons(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    group = rng.integers(1, 10, n) * 1000.0
    exposure_by_group = {g * 1000.0: x for g, x in zip(range(1, 10), np.linspace(-0.7, 0.7, 9))}
    theta_by_group = {g * 1000.0: x for g, x in zip(range(1, 10), np.linspace(0.3, 0.9, 9))}
    employment = np.where(rng.random(n) < 0.75, rng.lognormal(10.1, 0.6, n), 0.0)
    return pd.DataFrame(
        {
            "age": rng.integers(16, 65, n).astype(float),
            "employment_income": employment,
            "savings_interest_income": rng.lognormal(4.0, 1.0, n),
            "dividend_income": rng.lognormal(3.0, 1.5, n),
            "weight": rng.uniform(500.0, 2000.0, n),
            "soc_major_group": group,
            "exposure": pd.Series(group).map(exposure_by_group).to_numpy(),
            "complementarity": pd.Series(group).map(theta_by_group).to_numpy(),
        }
    )


def test_default_families_share_survivor_wage_and_capital_channels():
    """R2-3: with wage_axis='central', only the displacement mask differs."""
    persons = make_persons()
    seed = 7
    tables = {f: shocked_table_for(f, persons, seed=seed) for f in DEFAULT_FAMILIES}
    employed = persons["employment_income"].to_numpy() > 0

    # capital channel identical everywhere
    ref = tables["exposure"]
    for f, t in tables.items():
        for col in ("savings_interest_income", "dividend_income"):
            np.testing.assert_allclose(
                t[col].to_numpy(), ref[col].to_numpy(), err_msg=f"{f}:{col}"
            )

    # survivor wage changes identical: anyone not displaced in a pair of
    # families has the same shocked employment income in both
    fams = list(tables)
    masks = {f: tables[f]["displaced"].to_numpy() for f in fams}
    assert any(
        not np.array_equal(masks[a], masks[b])
        for i, a in enumerate(fams)
        for b in fams[i + 1 :]
    ), "displacement masks should differ across families"
    for i, a in enumerate(fams):
        for b in fams[i + 1 :]:
            both_survive = employed & ~masks[a] & ~masks[b]
            assert both_survive.any()
            np.testing.assert_allclose(
                tables[a]["employment_income"].to_numpy()[both_survive],
                tables[b]["employment_income"].to_numpy()[both_survive],
                err_msg=f"survivor wage channel differs between {a} and {b}",
            )


def test_exposure_family_reproduces_apply_shocks():
    persons = make_persons()
    expected = apply_shocks(persons, PRESETS["central"], seed=3)
    got = shocked_table_for("exposure", persons, seed=3)
    for col in (
        "displaced",
        "employment_income",
        "savings_interest_income",
        "dividend_income",
    ):
        np.testing.assert_allclose(
            got[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float)
        )


def test_compound_variants_change_only_the_wage_axis():
    persons = make_persons()
    seed = 5
    for base in ("uniform", "compression"):
        default = shocked_table_for(base, persons, seed=seed)
        compound = shocked_table_for(f"{base}_compound", persons, seed=seed)
        via_flag = shocked_table_for(base, persons, seed=seed, wage_axis="family")
        # same displacement mask; different survivor wages
        np.testing.assert_array_equal(
            default["displaced"].to_numpy(), compound["displaced"].to_numpy()
        )
        assert not np.allclose(
            default["employment_income"].to_numpy(),
            compound["employment_income"].to_numpy(),
        )
        pd.testing.assert_frame_equal(compound, via_flag)
    # exposure/junior are identical under either axis
    for base in ("exposure", "junior"):
        pd.testing.assert_frame_equal(
            shocked_table_for(base, persons, seed=seed),
            shocked_table_for(base, persons, seed=seed, wage_axis="family"),
        )


def test_unknown_wage_axis_and_family_raise():
    persons = make_persons(500)
    with pytest.raises(ValueError):
        shocked_table_for("exposure", persons, wage_axis="banana")
    with pytest.raises(ValueError):
        shocked_table_for("nope", persons)


def _elite_mask(persons):
    from incidence_scenarios import _weighted_tertile_threshold

    w = persons["weight"].to_numpy()
    earnings = persons["employment_income"].to_numpy(dtype=float)
    exposure = persons["exposure"].to_numpy()
    employed = earnings > 0
    med = _weighted_tertile_threshold(exposure[employed], w[employed], 0.5)
    top = _weighted_tertile_threshold(earnings[employed], w[employed], 2 / 3)
    return employed & (exposure > med) & (earnings >= top)


@pytest.mark.parametrize("low,high", [(1.0, 2.0), (1.5, 3.0)])
def test_compression_multiplier_changes_draw_tilt(low, high):
    """R2-6a: a larger multiplier concentrates displacement on the elite."""
    persons = make_persons(6000)
    elite = _elite_mask(persons)
    w = persons["weight"].to_numpy()
    scenario = PRESETS["central"]

    def elite_share(mult):
        shares = []
        for seed in range(5):
            d = compression_mask(persons, scenario, seed=seed, multiplier=mult)
            shares.append(w[d & elite].sum() / w[d].sum())
        return float(np.mean(shares))

    assert elite_share(high) > elite_share(low)


def test_multiplier_flows_through_shocked_table_for():
    persons = make_persons()
    seed = 11
    d_default = shocked_table_for(persons=persons, family="compression", seed=seed)
    d_default2 = shocked_table_for(
        "compression",
        persons,
        seed=seed,
        compression_multiplier=COMPRESSION_TOP_TERTILE_MULTIPLIER,
    )
    d_tilted = shocked_table_for(
        "compression", persons, seed=seed, compression_multiplier=3.0
    )
    pd.testing.assert_frame_equal(d_default, d_default2)
    assert not np.array_equal(
        d_default["displaced"].to_numpy(), d_tilted["displaced"].to_numpy()
    )
    # multiplier only affects the compression draw, not other families
    pd.testing.assert_frame_equal(
        shocked_table_for("junior", persons, seed=seed),
        shocked_table_for("junior", persons, seed=seed, compression_multiplier=3.0),
    )


def test_masks_match_displacement_mask_for():
    persons = make_persons()
    for f in DEFAULT_FAMILIES:
        np.testing.assert_array_equal(
            shocked_table_for(f, persons, seed=2)["displaced"].to_numpy(),
            displacement_mask_for(f, persons, seed=2),
        )


def test_aggregate_quota_held_across_families():
    persons = make_persons()
    w = persons["weight"].to_numpy()
    employed = persons["employment_income"].to_numpy() > 0
    quota = PRESETS["central"].displacement_rate * w[employed].sum()
    for f in DEFAULT_FAMILIES:
        # the draws hold the quota in expectation (systematic sampling with
        # prescribed inclusion probabilities), so average over seeds
        realised = np.mean(
            [w[displacement_mask_for(f, persons, seed=s)].sum() for s in range(10)]
        )
        assert realised == pytest.approx(quota, rel=0.02), f
