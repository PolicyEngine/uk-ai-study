"""Regression tests for issue #6: person-broadcast household income must be
divided by household size before per-person cash aggregation."""

import numpy as np
import pytest

from uk_ai_study.runner import per_capita_household_income


def test_two_person_household_change_is_per_capita():
    # Two-person household: £30,000 -> £28,000; one-person household: £20,000
    # (unchanged). map_to="person" broadcasts the household TOTAL per member.
    size = np.array([2.0, 2.0, 1.0])
    base = np.array([30_000.0, 30_000.0, 20_000.0])
    shocked = np.array([28_000.0, 28_000.0, 20_000.0])
    delta = per_capita_household_income(shocked, size) - per_capita_household_income(base, size)
    # Each member of the couple loses £1,000 per capita, NOT £2,000.
    np.testing.assert_allclose(delta, [-1_000.0, -1_000.0, 0.0])


def test_single_person_household_unchanged_by_division():
    np.testing.assert_allclose(
        per_capita_household_income(np.array([20_000.0]), np.array([1.0])),
        [20_000.0],
    )


@pytest.mark.parametrize("bad_size", [0.0, -1.0, np.nan, np.inf])
def test_invalid_household_size_raises(bad_size):
    with pytest.raises(ValueError):
        per_capita_household_income(np.array([30_000.0]), np.array([bad_size]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        per_capita_household_income(np.array([1.0, 2.0]), np.array([1.0]))
