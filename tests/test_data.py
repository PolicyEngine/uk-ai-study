"""Microcosm occupation coding: soc2020_major_group -> exposure-table keys."""

import numpy as np
import pytest

from uk_ai_study.data import soc_major_group
from uk_ai_study.exposure import exposure_for_major_group, load_major_group_exposure


class _Values:
    def __init__(self, values):
        self.values = np.asarray(values)


class _FakeSim:
    def __init__(self, values):
        self._values = values

    def calculate(self, variable, period=None, map_to=None):
        assert variable == "soc2020_major_group"
        return _Values(self._values)


def test_enum_names_map_to_soc_codes():
    sim = _FakeSim(["PROFESSIONAL", "NONE", "ELEMENTARY", "MANAGERS_DIRECTORS_AND_SENIOR_OFFICIALS"])
    codes = soc_major_group(sim, 2026)
    assert codes[0] == 2 and np.isnan(codes[1]) and codes[2] == 9 and codes[3] == 1


def test_enum_indices_map_to_soc_codes():
    # enum index 0 is NONE; 1..9 follow SOC order
    codes = soc_major_group(_FakeSim(np.array([0, 1, 5, 9])), 2026)
    assert np.isnan(codes[0])
    np.testing.assert_array_equal(codes[1:], [1, 5, 9])


def test_unknown_member_raises():
    with pytest.raises(ValueError):
        soc_major_group(_FakeSim(["NOT_A_GROUP"]), 2026)


def test_codes_key_the_exposure_table():
    table = load_major_group_exposure()
    codes = soc_major_group(_FakeSim([m for m in ["PROFESSIONAL", "ELEMENTARY", "NONE"]]), 2026)
    exposure = exposure_for_major_group(codes, "c_aioe")
    assert exposure[0] == table.loc[2, "c_aioe"]
    assert exposure[1] == table.loc[9, "c_aioe"]
    assert np.isnan(exposure[2])
