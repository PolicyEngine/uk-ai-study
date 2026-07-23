import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from policy_counterfactuals import household_decile_benefit_shares


def test_household_benefit_shares_do_not_broadcast_by_household_size():
    # Equal £100 gains in a one-person and five-person household with equal
    # household weights must split pounds 50/50, not 1/6 versus 5/6.
    shares = household_decile_benefit_shares(
        household_gain=np.array([100.0, 100.0]),
        household_weight=np.array([1.0, 1.0]),
        household_decile=np.array([1, 10]),
    )
    assert shares[1] == pytest.approx(50.0)
    assert shares[10] == pytest.approx(50.0)
