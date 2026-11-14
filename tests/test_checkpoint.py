"""Stamped, opt-in checkpoints (review of PR #18, finding 1)."""

import pandas as pd
import pytest

from uk_ai_study.checkpoint import (
    Checkpoint,
    StaleCheckpointError,
    certified_data_build_id,
    code_fingerprint,
)

STAMP = {"certified_data_build_id": "build-a", "code_fingerprint": "c0123"}


def test_build_id_comes_from_the_bundle_manifest():
    assert certified_data_build_id().startswith("microcosm-uk-2024-25")


def test_fingerprint_is_not_numeric_and_tracks_the_script(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("x = 2\n")
    fa = code_fingerprint(a)
    assert fa.startswith("c") and fa != code_fingerprint(b)
    assert fa == code_fingerprint(a)


def test_rows_carry_the_stamp_and_resume_skips_done_units(tmp_path):
    path = tmp_path / "draws.csv"
    ck = Checkpoint(path, ("seed",), resume=False, stamp=STAMP)
    ck.append({"seed": 0, "cost": 1.0})
    ck.append({"seed": 1, "cost": 2.0})
    d = pd.read_csv(path)
    assert set(d["certified_data_build_id"]) == {"build-a"}
    again = Checkpoint(path, ("seed",), resume=True, stamp=STAMP)
    assert again.is_done(0) and again.is_done(1) and not again.is_done(2)


def test_without_resume_an_existing_checkpoint_is_discarded(tmp_path):
    path = tmp_path / "draws.csv"
    Checkpoint(path, ("seed",), resume=False, stamp=STAMP).append({"seed": 0})
    fresh = Checkpoint(path, ("seed",), resume=False, stamp=STAMP)
    assert fresh.done == set() and not path.exists()


def test_resume_refuses_another_data_build(tmp_path):
    path = tmp_path / "draws.csv"
    Checkpoint(path, ("seed",), resume=False, stamp=STAMP).append({"seed": 0})
    other = {**STAMP, "certified_data_build_id": "build-b"}
    with pytest.raises(StaleCheckpointError, match="build-a"):
        Checkpoint(path, ("seed",), resume=True, stamp=other)


def test_resume_refuses_other_code(tmp_path):
    path = tmp_path / "draws.csv"
    Checkpoint(path, ("seed",), resume=False, stamp=STAMP).append({"seed": 0})
    with pytest.raises(StaleCheckpointError):
        Checkpoint(path, ("seed",), resume=True, stamp={**STAMP, "code_fingerprint": "c9"})


def test_resume_refuses_an_unstamped_legacy_file(tmp_path):
    # the FRS-era checkpoints had no stamp columns at all
    path = tmp_path / "draws.csv"
    pd.DataFrame({"seed": [0, 1], "cost": [18.2, 18.3]}).to_csv(path, index=False)
    with pytest.raises(StaleCheckpointError, match="no certified_data_build_id"):
        Checkpoint(path, ("seed",), resume=True, stamp=STAMP)


def test_resume_matches_grid_and_variant_keys(tmp_path):
    # the appendix grid key is (int, int, bool); the sensitivities use strings
    path = tmp_path / "grid.csv"
    ck = Checkpoint(path, ("unemployment_pct", "wage_pct", "capital_shock"), resume=False, stamp=STAMP)
    ck.append({"unemployment_pct": 3, "wage_pct": 0, "capital_shock": True, "x": 1.0})
    ck.append({"unemployment_pct": 3, "wage_pct": 0, "capital_shock": False, "x": 2.0})
    again = Checkpoint(path, ("unemployment_pct", "wage_pct", "capital_shock"), resume=True, stamp=STAMP)
    assert again.is_done(3, 0, True) and again.is_done(3, 0, False) and not again.is_done(3, 1, True)

    path = tmp_path / "tier.csv"
    ck = Checkpoint(path, ("variant", "seed"), resume=False, stamp=STAMP)
    ck.append({"variant": "3_0.8", "seed": 0})
    ck.append({"variant": "9.6_0.6", "seed": 0})
    again = Checkpoint(path, ("variant", "seed"), resume=True, stamp=STAMP)
    assert again.is_done("3_0.8", 0) and again.is_done("9.6_0.6", 0) and not again.is_done("6_0.7", 0)
