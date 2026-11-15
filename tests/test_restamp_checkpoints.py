"""The restamp verifier accepts only non-behavioural edits (review of PR #18, M7/M8)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import restamp_checkpoints as rs  # noqa: E402


def _tree(tmp_path, name, files):
    root = tmp_path / name
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return root


def _cmp(tmp_path, old_files, new_files, rel="analysis/stage.py"):
    return rs.compare(rel, _tree(tmp_path, "old", old_files), _tree(tmp_path, "new", new_files))


def test_constant_moved_to_package_import_is_equivalent(tmp_path):
    old = {"analysis/stage.py": "PERIOD = 2026\ndef f(sim):\n    return sim.calc(period=PERIOD)\n",
           "uk_ai_study/data.py": "PERIOD = 2026\n"}
    new = {"analysis/stage.py": "from uk_ai_study.data import PERIOD\n"
                                "def f(sim):\n    return sim.calc(period=PERIOD)\n",
           "uk_ai_study/data.py": "PERIOD = 2026\n"}
    assert _cmp(tmp_path, old, new) == "equivalent"


def test_imported_constant_with_another_value_is_a_change(tmp_path):
    old = {"analysis/stage.py": "PERIOD = 2026\nx = PERIOD\n", "uk_ai_study/data.py": "PERIOD = 2026\n"}
    new = {"analysis/stage.py": "from uk_ai_study.data import PERIOD\nx = PERIOD\n",
           "uk_ai_study/data.py": "PERIOD = 2027\n"}
    with pytest.raises(SystemExit, match="behavioural change"):
        _cmp(tmp_path, old, new)


def test_aliased_import_and_docstring_edits_are_equivalent(tmp_path):
    old = {"analysis/stage.py": '"""Not yet run."""\nCT_RATE = 0.25\ndef g(x):\n    """Old."""\n    return CT_RATE * x\n',
           "uk_ai_study/shocks.py": "CORPORATION_TAX_RATE = 0.25\n"}
    new = {"analysis/stage.py": '"""Run 13 Nov."""\nfrom uk_ai_study.shocks import CORPORATION_TAX_RATE as CT_RATE\n'
                                'def g(x):\n    """New."""\n    return CT_RATE * x\n',
           "uk_ai_study/shocks.py": "CORPORATION_TAX_RATE = 0.25\n"}
    assert _cmp(tmp_path, old, new) == "equivalent"


def test_arithmetic_change_is_caught(tmp_path):
    old = {"analysis/stage.py": "def g(x):\n    return 0.25 * x\n"}
    new = {"analysis/stage.py": "def g(x):\n    return 0.25 * x + 1\n"}
    with pytest.raises(SystemExit, match="behavioural change"):
        _cmp(tmp_path, old, new)


def test_unlisted_rename_is_caught(tmp_path):
    old = {"analysis/stage.py": "def g(a):\n    elite = a > 1\n    return elite\n"}
    new = {"analysis/stage.py": "def g(a):\n    top = a > 1\n    return top\n"}
    with pytest.raises(SystemExit, match="behavioural change"):
        _cmp(tmp_path, old, new)


def test_restamp_text_touches_only_the_stamp_column():
    text = "seed,cost,certified_data_build_id,code_fingerprint\n0,1.25,b,cOLD\n1,2.5,b,cOLD\n"
    out = rs.restamp_text(text, "cOLD", "cNEW", rows=2)
    assert out == text.replace("cOLD", "cNEW")
    with pytest.raises(SystemExit):
        rs.restamp_text(text.replace("1,2.5,b,cOLD", "1,2.5,b,cOTHER"), "cOLD", "cNEW", rows=2)
