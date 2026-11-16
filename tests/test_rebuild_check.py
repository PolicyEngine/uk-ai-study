"""rebuild.py --check compares presentation hashes and covers every stage
(review of PR #18, finding H1)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import rebuild  # noqa: E402


def test_every_tracked_result_has_a_stage():
    outputs = set(rebuild.all_outputs(rebuild.MANIFEST))
    known = outputs | set(rebuild.NON_GENERATED) | {rebuild.ATTESTATION, rebuild.BUILD_LOG}
    assert set(rebuild.tracked_results_files()) <= known
    for d in ("next_runs/", "takeup/", "wage_cut/"):
        assert any(o.startswith(d) for o in outputs), d


def test_tex_values_runs_strict_after_every_results_stage():
    by = {s["stage"]: s for s in rebuild.MANIFEST}
    assert "--strict" in by["tex_values"]["cmd"]
    producers = {s["stage"] for s in rebuild.MANIFEST if s["outputs"]}
    assert producers <= set(by["tex_values"]["after"])


def test_check_flags_a_presentation_file_that_differs_from_the_attestation(tmp_path, monkeypatch, capsys):
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.csv").write_text("x\n1\n")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "values_generated.tex").write_text("% macros\n")
    manifest = [{"stage": "s", "cmd": [], "outputs": ["a.csv"], "inputs": [], "after": []},
                {"stage": "tex", "cmd": [], "outputs": [], "inputs": [], "after": ["s"],
                 "repo_outputs": ["paper/values_generated.tex"]}]
    att = {"files": {"a.csv": rebuild.sha256(results / "a.csv")}, "inputs": {},
           "presentation_files": {"paper/values_generated.tex": rebuild.sha256(paper / "values_generated.tex")},
           "outside_clean_room": {"s": "run separately"}}
    (results / rebuild.ATTESTATION).write_text(json.dumps(att))
    monkeypatch.setattr(rebuild, "ROOT", tmp_path)
    monkeypatch.setattr(rebuild, "tracked_results_files", lambda: ["a.csv"])
    assert rebuild.check(manifest, results_dir=results) == 0
    assert "did not come out of a clean-room build" in capsys.readouterr().out
    (paper / "values_generated.tex").write_text("% edited after attestation\n")
    assert rebuild.check(manifest, results_dir=results) == 1
    assert "STALE     paper/values_generated.tex" in capsys.readouterr().out
