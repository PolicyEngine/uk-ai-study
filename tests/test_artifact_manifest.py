from pathlib import Path

from analysis.artifact_manifest import REQUIRED, RESULTS, sha256


def test_required_artifact_paths_are_unique():
    assert len(REQUIRED) == len(set(REQUIRED))
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in REQUIRED)


def test_sha256_is_stable(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_bytes(b"uk-ai-study\n")
    assert sha256(path) == "890974d53a21c889a3b0ea9d8e3e59953d0ee5012499c9709ace028d217ca568"
