"""Fetch the certified Microcosm UK 2024-25 dataset into data/.

The dataset is the one pinned by the policyengine.py release bundle
(``policyengine[uk]`` in pyproject.toml). It lives in a private Hugging Face
repository (policyengine/populace-uk-private), so a token with access must be
available as HUGGING_FACE_TOKEN or through a saved ``huggingface-cli login``.
The file lands in data/ (gitignored) and its sha256 is checked against the
bundle; every analysis script then resolves it through
``uk_ai_study.data.microsimulation()``. Running this first is optional (the
first simulation fetches the file too); it just front-loads the download.
"""

from policyengine.provenance.dataset_materialization import materialize_dataset

from uk_ai_study.data import _ensure_token


def main() -> None:
    _ensure_token()
    source = materialize_dataset("uk")
    print(source.path)
    print(source.source_uri)


if __name__ == "__main__":
    main()
