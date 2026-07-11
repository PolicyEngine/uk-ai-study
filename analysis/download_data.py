"""Download the FRS microdata from PolicyEngine's Hugging Face repo.

Requires HUGGING_FACE_TOKEN in the environment (token with access to
policyengine/policyengine-uk-data). Files land in data/ (gitignored).
"""

import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "policyengine/policyengine-uk-data"
FILES = ("frs_2024_25.h5", "frs_2024_25.zip")


def main() -> None:
    token = os.environ["HUGGING_FACE_TOKEN"]
    data = Path("data")
    data.mkdir(exist_ok=True)
    for name in FILES:
        path = hf_hub_download(REPO, name, token=token, local_dir=data)
        print(path)
    with zipfile.ZipFile(data / "frs_2024_25.zip") as zf:
        zf.extractall(data / "frs_2024_25")
    print("extracted adult.tab:", (data / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab").exists())


if __name__ == "__main__":
    main()
