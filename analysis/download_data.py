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

#: PIN THE DATASET REVISION. The HF repo republishes frs_2024_25.h5 in place
#: under the same filename, so downloading the default branch does NOT give
#: the build the paper's results were computed on. As of 11 Sept 2026 the
#: default resolved to a 4 Sept rebuild whose employee base is 27.99m weighted
#: against 22.2m in the paper's build, moving the central Exchequer result
#: from GBP 18.2bn to GBP 22.1bn with no code change and no warning.
#:
#: This revision's frs_2024_25.h5 has sha256
#: 623802aad0f7eef76df8a0ce1b19933b5648ac97486d5b01d371e7c7d374ba40, matching
#: results/BUILD_MANIFEST.json. Bump it deliberately, and rerun the full
#: results tree when you do.
REVISION = "5535b2f805"


def main() -> None:
    token = os.environ["HUGGING_FACE_TOKEN"]
    data = Path("data")
    data.mkdir(exist_ok=True)
    for name in FILES:
        path = hf_hub_download(
            REPO, name, revision=REVISION, token=token, local_dir=data
        )
        print(path)
    with zipfile.ZipFile(data / "frs_2024_25.zip") as zf:
        zf.extractall(data / "frs_2024_25")
    print("extracted adult.tab:", (data / "frs_2024_25" / "UKDA-9563-tab" / "tab" / "adult.tab").exists())


if __name__ == "__main__":
    main()
