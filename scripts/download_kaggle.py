"""Download the Home Credit Default Risk dataset from Kaggle.

Prereq:
  Place your kaggle.json API token at:
    Linux/Mac: ~/.kaggle/kaggle.json
    Windows:   %USERPROFILE%\\.kaggle\\kaggle.json
  Get one at https://www.kaggle.com/settings -> "Create New Token".

Usage:
  python scripts/download_kaggle.py
  python scripts/download_kaggle.py --files application_train.csv

The competition rules require accepting them on Kaggle once before the API
will let you download. If you see a 403, visit
https://www.kaggle.com/competitions/home-credit-default-risk/rules and click
"I Understand and Accept".
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
COMPETITION = "home-credit-default-risk"


def _check_credentials() -> None:
    home = Path.home()
    candidate = home / ".kaggle" / "kaggle.json"
    if not candidate.exists():
        print(
            f"!! kaggle.json not found at {candidate}\n"
            "   1. Go to https://www.kaggle.com/settings\n"
            "   2. Click 'Create New Token' (saves kaggle.json)\n"
            f"   3. Move it to:  {candidate}\n"
            "   4. Re-run this script."
        )
        sys.exit(2)
    # Tighten perms (Kaggle CLI complains otherwise on Linux/Mac)
    try:
        os.chmod(candidate, 0o600)
    except Exception:
        pass


def _download(files: list[str] | None) -> None:
    # Lazy import so users without the package get a friendly error.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print(f"!! Could not import kaggle: {e}")
        print("   Run: pip install kaggle")
        sys.exit(3)

    api = KaggleApi()
    api.authenticate()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if files:
        for f in files:
            print(f">> Downloading {f} from competition '{COMPETITION}' …")
            api.competition_download_file(
                COMPETITION, file_name=f, path=str(RAW_DIR), force=True
            )
    else:
        print(f">> Downloading ALL files from competition '{COMPETITION}' …")
        api.competition_download_files(
            COMPETITION, path=str(RAW_DIR), force=True, quiet=False
        )

    # Unzip anything that arrived as .zip (single-file or whole-archive).
    for zp in RAW_DIR.glob("*.zip"):
        print(f">> Extracting {zp.name} …")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(RAW_DIR)
        zp.unlink()

    # Summarise what we have.
    print("\n>> Files now in data/raw/:")
    for f in sorted(RAW_DIR.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1_000_000
            print(f"   {f.name:40s} {size_mb:8.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Specific files to fetch (e.g. --files application_train.csv). "
             "If omitted, downloads the entire competition zip.",
    )
    args = parser.parse_args()
    _check_credentials()
    _download(args.files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
