"""CLI: generate the bundled synthetic sample CSV."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sample_generator import write_sample  # noqa: E402

if __name__ == "__main__":
    out = PROJECT_ROOT / "data" / "sample" / "application_train_sample.csv"
    p = write_sample(out, n_rows=10_000)
    print(f"Wrote sample CSV -> {p}")
