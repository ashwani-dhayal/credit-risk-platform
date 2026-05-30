"""CLI entry point: train the model and persist artifacts."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.train import train  # noqa: E402


def main() -> int:
    print(">> Training LightGBM credit-risk model …")
    result = train()
    print(json.dumps(asdict(result), indent=2)[:1500])
    print(f"\n>> Model saved to {result.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
