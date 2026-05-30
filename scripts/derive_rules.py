"""CLI entry point: derive credit-policy rules from a small decision tree."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SETTINGS  # noqa: E402
from src.rules.derive import derive_rules, save_rules  # noqa: E402


def main() -> int:
    rules = derive_rules()
    out = save_rules(rules, SETTINGS.models_dir / "rules.json")
    print(f">> Saved {len(rules)} rules -> {out}\n")
    for r in rules:
        print("  " + r.as_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
