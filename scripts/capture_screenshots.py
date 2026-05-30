"""Capture full-page screenshots of every Streamlit section.

Prereq: Streamlit must already be running on http://localhost:8501.
Usage:
    python scripts/capture_screenshots.py

Outputs: documents/screenshots/01_overview.png, 02_eda.png, ...
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import Page, sync_playwright

URL = "http://localhost:8501"
OUT_DIR = PROJECT_ROOT / "documents" / "screenshots"

# (filename, sidebar radio label, optional pre-screenshot interaction)
SECTIONS: list[tuple[str, str, str | None]] = [
    ("01_overview.png", "🏠 Overview", None),
    ("02_eda.png", "📊 EDA", None),
    ("03_predict.png", "🎯 Risk Prediction", "predict"),
    ("04_explain.png", "🔍 Explainability", None),
    ("05_rules.png", "📜 Decision Rules", None),
    ("06_chatbot.png", "💬 Talk-to-Data", "chatbot"),
]


def _click_sidebar(page: Page, label: str) -> None:
    """Click a sidebar radio option by visible label."""
    page.get_by_text(label, exact=True).first.click()
    page.wait_for_timeout(2500)  # let charts render


def _interact_predict(page: Page) -> None:
    """Click the Predict button so the result + form are both visible."""
    btn = page.get_by_role("button", name="🚀 Predict")
    if btn.count() > 0:
        btn.first.click()
        page.wait_for_timeout(2500)


def _interact_chatbot(page: Page) -> None:
    """Click the first sample question + Ask so the chatbot output renders."""
    sample = page.get_by_role("button", name="How many applicants are there in total?")
    if sample.count() > 0:
        sample.first.click()
        page.wait_for_timeout(800)
    ask = page.get_by_role("button", name="🔎 Ask")
    if ask.count() > 0:
        ask.first.click()
        page.wait_for_timeout(2500)


INTERACTIONS = {
    "predict": _interact_predict,
    "chatbot": _interact_chatbot,
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1480, "height": 1100},
            device_scale_factor=2,  # retina-quality screenshots
        )
        page = context.new_page()
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(3500)

        for filename, label, action_key in SECTIONS:
            print(f">> Capturing {filename} ({label}) …")
            try:
                _click_sidebar(page, label)
            except Exception as e:
                print(f"   sidebar click failed for {label}: {e}")

            if action_key and action_key in INTERACTIONS:
                try:
                    INTERACTIONS[action_key](page)
                except Exception as e:
                    print(f"   interaction failed for {label}: {e}")

            page.wait_for_timeout(1500)
            out = OUT_DIR / filename
            page.screenshot(path=str(out), full_page=True)
            print(f"   saved {out}")

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
