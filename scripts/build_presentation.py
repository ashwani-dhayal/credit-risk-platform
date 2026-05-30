"""Build the solution deck as both a PowerPoint and a PDF.

Outputs (committed under documents/):
  - documents/presentation.pptx   (editable source)
  - documents/presentation.pdf    (pinned in /documents/ per the brief)

The PDF is generated with ReportLab so the build is fully offline and works
inside Docker / CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
RULES_PATH = PROJECT_ROOT / "models" / "rules.json"


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def _load_rules() -> list[dict]:
    if RULES_PATH.exists():
        return json.loads(RULES_PATH.read_text())
    return []


def _slide_data() -> list[dict]:
    m = _load_metrics()
    rules = _load_rules()[:5]

    return [
        {
            "title": "AI-Powered Credit Risk Intelligence Platform",
            "subtitle": "NeoStats AI Engineer Assignment — Submission",
            "bullets": [
                "Dataset: Home Credit Default Risk (Kaggle)",
                "Stack: Python 3.11, LightGBM, SHAP, Streamlit, SQLite, Docker",
                "LLM: Multi-provider (OpenAI / Groq / Gemini) with deterministic fallback",
            ],
        },
        {
            "title": "Business Problem",
            "bullets": [
                "Banks must make faster, more accurate, and explainable credit decisions.",
                "Identify high-risk applicants early; automate risk scoring.",
                "Provide auditable, regulator-friendly reasons for every decision.",
                "Let business analysts explore the portfolio in plain English.",
                "Bridge ML insights and credit policy through readable rules.",
            ],
        },
        {
            "title": "Solution at a Glance",
            "bullets": [
                "End-to-end platform: EDA → ML → Explainability → Rules → Chatbot.",
                "Single Streamlit UI with 6 sections; one Docker command to run.",
                "ML model returns probability + Low/Medium/High band + Decision.",
                "SHAP TreeExplainer attributes every score to ranked feature drivers.",
                "Talk-to-Data agent translates English to safe, read-only SQLite SQL.",
                "Decision-tree rules give credit officers an audit-ready policy view.",
            ],
        },
        {
            "title": "Architecture",
            "bullets": [
                "Streamlit UI → src/ml (LightGBM + SHAP) and src/rules (decision tree).",
                "src/data ingests CSV → SQLite (single source of truth for ML & chatbot).",
                "src/llm: provider auto-detect (OpenAI → Groq → Gemini → fallback parser).",
                "src/utils/sql_safety: blocks DDL/DML, single-statement, LIMIT enforced.",
                "Dockerfile pre-trains the model at build time so the app boots ready.",
            ],
        },
        {
            "title": "Module 1 — EDA Insights",
            "bullets": [
                "EXT_SOURCE_* are the strongest single predictors; high scores cut default ~5×.",
                "Credit-to-income ratio matters more than absolute income.",
                "Higher-education / Academic-degree applicants default less.",
                "Younger applicants (≤30y) are riskier; flattens after ~45.",
                "Employment tenure protects; sentinel DAYS_EMPLOYED=365243 must be NaN.",
            ],
        },
        {
            "title": "Module 2 — Talk-to-Data Chatbot",
            "bullets": [
                "Pipeline: NL question → LLM → JSON {sql} → safety guard → run → summarise.",
                "Pinned schema + 4 few-shot examples + temperature=0 + max 256 output tokens.",
                "Static guardrails: SELECT-only, single statement, LIMIT cap, table allowlist.",
                "Read-only SQLite connection (mode=ro) for execution.",
                "Deterministic fallback parser handles 9 canonical questions offline.",
            ],
        },
        {
            "title": "Module 3 — Machine Learning Layer",
            "bullets": [
                f"Algorithm: LightGBM • scale_pos_weight handles 8% imbalance.",
                f"Validation: 80/20 stratified split.",
                f"ROC-AUC: {m.get('roc_auc', '—')} | KS: {m.get('ks_statistic', '—')} | "
                f"PR-AUC: {m.get('pr_auc', '—')}",
                f"Operating threshold: {m.get('optimal_threshold', '—')} (Youden's J).",
                "Risk bands: Low <0.20, Medium <0.50, High otherwise.",
                "Decision: Approve / Review / Reject (configurable in .env).",
            ],
        },
        {
            "title": "Module 4 — Explainable AI (SHAP)",
            "bullets": [
                "Per-applicant: top contributors with sign and direction (increases / decreases risk).",
                "Global: mean |SHAP| over a sample of 400 rows.",
                "Exact Shapley values via TreeExplainer — no approximation.",
                "EXT_SOURCE_MEAN, CREDIT_INCOME_RATIO, EXT_SOURCE_2 dominate the global ranking.",
            ],
        },
        {
            "title": "Module 5 — Decision Rules",
            "bullets": [
                "Depth-4 tree fit on the same engineered features as the model.",
                "Each leaf → IF-THEN rule with support, default-rate, and lift.",
                "Top rules in the bundled run:",
            ] + [
                (
                    f"R{r['rule_id']}: {r['band']} | support {r['support_pct']}% | "
                    f"default rate {r['default_rate_pct']}% | lift {r['lift']}×"
                )
                for r in rules
            ],
        },
        {
            "title": "Module 6 — Dockerised Deployment",
            "bullets": [
                "One command: docker compose up --build.",
                "Multi-stage Dockerfile (builder + slim runtime, non-root user).",
                "Pre-trains the model at build time; healthcheck on /_stcore/health.",
                "Volumes: data/raw is read-only mount; models/ persisted on host.",
                ".env.example documents every variable; auto-detect picks any LLM key.",
            ],
        },
        {
            "title": "Prompt Engineering & Hallucination Control",
            "bullets": [
                "JSON-only contract: model emits {\"sql\": \"...\"}; we parse deterministically.",
                "Pinned schema with column descriptions reduces guessing.",
                "Few-shot examples are short (4 cases) to keep input tokens low.",
                "Temperature 0; max 256 output tokens.",
                "Static SQL safety: regex + sqlparse validates before execution.",
                "Result-grounded summary prompt forbids invented numbers.",
                "Hard fallback path keeps the demo working when the LLM is unavailable.",
            ],
        },
        {
            "title": "Token Optimisation",
            "bullets": [
                "Schema (~25 columns) sent once per turn; no DB introspection round-trip.",
                "Compact column descriptions (1 line each) instead of verbose comments.",
                "Few-shot examples reuse the same column names so the model latches faster.",
                "max_tokens caps output; we expect <50 tokens for most SQL replies.",
                "Summarisation only sees the result rows we actually return (≤20).",
            ],
        },
        {
            "title": "Evaluation Results",
            "metrics_table": [
                ["Metric", "Value"],
                ["ROC-AUC", str(m.get("roc_auc", "—"))],
                ["PR-AUC", str(m.get("pr_auc", "—"))],
                ["KS Statistic", str(m.get("ks_statistic", "—"))],
                ["F1 @ optimal threshold", str(m.get("f1_at_optimal_threshold", "—"))],
                ["Optimal threshold", str(m.get("optimal_threshold", "—"))],
                ["Default rate", str(m.get("default_rate", "—"))],
                ["Train rows / Val rows", f"{m.get('n_train','—')} / {m.get('n_val','—')}"],
            ],
        },
        {
            "title": "How To Run",
            "bullets": [
                "git clone https://github.com/ashwani-dhayal/credit-risk-platform.git",
                "cd credit-risk-platform",
                "cp .env.example .env  # add ONE LLM key (optional)",
                "docker compose up --build",
                "Open http://localhost:8501",
                "Local mode: python -m venv .venv && pip install -r requirements.txt; "
                "python scripts/train_model.py; streamlit run app/streamlit_app.py",
            ],
        },
        {
            "title": "Limitations & Next Steps",
            "bullets": [
                "Only application_train modelled — adding bureau / previous_application "
                "feature aggregations should lift AUC by ~0.03 on real Kaggle data.",
                "No data-drift monitoring yet (Evidently / WhyLogs are obvious next steps).",
                "Single-tenant SQLite — switch to Postgres for multi-user.",
                "No UI auth — put behind a reverse proxy in production.",
                "LLM cost cap is per-call; daily quotas would need Redis-based rate limiting.",
            ],
        },
        {
            "title": "Thank you",
            "bullets": [
                "Repo: https://github.com/ashwani-dhayal/credit-risk-platform",
                "Author: Ashwani Dhayal",
                "Submission for NeoStats AI Engineer assignment.",
            ],
        },
    ]


# ----------------------------- PowerPoint ----------------------------------
def _build_pptx(slides: list[dict], path: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    title_layout = prs.slide_layouts[0]
    bullet_layout = prs.slide_layouts[1]
    blank_layout = prs.slide_layouts[5]

    primary = RGBColor(0x1F, 0x77, 0xB4)

    for i, s in enumerate(slides):
        if i == 0:
            slide = prs.slides.add_slide(title_layout)
            slide.shapes.title.text = s["title"]
            slide.placeholders[1].text = s.get("subtitle", "")
        elif "metrics_table" in s:
            slide = prs.slides.add_slide(blank_layout)
            slide.shapes.title.text = s["title"]
            rows = s["metrics_table"]
            tbl = slide.shapes.add_table(
                rows=len(rows), cols=2,
                left=Inches(2.0), top=Inches(1.6),
                width=Inches(9.3), height=Inches(0.4 * len(rows)),
            ).table
            for r_idx, row in enumerate(rows):
                for c_idx, val in enumerate(row):
                    cell = tbl.cell(r_idx, c_idx)
                    cell.text = str(val)
                    for para in cell.text_frame.paragraphs:
                        for run in para.runs:
                            run.font.size = Pt(16)
                            if r_idx == 0:
                                run.font.bold = True
        else:
            slide = prs.slides.add_slide(bullet_layout)
            slide.shapes.title.text = s["title"]
            tf = slide.placeholders[1].text_frame
            tf.text = s["bullets"][0]
            for b in s["bullets"][1:]:
                p = tf.add_paragraph()
                p.text = b
                p.level = 0
            for para in tf.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(18)

        # Title styling
        title_shape = slide.shapes.title
        if title_shape and title_shape.text_frame:
            for para in title_shape.text_frame.paragraphs:
                for run in para.runs:
                    run.font.color.rgb = primary
                    run.font.bold = True

    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)
    return path


# ------------------------------- PDF (ReportLab) ----------------------------
def _build_pdf(slides: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="Credit Risk Intelligence Platform",
        author="Ashwani Dhayal",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=24, leading=28,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=14,
    )
    sub = ParagraphStyle(
        "Sub", parent=styles["Heading3"],
        fontSize=14, leading=18,
        textColor=colors.HexColor("#555555"),
        spaceAfter=12,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"],
        fontSize=12, leading=18, spaceAfter=6,
    )

    flow: list = []
    for i, s in enumerate(slides):
        flow.append(Paragraph(s["title"], h1))
        if "subtitle" in s and s["subtitle"]:
            flow.append(Paragraph(s["subtitle"], sub))

        if "metrics_table" in s:
            tbl = Table(s["metrics_table"], colWidths=[8 * cm, 8 * cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f8fa")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]))
            flow.append(tbl)
        else:
            for b in s.get("bullets", []):
                flow.append(Paragraph(f"• {b}", body))

        if i != len(slides) - 1:
            flow.append(PageBreak())

    doc.build(flow)
    return path


def main() -> int:
    slides = _slide_data()
    out_dir = PROJECT_ROOT / "documents"
    pptx_path = _build_pptx(slides, out_dir / "presentation.pptx")
    pdf_path = _build_pdf(slides, out_dir / "presentation.pdf")
    print(f">> PPTX -> {pptx_path}")
    print(f">> PDF  -> {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
