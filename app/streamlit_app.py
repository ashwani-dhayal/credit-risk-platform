"""Streamlit multi-section UI for the Credit Risk Intelligence Platform.

Sections (left sidebar):
  1. Overview
  2. EDA
  3. Risk Prediction
  4. Explainability
  5. Decision Rules
  6. Talk-to-Data Chatbot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src` importable when running with `streamlit run app/streamlit_app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import SETTINGS  # noqa: E402
from src.data.loader import ingest_to_sqlite, load_dataframe  # noqa: E402
from src.data.preprocess import add_engineered_features  # noqa: E402
from src.data.schema import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    COLUMN_DESCRIPTIONS,
    UI_INPUT_COLUMNS,
)

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------- Cached data loaders ----------------------------
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    ingest_to_sqlite(force=False)
    df = load_dataframe()
    return add_engineered_features(df)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict | None:
    p = SETTINGS.models_dir / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


@st.cache_data(show_spinner=False)
def get_rules() -> list[dict]:
    p = SETTINGS.models_dir / "rules.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


# ----------------------------- Sidebar nav ----------------------------
with st.sidebar:
    st.title("💳 Credit Risk")
    st.caption("AI-Powered Credit Risk Intelligence Platform")
    section = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 EDA",
            "🎯 Risk Prediction",
            "🔍 Explainability",
            "📜 Decision Rules",
            "💬 Talk-to-Data",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**LLM provider**")
    st.code(SETTINGS.active_llm_provider, language="text")
    st.markdown("**Model artifact**")
    if SETTINGS.model_path.exists():
        st.success(f"loaded: {SETTINGS.model_path.name}")
    else:
        st.warning("Not trained yet — run\n`python scripts/train_model.py`")


# =========================================================================
# Section 1: Overview
# =========================================================================
def render_overview():
    st.title("AI-Powered Credit Risk Intelligence Platform")
    st.markdown(
        "An end-to-end platform that combines **machine learning, "
        "explainable AI, and a natural-language SQL agent** to support "
        "credit-risk decisions. Built on the Home Credit Default Risk dataset."
    )

    df = get_data()
    metrics = get_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(df):,}")
    c2.metric("Default Rate", f"{df['TARGET'].mean()*100:.2f}%")
    c3.metric("Features", f"{df.shape[1]-2}")
    if metrics:
        c4.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")
    else:
        c4.metric("Model ROC-AUC", "—")

    st.divider()
    st.subheader("How to use this app")
    st.markdown(
        """
        1. **EDA** — explore distributions, default rates, demographics.
        2. **Risk Prediction** — score a single applicant using the trained model.
        3. **Explainability** — see SHAP-driven reasons for each prediction.
        4. **Decision Rules** — readable if-then rules derived from a tree.
        5. **Talk-to-Data** — ask plain-English questions about the data.
        """
    )


# =========================================================================
# Section 2: EDA
# =========================================================================
def render_eda():
    st.title("📊 Exploratory Data Analysis")
    df = get_data()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary", "Demographics", "Financials", "Default Drivers"]
    )

    with tab1:
        st.subheader("Dataset summary")
        c1, c2 = st.columns(2)
        c1.write(f"**Rows:** {len(df):,}")
        c1.write(f"**Columns:** {df.shape[1]}")
        c1.write(f"**Default rate:** {df['TARGET'].mean()*100:.2f}%")
        c2.write("**Missing values (top 10):**")
        missing = (
            df.isna().sum().sort_values(ascending=False).head(10).reset_index()
        )
        missing.columns = ["column", "missing_count"]
        missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
        c2.dataframe(missing, use_container_width=True, hide_index=True)

        st.markdown("**Sample rows**")
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        st.subheader("Demographics")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df, x="AGE_YEARS", color="TARGET", nbins=30,
                barmode="overlay", opacity=0.6,
                title="Age distribution by default status",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            gender_def = (
                df.groupby("CODE_GENDER")["TARGET"].mean().reset_index()
            )
            gender_def["default_rate_pct"] = gender_def["TARGET"] * 100
            fig = px.bar(
                gender_def, x="CODE_GENDER", y="default_rate_pct",
                title="Default rate by gender", text_auto=".2f",
            )
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            edu = (
                df.groupby("NAME_EDUCATION_TYPE")
                .agg(default_rate=("TARGET", "mean"), n=("TARGET", "size"))
                .reset_index()
                .sort_values("default_rate", ascending=False)
            )
            edu["default_rate_pct"] = edu["default_rate"] * 100
            fig = px.bar(
                edu, x="default_rate_pct", y="NAME_EDUCATION_TYPE",
                orientation="h",
                title="Default rate by education", text_auto=".2f",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            inc = (
                df.groupby("NAME_INCOME_TYPE")
                .agg(default_rate=("TARGET", "mean"), n=("TARGET", "size"))
                .reset_index()
                .sort_values("default_rate", ascending=False)
            )
            inc["default_rate_pct"] = inc["default_rate"] * 100
            fig = px.bar(
                inc, x="default_rate_pct", y="NAME_INCOME_TYPE",
                orientation="h",
                title="Default rate by income type", text_auto=".2f",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Financials")
        # Trim extreme tails for visualisation only.
        view = df[df["AMT_INCOME_TOTAL"] < df["AMT_INCOME_TOTAL"].quantile(0.99)]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                view, x="AMT_INCOME_TOTAL", color="TARGET", nbins=40,
                barmode="overlay", opacity=0.6,
                title="Income distribution (≤99th pct) by default status",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(
                view, x="CREDIT_INCOME_RATIO", color="TARGET", nbins=40,
                barmode="overlay", opacity=0.6,
                title="Credit / income ratio by default status",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Median financials by default status**")
        agg = (
            df.groupby("TARGET")[
                ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO"]
            ]
            .median()
            .round(2)
        )
        agg.index = ["Repaid (0)", "Defaulted (1)"]
        st.dataframe(agg, use_container_width=True)

    with tab4:
        st.subheader("Top numeric correlations with TARGET")
        num = df.select_dtypes("number").drop(columns=["SK_ID_CURR"], errors="ignore")
        corr = num.corr(numeric_only=True)["TARGET"].drop("TARGET")
        top = corr.abs().sort_values(ascending=False).head(15)
        top_signed = corr.loc[top.index].round(4)
        df_corr = top_signed.reset_index()
        df_corr.columns = ["feature", "pearson_corr_with_TARGET"]
        fig = px.bar(
            df_corr.sort_values("pearson_corr_with_TARGET"),
            x="pearson_corr_with_TARGET", y="feature", orientation="h",
            title="Correlation with default (signed)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Five business insights")
    st.markdown(
        """
        1. **External credit scores dominate**: `EXT_SOURCE_*` columns
           consistently rank as the strongest predictors and are negatively
           correlated with default — clients with higher external scores
           default far less.
        2. **Credit-to-income ratio matters more than absolute income**:
           Defaulters show meaningfully higher loan-to-income ratios.
        3. **Education is a strong demographic signal**: Higher-education and
           Academic-degree clients default at noticeably lower rates than
           Lower-secondary.
        4. **Age skew**: Younger applicants (≤ 30y) carry higher default
           probability; the curve flattens after ~45 years.
        5. **Employment stability**: Longer employment tenure (years) is
           protective; the unemployed sentinel `DAYS_EMPLOYED = 365243`
           must be treated as missing or it dominates the signal.
        """
    )


# =========================================================================
# Section 3: Risk Prediction
# =========================================================================
def render_predict():
    st.title("🎯 Risk Prediction")
    st.markdown(
        "Enter applicant details below and click **Predict** to get a "
        "probability of default, a risk band, and a suggested decision."
    )

    if not SETTINGS.model_path.exists():
        st.error(
            "No trained model found.\n\n"
            "Run `python scripts/train_model.py` first, then refresh this page."
        )
        return

    from src.ml.predict import predict_one  # local import to avoid cold start

    df = get_data()

    def _options(col: str) -> list[str]:
        return sorted([str(x) for x in df[col].dropna().unique().tolist()])

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            contract = st.selectbox(
                "Contract type", _options("NAME_CONTRACT_TYPE"), index=0
            )
            gender = st.selectbox(
                "Gender", _options("CODE_GENDER"), index=0
            )
            own_car = st.selectbox("Owns car", ["Y", "N"], index=1)
            own_realty = st.selectbox("Owns realty", ["Y", "N"], index=0)
            children = st.number_input("Children", 0, 10, 0)
            fam_members = st.number_input("Family members", 1, 12, 2)

        with c2:
            income = st.number_input("Annual income", 25_000, 5_000_000, 180_000, step=5_000)
            credit = st.number_input("Loan amount", 45_000, 4_000_000, 600_000, step=10_000)
            annuity = st.number_input("Annuity (yearly)", 5_000, 400_000, 30_000, step=1_000)
            goods = st.number_input("Goods price", 45_000, 4_000_000, 540_000, step=10_000)
            region = st.selectbox("Region rating (1=best, 3=worst)", [1, 2, 3], index=1)

        with c3:
            income_type = st.selectbox("Income type", _options("NAME_INCOME_TYPE"))
            education = st.selectbox("Education", _options("NAME_EDUCATION_TYPE"))
            family_status = st.selectbox("Family status", _options("NAME_FAMILY_STATUS"))
            housing = st.selectbox("Housing", _options("NAME_HOUSING_TYPE"))
            occupation_opts = ["(unknown)"] + _options("OCCUPATION_TYPE")
            occupation = st.selectbox("Occupation", occupation_opts, index=0)
            age_years = st.number_input("Age (years)", 21, 70, 35)
            emp_years = st.number_input("Years employed", 0.0, 45.0, 5.0, step=0.5)

        st.markdown("**External credit scores (0-1, blank = unknown)**")
        ec1, ec2, ec3 = st.columns(3)
        ext1 = ec1.text_input("EXT_SOURCE_1", value="")
        ext2 = ec2.text_input("EXT_SOURCE_2", value="0.55")
        ext3 = ec3.text_input("EXT_SOURCE_3", value="0.50")

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if not submitted:
        return

    def _f(s: str):
        s = s.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    applicant = {
        "NAME_CONTRACT_TYPE": contract,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_BIRTH": -int(age_years * 365.25),
        "DAYS_EMPLOYED": -int(emp_years * 365.25) if emp_years > 0 else 365243,
        "OCCUPATION_TYPE": None if occupation == "(unknown)" else occupation,
        "CNT_FAM_MEMBERS": fam_members,
        "REGION_RATING_CLIENT": region,
        "EXT_SOURCE_1": _f(ext1),
        "EXT_SOURCE_2": _f(ext2),
        "EXT_SOURCE_3": _f(ext3),
    }

    pred = predict_one(applicant)

    band_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[pred.risk_band]
    decision_emoji = {"Approve": "✅", "Review": "⚠️", "Reject": "🛑"}[pred.decision]

    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of default", f"{pred.probability_default*100:.2f}%")
    c2.metric("Risk band", f"{band_color} {pred.risk_band}")
    c3.metric("Decision", f"{decision_emoji} {pred.decision}")

    st.progress(min(pred.probability_default, 1.0))

    # Stash for the explainability tab
    st.session_state["last_applicant"] = applicant
    st.session_state["last_prediction"] = pred.__dict__

    with st.expander("Applicant payload sent to the model"):
        st.json(applicant)


# =========================================================================
# Section 4: Explainability
# =========================================================================
def render_explain():
    st.title("🔍 Explainability (SHAP)")

    if not SETTINGS.model_path.exists():
        st.error("Train the model first: `python scripts/train_model.py`")
        return

    from src.ml.explain import explain_one, global_importance

    tab1, tab2 = st.tabs(["Per-applicant explanation", "Global drivers"])

    with tab1:
        applicant = st.session_state.get("last_applicant")
        if not applicant:
            st.info("Score an applicant in the **Risk Prediction** section first.")
            return
        contributions = explain_one(applicant, top_n=10)
        df = pd.DataFrame([c.__dict__ for c in contributions])
        df = df.sort_values("shap_value", key=lambda s: s.abs(), ascending=True)
        fig = px.bar(
            df, x="shap_value", y="feature", color="direction",
            orientation="h",
            title="Top contributors to this prediction (SHAP values)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[["feature", "value", "shap_value", "direction"]],
                     use_container_width=True, hide_index=True)

    with tab2:
        with st.spinner("Computing SHAP across a sample…"):
            gi = global_importance(sample_size=400).head(15)
        fig = px.bar(
            gi.sort_values("mean_abs_shap"),
            x="mean_abs_shap", y="feature", orientation="h",
            title="Top 15 global drivers (mean |SHAP|)",
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section 5: Decision Rules
# =========================================================================
def render_rules():
    st.title("📜 Business-Readable Decision Rules")
    st.markdown(
        "Rules are derived from a depth-limited decision tree fit on the same "
        "engineered features as the LightGBM model. Each row is a leaf — the "
        "table shows its support, default rate, and lift vs. the base rate."
    )
    rules = get_rules()
    if not rules:
        st.warning(
            "No rules artifact found.\n"
            "Run `python scripts/derive_rules.py` after training."
        )
        return

    df = pd.DataFrame(rules)
    df["conditions"] = df["conditions"].apply(lambda c: " AND ".join(c) if c else "—")
    show = df[["rule_id", "band", "conditions", "support_pct",
               "default_rate_pct", "lift", "n_samples"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Rule cards")
    for _, r in df.iterrows():
        color = {"Low": "green", "Medium": "orange", "High": "red"}[r["band"]]
        with st.container(border=True):
            st.markdown(
                f"**Rule {r['rule_id']}** — :{color}[{r['band']} risk] · "
                f"support {r['support_pct']}% · default rate "
                f"{r['default_rate_pct']}% · lift {r['lift']}×"
            )
            st.code(f"IF {r['conditions']} THEN {r['band']} risk", language="text")


# =========================================================================
# Section 6: Talk-to-Data Chatbot
# =========================================================================
def render_chatbot():
    st.title("💬 Talk-to-Data")
    st.caption(
        f"Active LLM provider: **{SETTINGS.active_llm_provider}** "
        "(set OPENAI_API_KEY / GROQ_API_KEY / GEMINI_API_KEY to switch)"
    )

    from src.llm.nl_to_sql import answer

    suggestions = [
        "How many applicants are there in total?",
        "What is the overall default rate?",
        "Default rate by education level",
        "Top 5 occupations by default rate (min 100 clients)",
        "Average income for defaulters vs non-defaulters",
        "Default rate by gender",
    ]
    st.markdown("**Try a sample question:**")
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"sg_{i}", use_container_width=True):
            st.session_state["chat_input"] = s

    question = st.text_input(
        "Ask a question about the data:",
        value=st.session_state.get("chat_input", ""),
        placeholder="e.g. average loan amount by housing type",
    )
    go = st.button("🔎 Ask", type="primary")

    if go and question.strip():
        with st.spinner("Translating your question to SQL and running it…"):
            res = answer(question.strip())
        if res.error:
            st.error(res.answer)
        else:
            st.success(res.answer)
        with st.expander("Generated SQL"):
            st.code(res.sql, language="sql")
        if res.rows:
            st.markdown("**Result rows**")
            st.dataframe(pd.DataFrame(res.rows), use_container_width=True)
        st.caption(
            f"Provider: {res.provider}"
            + (" (deterministic fallback)" if res.used_fallback else "")
        )


# =================================== Router ==================================
ROUTES = {
    "🏠 Overview": render_overview,
    "📊 EDA": render_eda,
    "🎯 Risk Prediction": render_predict,
    "🔍 Explainability": render_explain,
    "📜 Decision Rules": render_rules,
    "💬 Talk-to-Data": render_chatbot,
}
ROUTES[section]()
