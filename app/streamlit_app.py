"""Streamlit UI: Overview, EDA, Predict, Explain, Rules, Chatbot."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import SETTINGS
from src.data.loader import ingest_to_sqlite, load_dataframe
from src.data.preprocess import add_engineered_features

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# custom CSS for cards and spacing
st.markdown("""
<style>
.big-metric {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    margin: 0;
}
.metric-label {
    font-size: 0.9rem;
    color: #666;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.5rem;
    color: white;
    margin-bottom: 1rem;
}
.info-card h3 {
    margin: 0 0 0.5rem 0;
    color: white;
}
.info-card p {
    margin: 0;
    opacity: 0.9;
}
.feature-box {
    background: #f8f9fa;
    border-left: 4px solid #1f77b4;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-radius: 0 8px 8px 0;
}
.risk-high { color: #d32f2f; font-weight: bold; }
.risk-medium { color: #f57c00; font-weight: bold; }
.risk-low { color: #388e3c; font-weight: bold; }
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    color: #a0a0a0;
    text-align: center;
    padding: 8px 0;
    font-size: 0.75rem;
    z-index: 999;
    border-top: 1px solid #333;
}
.footer a { color: #667eea; text-decoration: none; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa 0%, #e8ecf0 100%);
}
div[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 4px 0;
}
</style>
<div class="footer">
    Credit Risk Intelligence Platform &bull; Built by Ashwani Dhayal &bull;
    <a href="https://github.com/ashwani-dhayal/credit-risk-platform" target="_blank">GitHub</a>
    &bull; May 2026
</div>
""", unsafe_allow_html=True)


# ---- cached loaders ----
@st.cache_data(show_spinner=False)
def get_data():
    ingest_to_sqlite(force=False)
    df = load_dataframe()
    return add_engineered_features(df)


@st.cache_data(show_spinner=False)
def get_metrics():
    p = SETTINGS.models_dir / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


@st.cache_data(show_spinner=False)
def get_rules():
    p = SETTINGS.models_dir / "rules.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


# ---- sidebar ----
with st.sidebar:
    st.markdown("### 🏦 Credit Risk AI")
    st.caption("Intelligent Credit Risk Assessment Platform")
    st.divider()
    section = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 EDA",
            "🎯 Risk Prediction",
            "🔍 Explainability",
            "📜 Decision Rules",
            "📊 Model Performance",
            "💬 Talk-to-Data",
            "📈 Improve Your Score",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("##### System Status")
    if SETTINGS.model_path.exists():
        st.success("Model: Loaded", icon="✅")
    else:
        st.error("Model: Not trained", icon="❌")
    provider = SETTINGS.active_llm_provider
    if provider != "fallback":
        st.success(f"LLM: {provider.title()}", icon="🤖")
    else:
        st.warning("LLM: Fallback mode", icon="⚠️")
    st.divider()
    st.caption("Built with Streamlit + LightGBM + SHAP")


# ============================ Overview ============================
def render_overview():
    st.markdown("# 💳 AI-Powered Credit Risk Intelligence Platform")
    st.markdown("---")

    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### What is Credit Risk?

        **Credit risk** is the possibility that a borrower will fail to repay a loan
        or meet contractual obligations. It's the single largest risk that banks and
        financial institutions face — a poorly managed loan portfolio can lead to
        massive losses.

        Traditional credit assessment relies on manual review of applicant profiles,
        which is slow, inconsistent, and doesn't scale. This platform brings
        **machine learning and AI** to the problem, enabling:

        - **Faster decisions** — score an applicant in under a second
        - **Consistent policy** — every applicant evaluated by the same model
        - **Transparency** — every prediction is explained with SHAP values
        - **Self-service analytics** — ask questions in plain English, get SQL-backed answers
        """)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Quick Stats</h3>
            <p>This platform analyses <strong>10,000+</strong> historical loan
            applications to predict which future applicants are likely to default.</p>
            <br/>
            <p>Model accuracy: <strong>ROC-AUC 0.895</strong></p>
            <p>Default detection: <strong>KS 0.657</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # KPI metrics
    df = get_data()
    metrics = get_metrics()

    st.markdown("### 📈 Portfolio at a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Applicants", f"{len(df):,}")
    c2.metric("Default Rate", f"{df['TARGET'].mean()*100:.1f}%")
    c3.metric("Features Used", f"{df.shape[1]-2}")
    if metrics:
        c4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c5.metric("KS Statistic", f"{metrics['ks_statistic']:.3f}")
    else:
        c4.metric("ROC-AUC", "—")
        c5.metric("KS Statistic", "—")

    st.markdown("---")

    # Platform features
    st.markdown("### 🧩 Platform Capabilities")
    f1, f2, f3 = st.columns(3)
    with f1:
        with st.container(border=True):
            st.markdown("#### 📊 Exploratory Analysis")
            st.markdown(
                "Dive into the loan portfolio with interactive charts. "
                "Understand default patterns across demographics, income levels, "
                "education, employment, and external credit scores."
            )
            st.markdown("*→ Navigate to **EDA** in the sidebar*")
    with f2:
        with st.container(border=True):
            st.markdown("#### 🎯 ML Risk Scoring")
            st.markdown(
                "Enter any applicant's details and get an instant risk score. "
                "The LightGBM model outputs a probability of default, classifies "
                "into Low/Medium/High risk, and recommends Approve/Review/Reject."
            )
            st.markdown("*→ Navigate to **Risk Prediction***")
    with f3:
        with st.container(border=True):
            st.markdown("#### 🔍 Explainable AI")
            st.markdown(
                "Every prediction comes with SHAP-based explanations showing "
                "exactly which features pushed the risk up or down. No black box — "
                "full transparency for compliance and auditing."
            )
            st.markdown("*→ Navigate to **Explainability***")

    f4, f5, f6 = st.columns(3)
    with f4:
        with st.container(border=True):
            st.markdown("#### 📜 Decision Rules")
            st.markdown(
                "Machine-derived IF-THEN rules that translate the complex model "
                "into business-readable policy. Each rule shows population support, "
                "default rate, and lift over the baseline."
            )
            st.markdown("*→ Navigate to **Decision Rules***")
    with f5:
        with st.container(border=True):
            st.markdown("#### 💬 Talk-to-Data")
            st.markdown(
                "Ask questions about the loan data in plain English. The AI agent "
                "translates your question into SQL, executes it safely, and "
                "summarises the results with concrete numbers."
            )
            st.markdown("*→ Navigate to **Talk-to-Data***")
    with f6:
        with st.container(border=True):
            st.markdown("#### 🛡️ Safety & Guardrails")
            st.markdown(
                "SQL injection protection, read-only database access, keyword "
                "blocking, table allowlisting, and automatic LIMIT caps ensure "
                "the chatbot can never modify or exfiltrate data."
            )
            st.markdown("*Built into the SQL safety module*")

    st.markdown("---")

    # How it works
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    ```
    ┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
    │  Raw Data   │────▶│ Feature Engineer  │────▶│   LightGBM     │
    │  (CSV/DB)   │     │  + Preprocessing │     │   Classifier   │
    └─────────────┘     └──────────────────┘     └───────┬────────┘
                                                         │
                              ┌───────────────────────────┼───────────────────┐
                              ▼                           ▼                   ▼
                    ┌─────────────────┐      ┌───────────────────┐   ┌──────────────┐
                    │  SHAP Explainer │      │  Decision Rules   │   │  Risk Band   │
                    │  (per-applicant │      │  (Tree → IF-THEN) │   │  + Decision  │
                    │   & global)     │      └───────────────────┘   └──────────────┘
                    └─────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        Talk-to-Data Chatbot                                  │
    │  Question → LLM → SQL → Safety Check → Execute → LLM Summary → Answer      │
    └─────────────────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")

    # Dataset info
    st.markdown("### 📁 About the Dataset")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        **Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
        (Kaggle Competition)

        **What it contains:**
        - Historical loan applications with outcomes (repaid vs defaulted)
        - Demographics: age, gender, education, family status, housing
        - Financials: income, loan amount, annuity, goods price
        - External data: normalised credit bureau scores (EXT_SOURCE_1/2/3)
        - Employment: occupation, tenure, income source type
        """)
    with d2:
        st.markdown("""
        **Key characteristics:**
        - ~308,000 applications in the full dataset (10,000 in demo mode)
        - Binary target: 1 = defaulted, 0 = repaid on time
        - ~8% default rate (class imbalance)
        - Mix of categorical and numeric features
        - Real-world missingness patterns (e.g. 55% of EXT_SOURCE_1 is NaN)

        **Why this dataset?** It's the industry standard benchmark for credit
        risk ML — used in academic papers, Kaggle competitions, and as a
        training ground for production credit models.
        """)

    st.markdown("---")

    # Tech stack
    st.markdown("### 🛠️ Technology Stack")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("**ML / Data**")
        st.markdown("- LightGBM\n- scikit-learn\n- pandas / NumPy\n- SQLite")
    with t2:
        st.markdown("**Explainability**")
        st.markdown("- SHAP\n- Decision Trees\n- Feature importance")
    with t3:
        st.markdown("**LLM / NLP**")
        st.markdown("- OpenAI / Groq / Gemini\n- NL-to-SQL agent\n- Safety guardrails")
    with t4:
        st.markdown("**Infrastructure**")
        st.markdown("- Streamlit\n- Docker\n- GitHub CI")


# ============================ EDA ============================
def render_eda():
    st.title("📊 Exploratory Data Analysis")
    st.markdown(
        "Interactive exploration of the loan portfolio. Use the tabs below to "
        "investigate different dimensions of the data."
    )
    df = get_data()

    # Quick stats row at the top
    st.markdown("#### Quick Numbers")
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Defaults", f"{df['TARGET'].sum():,.0f}")
    q3.metric("Default %", f"{df['TARGET'].mean()*100:.1f}%")
    q4.metric("Avg Income", f"₹{df['AMT_INCOME_TOTAL'].mean():,.0f}")
    q5.metric("Avg Loan", f"₹{df['AMT_CREDIT'].mean():,.0f}")
    q6.metric("Avg Age", f"{df['AGE_YEARS'].mean():.0f} yrs")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Summary", "👥 Demographics", "💰 Financials", "📉 Default Drivers"]
    )

    with tab1:
        st.subheader("Dataset Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Data composition:**")
            comp = pd.DataFrame({
                "Category": ["Repaid (TARGET=0)", "Defaulted (TARGET=1)"],
                "Count": [int((df["TARGET"]==0).sum()), int((df["TARGET"]==1).sum())],
                "Percentage": [
                    f"{(1-df['TARGET'].mean())*100:.1f}%",
                    f"{df['TARGET'].mean()*100:.1f}%"
                ],
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)

            fig = px.pie(
                values=[int((df["TARGET"]==0).sum()), int((df["TARGET"]==1).sum())],
                names=["Repaid", "Defaulted"],
                title="Target Distribution",
                color_discrete_sequence=["#3498db", "#e74c3c"],
                hole=0.4,
            )
            fig.update_traces(textposition='outside', textinfo='percent+label',
                              pull=[0, 0.05])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Missing values (top 10):**")
            missing = (
                df.isna().sum().sort_values(ascending=False).head(10).reset_index()
            )
            missing.columns = ["Column", "Missing Count"]
            missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(1)
            missing = missing[missing["Missing Count"] > 0]
            if len(missing) > 0:
                fig = px.bar(
                    missing.sort_values("Missing %", ascending=True),
                    x="Missing %", y="Column", orientation="h",
                    title="Missing Value Proportions",
                    color="Missing %",
                    color_continuous_scale=["#f39c12", "#e74c3c"],
                )
                fig.update_layout(coloraxis_showscale=False)
                fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant missing values in this sample.")

        st.markdown("**Sample rows (first 8)**")
        st.dataframe(df.head(8), use_container_width=True)

    with tab2:
        st.subheader("Demographic Analysis")
        st.markdown("How does default rate vary across different applicant segments?")
        c1, c2 = st.columns(2)
        with c1:
            df_hist = df.copy()
            df_hist["Status"] = df_hist["TARGET"].map({0: "Repaid", 1: "Defaulted"})
            fig = px.histogram(
                df_hist, x="AGE_YEARS", color="Status", nbins=30,
                barmode="overlay", opacity=0.65,
                title="Age Distribution by Default Status",
                labels={"AGE_YEARS": "Age (years)"},
                color_discrete_map={"Repaid": "#3498db", "Defaulted": "#e74c3c"},
            )
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            gender_def = df.groupby("CODE_GENDER").agg(
                default_rate=("TARGET", "mean"),
                count=("TARGET", "size")
            ).reset_index()
            gender_def["default_rate_pct"] = gender_def["default_rate"] * 100
            fig = px.bar(
                gender_def, x="CODE_GENDER", y="default_rate_pct",
                title="Default Rate by Gender",
                text_auto=".2f",
                color="default_rate_pct",
                color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
                labels={"CODE_GENDER": "Gender", "default_rate_pct": "Default Rate (%)"},
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            edu = (
                df.groupby("NAME_EDUCATION_TYPE")
                .agg(default_rate=("TARGET", "mean"), n=("TARGET", "size"))
                .reset_index()
                .sort_values("default_rate", ascending=True)
            )
            edu["default_rate_pct"] = edu["default_rate"] * 100
            fig = px.bar(
                edu, x="default_rate_pct", y="NAME_EDUCATION_TYPE",
                orientation="h",
                title="Default Rate by Education Level",
                text_auto=".1f",
                color="default_rate_pct",
                color_continuous_scale=["#27ae60", "#f1c40f", "#e67e22", "#e74c3c"],
                labels={"NAME_EDUCATION_TYPE": "", "default_rate_pct": "Default Rate (%)"},
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            inc = (
                df.groupby("NAME_INCOME_TYPE")
                .agg(default_rate=("TARGET", "mean"), n=("TARGET", "size"))
                .reset_index()
                .sort_values("default_rate", ascending=True)
            )
            inc["default_rate_pct"] = inc["default_rate"] * 100
            fig = px.bar(
                inc, x="default_rate_pct", y="NAME_INCOME_TYPE",
                orientation="h",
                title="Default Rate by Income Type",
                text_auto=".1f",
                color="default_rate_pct",
                color_continuous_scale=["#27ae60", "#f1c40f", "#e67e22", "#e74c3c"],
                labels={"NAME_INCOME_TYPE": "", "default_rate_pct": "Default Rate (%)"},
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        # Family status
        fam = (
            df.groupby("NAME_FAMILY_STATUS")
            .agg(default_rate=("TARGET", "mean"), n=("TARGET", "size"))
            .reset_index()
            .sort_values("default_rate", ascending=False)
        )
        fam["default_rate_pct"] = fam["default_rate"] * 100
        fig = px.bar(
            fam, x="NAME_FAMILY_STATUS", y="default_rate_pct",
            title="Default Rate by Family Status",
            text_auto=".1f",
            color="default_rate_pct",
            color_continuous_scale=["#27ae60", "#f1c40f", "#e67e22", "#e74c3c"],
            labels={"NAME_FAMILY_STATUS": "Family Status", "default_rate_pct": "Default Rate (%)"},
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Financial Analysis")
        st.markdown("How do financial characteristics differ between defaulters and non-defaulters?")
        view = df[df["AMT_INCOME_TOTAL"] < df["AMT_INCOME_TOTAL"].quantile(0.99)].copy()
        view["Status"] = view["TARGET"].map({0: "Repaid", 1: "Defaulted"})
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                view, x="AMT_INCOME_TOTAL", color="Status", nbins=40,
                barmode="overlay", opacity=0.65,
                title="Income Distribution (≤99th percentile)",
                color_discrete_map={"Repaid": "#3498db", "Defaulted": "#e74c3c"},
                labels={"AMT_INCOME_TOTAL": "Annual Income (₹)"},
            )
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(
                view, x="CREDIT_INCOME_RATIO", color="Status", nbins=40,
                barmode="overlay", opacity=0.65,
                title="Credit-to-Income Ratio Distribution",
                color_discrete_map={"Repaid": "#3498db", "Defaulted": "#e74c3c"},
                labels={"CREDIT_INCOME_RATIO": "Loan / Income Ratio"},
            )
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        # Box plots for key financials
        df_box = df.copy()
        df_box["Status"] = df_box["TARGET"].map({0: "Repaid", 1: "Defaulted"})
        c3, c4 = st.columns(2)
        with c3:
            fig = px.box(
                df_box, x="Status", y="AMT_CREDIT",
                title="Loan Amount by Default Status",
                color="Status",
                color_discrete_map={"Repaid": "#3498db", "Defaulted": "#e74c3c"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.box(
                df_box, x="Status", y="ANNUITY_INCOME_RATIO",
                title="Annuity/Income Ratio by Default Status",
                color="Status",
                color_discrete_map={"Repaid": "#3498db", "Defaulted": "#e74c3c"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Median Financial Metrics by Default Status:**")
        agg = (
            df.groupby("TARGET")[
                ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                 "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]
            ]
            .median()
            .round(2)
        )
        agg.index = ["✅ Repaid", "❌ Defaulted"]
        st.dataframe(agg, use_container_width=True)

    with tab4:
        st.subheader("What Drives Defaults?")
        st.markdown("These are the features most correlated with loan default in this dataset.")
        num = df.select_dtypes("number").drop(columns=["SK_ID_CURR"], errors="ignore")
        corr = num.corr(numeric_only=True)["TARGET"].drop("TARGET")
        top = corr.abs().sort_values(ascending=False).head(15)
        top_signed = corr.loc[top.index].round(4)
        df_corr = top_signed.reset_index()
        df_corr.columns = ["Feature", "Correlation with Default"]
        df_corr["Direction"] = df_corr["Correlation with Default"].apply(
            lambda x: "Increases Risk ↑" if x > 0 else "Decreases Risk ↓"
        )
        fig = px.bar(
            df_corr.sort_values("Correlation with Default"),
            x="Correlation with Default", y="Feature", orientation="h",
            title="Top 15 Features Correlated with Default",
            color="Direction",
            color_discrete_map={
                "Increases Risk ↑": "#e74c3c",
                "Decreases Risk ↓": "#3498db",
            },
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap for top features
        st.markdown("---")
        st.markdown("#### 🔥 Feature Correlation Heatmap (Top 10)")
        top10_features = list(top.head(10).index) + ["TARGET"]
        corr_matrix = num[top10_features].corr()
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[c.replace("_", " ")[:15] for c in corr_matrix.columns],
            y=[c.replace("_", " ")[:15] for c in corr_matrix.columns],
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 9},
        ))
        fig_heat.update_layout(height=500, title="Inter-Feature Correlations")
        st.plotly_chart(fig_heat, use_container_width=True)

        # Interactive feature explorer
        st.markdown("---")
        st.markdown("#### 🔬 Feature Explorer")
        st.markdown("Select a numeric feature to see its distribution split by default status.")
        numeric_features = [c for c in num.columns if c != "TARGET" and df[c].nunique() > 5]
        selected_feat = st.selectbox("Choose a feature:", sorted(numeric_features), index=numeric_features.index("CREDIT_INCOME_RATIO") if "CREDIT_INCOME_RATIO" in numeric_features else 0)

        if selected_feat:
            explore_df = df[[selected_feat, "TARGET"]].dropna().copy()
            explore_df["Status"] = explore_df["TARGET"].map({0: "Repaid", 1: "Defaulted"})
            q99 = explore_df[selected_feat].quantile(0.99)
            explore_df = explore_df[explore_df[selected_feat] <= q99]

            fig_explore = px.histogram(
                explore_df, x=selected_feat, color="Status",
                barmode="overlay", opacity=0.6, nbins=40,
                color_discrete_map={"Repaid": "#3498db", "Defaulted": "#e74c3c"},
                title=f"Distribution of {selected_feat} by Default Status",
            )
            fig_explore.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_explore, use_container_width=True)

            # Quick stats for that feature
            sc1, sc2, sc3, sc4 = st.columns(4)
            repaid_vals = explore_df[explore_df["TARGET"] == 0][selected_feat]
            defaulted_vals = explore_df[explore_df["TARGET"] == 1][selected_feat]
            sc1.metric("Mean (Repaid)", f"{repaid_vals.mean():.3f}")
            sc2.metric("Mean (Defaulted)", f"{defaulted_vals.mean():.3f}")
            sc3.metric("Median (Repaid)", f"{repaid_vals.median():.3f}")
            sc4.metric("Median (Defaulted)", f"{defaulted_vals.median():.3f}")

        st.markdown("---")
        st.markdown("#### 💡 Key Insights from EDA")
        with st.container(border=True):
            st.markdown("""
            1. **External credit scores are the strongest signal** — `EXT_SOURCE_MEAN` has the
               highest negative correlation with default. Clients with high external scores
               default ~5x less than those with low scores.

            2. **Leverage matters more than income** — It's not how much you earn, it's how
               much you borrow relative to earnings. `CREDIT_INCOME_RATIO` is a stronger
               predictor than `AMT_INCOME_TOTAL` alone.

            3. **Age is protective** — Younger applicants (under 30) default at higher rates.
               The relationship is roughly linear until age 45, then flattens.

            4. **Education creates separation** — Higher education correlates with lower
               default rates, likely because it proxies for financial literacy and stable employment.

            5. **Employment tenure reduces risk** — Each additional year of employment
               measurably decreases default probability.
            """)


# ============================ Predict ============================
def render_predict():
    st.title("🎯 Risk Prediction Engine")
    st.markdown("---")

    # Explanation section
    with st.container(border=True):
        st.markdown("""
        **How this works:**
        Enter the applicant's details in the form below. The trained LightGBM model
        will score the application and return:
        - **Probability of Default** — a number between 0% and 100%
        - **Risk Band** — Low (< 20%), Medium (20-50%), or High (> 50%)
        - **Recommended Decision** — Approve, Review, or Reject

        The model was trained on historical loan data and achieves ROC-AUC of 0.895
        on the validation set.
        """)

    if not SETTINGS.model_path.exists():
        st.error(
            "⚠️ No trained model found. "
            "Run `python scripts/train_model.py` first, then refresh this page."
        )
        return

    from src.ml.predict import predict_one

    df = get_data()

    def options(col):
        return sorted([str(x) for x in df[col].dropna().unique().tolist()])

    st.markdown("### Applicant Details")
    with st.form("predict_form"):
        st.markdown("##### 👤 Personal Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", options("CODE_GENDER"), index=0)
            age_years = st.number_input("Age (years)", 21, 70, 35)
        with c2:
            education = st.selectbox("Education", options("NAME_EDUCATION_TYPE"))
            family_status = st.selectbox("Family status", options("NAME_FAMILY_STATUS"))
        with c3:
            housing = st.selectbox("Housing", options("NAME_HOUSING_TYPE"))
            occupation_opts = ["(unknown)"] + options("OCCUPATION_TYPE")
            occupation = st.selectbox("Occupation", occupation_opts, index=0)
        with c4:
            children = st.number_input("Children", 0, 10, 0)
            fam_members = st.number_input("Family members", 1, 12, 2)

        st.markdown("##### 💰 Financial Details")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            income = st.number_input("Annual income (₹)", 25_000, 5_000_000, 180_000, step=5_000)
            income_type = st.selectbox("Income type", options("NAME_INCOME_TYPE"))
        with c2:
            credit = st.number_input("Loan amount (₹)", 45_000, 4_000_000, 600_000, step=10_000)
            contract = st.selectbox("Contract type", options("NAME_CONTRACT_TYPE"), index=0)
        with c3:
            annuity = st.number_input("Annuity (yearly ₹)", 5_000, 400_000, 30_000, step=1_000)
            goods = st.number_input("Goods price (₹)", 45_000, 4_000_000, 540_000, step=10_000)
        with c4:
            emp_years = st.number_input("Years employed", 0.0, 45.0, 5.0, step=0.5)
            region = st.selectbox("Region rating (1=best)", [1, 2, 3], index=1)

        st.markdown("##### 🏦 Assets & Credit Scores")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            own_car = st.selectbox("Owns car?", ["Y", "N"], index=1)
        with c2:
            own_realty = st.selectbox("Owns property?", ["Y", "N"], index=0)
        with c3:
            ext1 = st.text_input("EXT_SOURCE_1 (0-1)", value="")
        with c4:
            ext2 = st.text_input("EXT_SOURCE_2 (0-1)", value="0.55")
        with c5:
            ext3 = st.text_input("EXT_SOURCE_3 (0-1)", value="0.50")

        submitted = st.form_submit_button(
            "🚀 Score This Applicant", use_container_width=True, type="primary"
        )

    if not submitted:
        st.info("👆 Fill in the details above and click **Score This Applicant** to get a prediction.")
        return

    def parse_float(s):
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
        "EXT_SOURCE_1": parse_float(ext1),
        "EXT_SOURCE_2": parse_float(ext2),
        "EXT_SOURCE_3": parse_float(ext3),
    }

    pred = predict_one(applicant)
    st.session_state["last_applicant"] = applicant
    st.session_state["last_prediction"] = pred.__dict__

    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    band_colors = {"Low": "#388e3c", "Medium": "#f57c00", "High": "#d32f2f"}
    band_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    decision_emoji = {"Approve": "✅", "Review": "⚠️", "Reject": "🛑"}

    r1, r2, r3 = st.columns(3)
    with r1:
        with st.container(border=True):
            st.markdown("**Probability of Default**")
            st.markdown(
                f"<p class='big-metric'>{pred.probability_default*100:.2f}%</p>",
                unsafe_allow_html=True
            )
            st.progress(min(pred.probability_default, 1.0))
    with r2:
        with st.container(border=True):
            st.markdown("**Risk Band**")
            st.markdown(
                f"<p class='big-metric' style='color:{band_colors[pred.risk_band]}'>"
                f"{band_emoji[pred.risk_band]} {pred.risk_band}</p>",
                unsafe_allow_html=True
            )
            st.caption("Low < 20% | Medium 20-50% | High > 50%")
    with r3:
        with st.container(border=True):
            st.markdown("**Recommended Decision**")
            st.markdown(
                f"<p class='big-metric'>{decision_emoji[pred.decision]} {pred.decision}</p>",
                unsafe_allow_html=True
            )
            st.caption("Based on risk band thresholds")

    # Key ratios computed
    st.markdown("---")
    st.markdown("##### 📐 Computed Risk Ratios")
    ratio_cols = st.columns(4)
    ratio_cols[0].metric("Credit/Income", f"{credit/max(income,1):.2f}x")
    ratio_cols[1].metric("Annuity/Income", f"{annuity/max(income,1)*100:.1f}%")
    ratio_cols[2].metric("Loan Term (yrs)", f"{credit/max(annuity,1):.1f}")
    ratio_cols[3].metric("Goods Coverage", f"{goods/max(credit,1)*100:.0f}%")

    with st.expander("🔧 Raw applicant payload sent to model"):
        st.json(applicant)


# ============================ Explain ============================
def render_explain():
    st.title("🔍 Explainability (SHAP)")
    st.markdown("---")

    with st.container(border=True):
        st.markdown("""
        **What is SHAP?**

        SHAP (SHapley Additive exPlanations) uses game theory to assign each feature
        a contribution to the prediction. Positive SHAP values push the prediction toward
        default; negative values push toward repayment. This gives us **exact, per-feature
        accountability** for every single prediction.
        """)

    if not SETTINGS.model_path.exists():
        st.error("Train the model first: `python scripts/train_model.py`")
        return

    from src.ml.explain import explain_one, global_importance

    tab1, tab2 = st.tabs(["🔬 Per-Applicant Explanation", "🌍 Global Feature Importance"])

    with tab1:
        applicant = st.session_state.get("last_applicant")
        if not applicant:
            st.info(
                "👈 Score an applicant in the **Risk Prediction** section first, "
                "then come back here to see what drove that prediction."
            )
            return

        pred = st.session_state.get("last_prediction", {})
        if pred:
            st.markdown(
                f"**Explaining prediction:** P(default) = "
                f"**{pred.get('probability_default', 0)*100:.2f}%** → "
                f"**{pred.get('risk_band', '?')}** risk"
            )

        contributions = explain_one(applicant, top_n=10)
        df_exp = pd.DataFrame([c.__dict__ for c in contributions])
        df_exp = df_exp.sort_values("shap_value", key=lambda s: s.abs(), ascending=True)

        fig = px.bar(
            df_exp, x="shap_value", y="feature", color="direction",
            orientation="h",
            title="Top 10 Feature Contributions (SHAP Values)",
            color_discrete_map={
                "increases risk": "#e74c3c",
                "decreases risk": "#2ecc71",
            },
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Detailed breakdown:**")
        show_df = df_exp[["feature", "value", "shap_value", "direction"]].copy()
        show_df.columns = ["Feature", "Value", "SHAP Value", "Effect"]
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(
            "Global importance shows which features matter most **across all applicants**, "
            "not just one. Computed as mean |SHAP| over a 400-row sample."
        )
        with st.spinner("Computing global SHAP values (400 samples)…"):
            gi = global_importance(sample_size=400).head(15)
        fig = px.bar(
            gi.sort_values("mean_abs_shap"),
            x="mean_abs_shap", y="feature", orientation="h",
            title="Top 15 Global Feature Drivers (Mean |SHAP|)",
            color="mean_abs_shap",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretation:** Features at the top of this chart have the most influence
        on predictions across the entire portfolio. External credit scores (`EXT_SOURCE_*`)
        consistently dominate, followed by credit-to-income ratios and age.
        """)


# ============================ Rules ============================
def render_rules():
    st.title("📜 Business Decision Rules")
    st.markdown("---")

    with st.container(border=True):
        st.markdown("""
        **What are these rules?**

        These are human-readable IF-THEN rules automatically extracted from a decision tree
        trained on the same data as the main model. They translate complex ML patterns into
        actionable business policy that compliance teams and loan officers can understand
        without any data science background.

        Each rule shows:
        - **Support** — what % of the population falls into this rule
        - **Default Rate** — what % of that group actually defaulted
        - **Lift** — how much riskier/safer this group is vs. the 8% baseline
        """)

    rules = get_rules()
    if not rules:
        st.warning("No rules found. Run `python scripts/derive_rules.py` after training.")
        return

    # Summary stats
    df = pd.DataFrame(rules)
    r1, r2, r3 = st.columns(3)
    r1.metric("Total Rules", len(df))
    r2.metric("High Risk Rules", len(df[df["band"] == "High"]))
    r3.metric("Low Risk Rules", len(df[df["band"] == "Low"]))

    st.markdown("---")

    # Rules table
    st.markdown("### Rules Summary Table")
    df_show = df.copy()
    df_show["conditions"] = df_show["conditions"].apply(lambda c: " AND ".join(c) if c else "—")
    show = df_show[["rule_id", "band", "conditions", "support_pct",
                    "default_rate_pct", "lift", "n_samples"]]
    show.columns = ["#", "Risk", "Conditions", "Support %", "Default %", "Lift", "Samples"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Rule cards
    st.markdown("---")
    st.markdown("### Rule Cards")

    def explain_rule(conditions, band, default_rate, lift, support, n):
        """Turn conditions into a human-readable explanation."""
        if not conditions:
            return "This is a catch-all rule for applicants not matched elsewhere."
        parts = []
        for c in conditions:
            # Make conditions more readable
            c_clean = c.replace("_", " ").replace("  ", " ")
            parts.append(f"- {c}")
        cond_text = "\n".join(parts)

        if band == "High":
            verdict = (
                f"Applicants matching these conditions default at **{default_rate}%** — "
                f"that's **{lift}× higher** than the overall 8% base rate. "
                f"This group represents {support}% of all applicants ({n:,} people). "
                "These are high-risk profiles that warrant rejection or deeper review."
            )
        elif band == "Medium":
            verdict = (
                f"This segment defaults at **{default_rate}%** — "
                f"**{lift}× the baseline**. Covers {support}% of applicants ({n:,} people). "
                "These cases need manual review before a final decision."
            )
        else:
            verdict = (
                f"Only **{default_rate}%** default in this group — "
                f"**{lift}× the baseline** (lower is better here). "
                f"Covers {support}% of applicants ({n:,} people). "
                "These are strong profiles suitable for fast-track approval."
            )
        return cond_text, verdict

    # Group by risk band
    for band in ["High", "Medium", "Low"]:
        band_rules = df[df["band"] == band]
        if band_rules.empty:
            continue
        band_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[band]
        band_color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}[band]
        st.markdown(f"#### {band_icon} {band} Risk Rules")
        for _, r in band_rules.iterrows():
            conditions = r["conditions"] if r["conditions"] else []
            with st.container(border=True):
                st.markdown(
                    f'<div style="border-left: 5px solid {band_color}; padding-left: 12px;">',
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Rule {r['rule_id']}** — {band_icon} **{r['band']} Risk**")
                    cond_text, verdict = explain_rule(
                        conditions, r["band"], r["default_rate_pct"],
                        r["lift"], r["support_pct"], r["n_samples"]
                    )
                    st.markdown("**Conditions:**")
                    for c in conditions:
                        st.markdown(f"  `{c}`")
                    st.markdown("")
                    st.markdown(verdict)
                with col2:
                    st.metric("Default Rate", f"{r['default_rate_pct']}%")
                    st.metric("Lift vs Base", f"{r['lift']}×")
                    st.metric("Population", f"{r['support_pct']}%")
                    st.caption(f"{r['n_samples']:,} applicants")
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")


# ============================ Model Performance ============================
def render_model_performance():
    st.title("📊 Model Performance & Evaluation")
    st.markdown("---")

    if not SETTINGS.model_path.exists():
        st.error("Train the model first: `python scripts/train_model.py`")
        return

    metrics = get_metrics()
    if not metrics:
        st.warning("No metrics.json found. Re-run training.")
        return

    with st.container(border=True):
        st.markdown("""
        **Model evaluation dashboard** — Understand how well the LightGBM classifier
        performs on held-out validation data. All metrics below were computed on a
        stratified 20% test set that the model never saw during training.
        """)

    # Key metrics row
    st.markdown("### 🎯 Key Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", help="Area under the ROC curve. 1.0 is perfect, 0.5 is random.")
    m2.metric("KS Statistic", f"{metrics.get('ks_statistic', 0):.3f}", help="Max separation between cumulative distributions of defaulters vs non-defaulters.")
    m3.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}", help="Precision-Recall AUC. Robust to class imbalance.")
    m4.metric("F1 Score", f"{metrics.get('f1', 0):.3f}", help="Harmonic mean of precision and recall at optimal threshold.")

    st.markdown("---")

    # Generate ROC curve from model predictions
    from src.ml.predict import _load_model_and_preprocessor
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

    try:
        model, preprocessor, feature_cols = _load_model_and_preprocessor()
        df = get_data()
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["TARGET"])

        X_val = val_df[feature_cols] if all(c in val_df.columns for c in feature_cols) else val_df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
        y_val = val_df["TARGET"].values
        X_proc = preprocessor.transform(X_val)
        y_prob = model.predict_proba(X_proc)[:, 1]

        tab1, tab2, tab3 = st.tabs(["📈 ROC Curve", "🎯 Precision-Recall", "📊 Threshold Analysis"])

        with tab1:
            fpr, tpr, thresholds_roc = roc_curve(y_val, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"Model (AUC={metrics.get('roc_auc', 0):.3f})", line=dict(color='#3498db', width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random Baseline", line=dict(color='#95a5a6', dash='dash')))
            fig.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
                legend=dict(x=0.55, y=0.1),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Interpretation:** The ROC curve shows the trade-off between catching defaults
            (True Positive Rate) and false alarms (False Positive Rate). The further the
            blue line is from the dashed diagonal, the better the model discriminates.
            """)

        with tab2:
            precision, recall, thresholds_pr = precision_recall_curve(y_val, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name="Model", line=dict(color='#e74c3c', width=2.5)))
            fig.add_hline(y=y_val.mean(), line_dash="dash", line_color="#95a5a6", annotation_text=f"Baseline ({y_val.mean():.2%})")
            fig.update_layout(
                title="Precision-Recall Curve",
                xaxis_title="Recall (Sensitivity)",
                yaxis_title="Precision",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Interpretation:** The PR curve is especially informative for imbalanced data.
            Precision = "Of those I flagged as defaulters, how many actually defaulted?"
            Recall = "Of all actual defaulters, how many did I catch?"
            """)

        with tab3:
            st.markdown("#### How the decision threshold affects outcomes")
            st.markdown(
                "Move the slider to see how different probability thresholds change "
                "the confusion matrix and approval/rejection rates."
            )
            threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.20, 0.05)
            y_pred = (y_prob >= threshold).astype(int)
            cm = confusion_matrix(y_val, y_pred)
            tn, fp, fn, tp = cm.ravel()

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("True Negatives (Correct Approvals)", f"{tn:,}")
            cm2.metric("False Positives (Wrongly Rejected)", f"{fp:,}")
            cm3.metric("False Negatives (Missed Defaults)", f"{fn:,}")
            cm4.metric("True Positives (Caught Defaults)", f"{tp:,}")

            approval_rate = (tn + fn) / len(y_val) * 100
            catch_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            false_alarm = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

            st.markdown("---")
            a1, a2, a3 = st.columns(3)
            a1.metric("Approval Rate", f"{approval_rate:.1f}%", help="% of applicants that would be approved")
            a2.metric("Default Catch Rate", f"{catch_rate:.1f}%", help="% of actual defaulters identified")
            a3.metric("False Alarm Rate", f"{false_alarm:.1f}%", help="% of good applicants wrongly flagged")

            # Confusion matrix heatmap
            fig = go.Figure(data=go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted: Repaid", "Predicted: Default"],
                y=["Actual: Repaid", "Actual: Default"],
                colorscale="Blues",
                text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=False,
            ))
            fig.update_layout(title=f"Confusion Matrix (threshold = {threshold:.2f})", height=350)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not generate model curves: {e}")
        st.info("This may happen if the model was trained with a different feature set. Re-train to fix.")


# ============================ Chatbot ============================
def render_chatbot():
    st.title("💬 Talk-to-Data: AI-Powered Q&A")
    st.markdown("---")

    with st.container(border=True):
        st.markdown(f"""
        **Ask any question about the loan data in plain English.**
        The AI agent translates your question into SQL, runs it safely against the database,
        and explains the results. You can also ask general finance/credit questions.

        Currently using: **{SETTINGS.active_llm_provider.title()}**
        {"" if SETTINGS.active_llm_provider != "fallback" else " (limited to 9 preset queries — add an API key for unlimited questions)"}
        """)

    from src.llm.nl_to_sql import answer

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggestions
    st.markdown("#### 💡 Try These:")
    suggestions = [
        ("📊", "How many applicants are there?"),
        ("📈", "Overall default rate"),
        ("🎓", "Default rate by education"),
        ("💼", "Top occupations by default"),
        ("💰", "Income: defaulters vs non-defaulters"),
        ("🏠", "Default rate by housing type"),
        ("💳", "What is credit risk?"),
        ("🏦", "What is a home loan?"),
    ]
    cols = st.columns(4)
    for i, (icon, s) in enumerate(suggestions):
        if cols[i % 4].button(f"{icon} {s}", key=f"sg_{i}", use_container_width=True):
            st.session_state["chat_input"] = s

    st.markdown("---")

    question = st.text_input(
        "🔎 Your question:",
        value=st.session_state.get("chat_input", ""),
        placeholder="e.g. What is the average loan amount by housing type?",
    )

    col_ask, col_clear = st.columns([4, 1])
    with col_ask:
        go = st.button("Ask →", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    if go and question.strip():
        with st.spinner("🤖 Thinking…"):
            res = answer(question.strip())

        st.session_state["chat_history"].append({
            "question": question.strip(),
            "answer": res.answer,
            "sql": res.sql,
            "provider": res.provider,
            "rows": res.rows,
            "error": res.error,
            "used_fallback": res.used_fallback,
        })
        st.session_state["chat_input"] = ""
        st.rerun()

    # Render chat history (most recent first)
    if st.session_state["chat_history"]:
        st.markdown("---")
        st.markdown("### 💬 Conversation")

        for i, entry in enumerate(reversed(st.session_state["chat_history"])):
            idx = len(st.session_state["chat_history"]) - 1 - i

            # User message
            st.markdown(f"**🧑 You:** {entry['question']}")

            # Assistant response
            if entry.get("error"):
                st.error(f"❌ {entry['answer']}")
            else:
                with st.container(border=True):
                    st.markdown(entry["answer"])

            # Expandable details
            with st.expander(f"🔧 Details (Provider: {entry['provider']})", expanded=False):
                st.code(entry["sql"], language="sql")
                if entry["rows"]:
                    result_df = pd.DataFrame(entry["rows"])
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    if len(result_df.columns) >= 2 and len(result_df) > 1:
                        try:
                            numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()
                            str_cols = result_df.select_dtypes(include=["object"]).columns.tolist()
                            if numeric_cols and str_cols:
                                chart_df = result_df[[str_cols[0], numeric_cols[0]]].dropna()
                                if len(chart_df) > 0:
                                    fig = px.bar(
                                        chart_df,
                                        x=str_cols[0], y=numeric_cols[0],
                                        title=f"{numeric_cols[0].replace('_', ' ').title()} by {str_cols[0].replace('_', ' ').title()}",
                                        color=str_cols[0],
                                        color_discrete_sequence=px.colors.qualitative.Set2,
                                    )
                                    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

            if i < len(st.session_state["chat_history"]) - 1:
                st.markdown("---")


# ============================ Improve Score ============================
def render_improve_score():
    st.title("📈 How to Improve Your Credit Score")
    st.markdown("---")

    st.markdown("""
    Your **credit score** is a number that represents how likely you are to repay
    a loan on time. Lenders use it to decide whether to approve your application,
    what interest rate to offer, and how much credit to extend. A higher score means
    better loan terms and lower interest rates.
    """)

    # Score ranges
    st.markdown("### 🎯 Credit Score Ranges (CIBIL / India)")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("""
            <div style="text-align:center; padding:1rem; background:#e74c3c22; border-radius:8px;">
                <h2 style="color:#e74c3c; margin:0;">300-549</h2>
                <p style="margin:0;"><strong>Poor</strong></p>
                <p style="font-size:0.8rem; margin:0;">Loan rejection likely</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="text-align:center; padding:1rem; background:#f39c1222; border-radius:8px;">
                <h2 style="color:#f39c12; margin:0;">550-649</h2>
                <p style="margin:0;"><strong>Fair</strong></p>
                <p style="font-size:0.8rem; margin:0;">Higher interest rates</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div style="text-align:center; padding:1rem; background:#3498db22; border-radius:8px;">
                <h2 style="color:#3498db; margin:0;">650-749</h2>
                <p style="margin:0;"><strong>Good</strong></p>
                <p style="font-size:0.8rem; margin:0;">Standard rates</p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
            <div style="text-align:center; padding:1rem; background:#27ae6022; border-radius:8px;">
                <h2 style="color:#27ae60; margin:0;">750-900</h2>
                <p style="margin:0;"><strong>Excellent</strong></p>
                <p style="font-size:0.8rem; margin:0;">Best rates & fast approval</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tips section
    st.markdown("### 💡 10 Actionable Steps to Improve Your Credit Score")

    tips = [
        {
            "icon": "💳",
            "title": "Pay Bills On Time, Every Time",
            "desc": (
                "Payment history is the single biggest factor (35% weightage). Even one "
                "missed EMI or credit card payment can drop your score by 50-100 points. "
                "Set up auto-pay or reminders for all dues."
            ),
            "impact": "High",
        },
        {
            "icon": "📊",
            "title": "Keep Credit Utilization Below 30%",
            "desc": (
                "If your credit card limit is ₹1,00,000, don't use more than ₹30,000 at any time. "
                "High utilization signals desperation to lenders. Ideal is 10-20%."
            ),
            "impact": "High",
        },
        {
            "icon": "🚫",
            "title": "Don't Apply for Too Many Loans at Once",
            "desc": (
                "Each loan application triggers a 'hard inquiry' on your credit report, "
                "which temporarily lowers your score by 5-10 points. Space out applications "
                "by at least 3-6 months."
            ),
            "impact": "Medium",
        },
        {
            "icon": "📅",
            "title": "Maintain Old Credit Accounts",
            "desc": (
                "Length of credit history matters. Don't close your oldest credit card "
                "even if you rarely use it — it adds to your credit age. A longer history "
                "shows stability."
            ),
            "impact": "Medium",
        },
        {
            "icon": "🔄",
            "title": "Mix Your Credit Types",
            "desc": (
                "Having a healthy mix of secured loans (home, car) and unsecured credit "
                "(credit cards, personal loans) shows lenders you can manage different "
                "types of debt responsibly."
            ),
            "impact": "Medium",
        },
        {
            "icon": "🔍",
            "title": "Check Your Credit Report for Errors",
            "desc": (
                "Mistakes happen — wrong accounts, incorrect payment statuses, or "
                "duplicate entries. Check your CIBIL report at least once a year and "
                "dispute any errors immediately. This alone can boost your score."
            ),
            "impact": "High",
        },
        {
            "icon": "💰",
            "title": "Pay Off High-Interest Debt First",
            "desc": (
                "Focus on clearing credit card balances and personal loans first "
                "(debt avalanche method). Reducing outstanding debt directly improves "
                "your credit utilization ratio."
            ),
            "impact": "High",
        },
        {
            "icon": "🏦",
            "title": "Become an Authorized User",
            "desc": (
                "If a family member with excellent credit adds you as an authorized user "
                "on their card, their positive payment history can reflect on your report too. "
                "Make sure their utilization is low."
            ),
            "impact": "Low",
        },
        {
            "icon": "📱",
            "title": "Use Credit Monitoring Tools",
            "desc": (
                "Apps like CIBIL, Experian, or your bank's built-in tools let you "
                "track your score monthly. Watching the trend helps you catch problems "
                "early and stay motivated."
            ),
            "impact": "Low",
        },
        {
            "icon": "⏰",
            "title": "Be Patient — It Takes Time",
            "desc": (
                "Credit scores don't jump overnight. Consistent good behavior over "
                "6-12 months is what moves the needle. A single negative mark takes "
                "2-3 years to fully fade from your report."
            ),
            "impact": "Low",
        },
    ]

    for i, tip in enumerate(tips, 1):
        impact_color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#3498db"}[tip["impact"]]
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{tip['icon']} Tip {i}: {tip['title']}**")
                st.markdown(tip["desc"])
            with col2:
                st.markdown(
                    f'<div style="text-align:center; padding:0.5rem; '
                    f'background:{impact_color}22; border-radius:6px; margin-top:0.5rem;">'
                    f'<span style="color:{impact_color}; font-weight:bold; font-size:0.8rem;">'
                    f'{tip["impact"]} Impact</span></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # What our model looks at
    st.markdown("### 🤖 What Our ML Model Considers Most Important")
    st.markdown("""
    Based on our trained model's SHAP analysis, these are the top factors that
    determine whether an applicant is classified as high or low risk:
    """)

    factors = pd.DataFrame({
        "Factor": [
            "External Credit Scores (EXT_SOURCE)",
            "Credit-to-Income Ratio",
            "Age / Years of Experience",
            "Education Level",
            "Employment Stability (tenure)",
            "Annuity-to-Income Ratio",
            "Region Rating",
            "Number of Children",
        ],
        "What It Means": [
            "Scores from credit bureaus (CIBIL, Experian) — single strongest predictor",
            "How much you're borrowing vs. how much you earn",
            "Older applicants with more experience tend to default less",
            "Higher education correlates with lower default rates",
            "Longer time at current job = more stable = lower risk",
            "Monthly payment as % of income — lower is better",
            "Where you live affects risk (infrastructure, employment opportunities)",
            "More dependents = more expenses = slightly higher risk",
        ],
        "You Can Improve?": [
            "✅ Yes — pay EMIs on time, reduce debt",
            "✅ Yes — borrow less or earn more",
            "❌ No — but time is on your side",
            "✅ Yes — pursue higher education/certifications",
            "✅ Yes — stay at your current job longer",
            "✅ Yes — choose longer loan tenure or smaller EMI",
            "❌ No — geographic factor",
            "❌ No — family planning is personal",
        ],
    })
    st.dataframe(factors, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Quick self-assessment
    st.markdown("### ✅ Quick Self-Assessment Checklist")
    st.markdown("Check how many of these you're doing right:")

    checks = [
        "I pay all EMIs and credit card bills before the due date",
        "My credit card usage is below 30% of the limit",
        "I haven't applied for new credit in the last 6 months",
        "I have at least one credit account older than 3 years",
        "I have a mix of credit types (card + loan)",
        "I've checked my CIBIL report in the last 12 months",
        "I have no outstanding defaults or write-offs",
        "My total debt is less than 40% of my annual income",
    ]

    score = 0
    for check in checks:
        if st.checkbox(check):
            score += 1

    if score > 0:
        st.markdown("---")
        if score >= 7:
            st.success(f"**{score}/8** — Excellent! You're doing most things right. "
                      "Your credit score should be in good shape.")
        elif score >= 5:
            st.info(f"**{score}/8** — Good foundation, but there's room to improve. "
                   "Focus on the unchecked items above.")
        elif score >= 3:
            st.warning(f"**{score}/8** — Fair. You have significant room for improvement. "
                      "Start with Tip 1 (pay on time) and Tip 2 (reduce utilization).")
        else:
            st.error(f"**{score}/8** — Needs work. But don't worry — consistent effort "
                    "over 6-12 months can dramatically improve your score. Start today.")


# ---- router ----
ROUTES = {
    "🏠 Overview": render_overview,
    "📊 EDA": render_eda,
    "🎯 Risk Prediction": render_predict,
    "🔍 Explainability": render_explain,
    "📜 Decision Rules": render_rules,
    "📊 Model Performance": render_model_performance,
    "💬 Talk-to-Data": render_chatbot,
    "📈 Improve Your Score": render_improve_score,
}
ROUTES[section]()
