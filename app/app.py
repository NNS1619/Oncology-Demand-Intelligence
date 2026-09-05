# ============================================================
# Patient-Informed Oncology Demand Forecasting App
# Streamlit + Scenario Intelligence + Project RAG
# ============================================================

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_SEARCH_DIRS = [
    PROJECT_ROOT / "data" / "outputs",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data",
    PROJECT_ROOT,
]

try:
    from rag.rag_utils import answer_question_with_rag, evidence_preview
    RAG_AVAILABLE = True
except Exception as import_error:
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = import_error


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Oncology Demand Intelligence",
    page_icon="💊",
    layout="wide",
)

st.title("Patient-Informed Oncology Demand Forecasting")
st.caption(
    "A synthetic pharmaceutical analytics POC for forecasting comparison, "
    "scenario planning, uncertainty, and evidence-grounded explanation."
)


# ------------------------------------------------------------
# File loading helpers
# ------------------------------------------------------------

def find_file(filename):
    for folder in DATA_SEARCH_DIRS:
        candidate = folder / filename
        if candidate.exists():
            return candidate

    return None


def read_csv_required(filename):
    path = find_file(filename)

    if path is None:
        raise FileNotFoundError(
            f"Required file not found: {filename}. "
            "Place it in data/outputs, data/processed, or data."
        )

    return pd.read_csv(path)


def read_csv_optional(filename):
    path = find_file(filename)

    if path is None:
        return None

    return pd.read_csv(path)


@st.cache_data
def load_project_outputs():
    data = {}

    data["compact_metrics"] = read_csv_required(
        "compact_model_comparison.csv"
    )

    data["metrics_by_horizon"] = read_csv_required(
        "metrics_by_horizon.csv"
    )

    data["metrics_by_horizon_regime"] = read_csv_required(
        "metrics_by_horizon_regime.csv"
    )

    data["forecast_results"] = read_csv_optional(
        "forecast_results.csv"
    )

    data["scenario_summary_with_uncertainty"] = read_csv_required(
        "scenario_summary_with_uncertainty.csv"
    )

    data["therapy_scenario_summary"] = read_csv_required(
        "therapy_scenario_summary.csv"
    )

    data["scenario_row_outputs"] = read_csv_optional(
        "scenario_row_outputs.csv"
    )

    data["scenario_catalog"] = read_csv_optional(
        "scenario_catalog.csv"
    )

    data["regime_positive_fva"] = read_csv_optional(
        "regime_positive_fva.csv"
    )

    data["claim_discipline"] = read_csv_optional(
        "claim_discipline.csv"
    )

    data["senior_review_gate"] = read_csv_optional(
        "notebook_3_senior_review_gate.csv"
    )

    data["notebook_3_conclusion"] = read_csv_optional(
        "notebook_3_conclusion.csv"
    )

    return data


try:
    outputs = load_project_outputs()
except Exception as error:
    st.error("The app could not load the required project output files.")
    st.exception(error)
    st.stop()


compact_metrics = outputs["compact_metrics"]
metrics_by_horizon = outputs["metrics_by_horizon"]
metrics_by_horizon_regime = outputs["metrics_by_horizon_regime"]
forecast_results = outputs["forecast_results"]
scenario_summary_with_uncertainty = outputs["scenario_summary_with_uncertainty"]
therapy_scenario_summary = outputs["therapy_scenario_summary"]
scenario_row_outputs = outputs["scenario_row_outputs"]
scenario_catalog = outputs["scenario_catalog"]
regime_positive_fva = outputs["regime_positive_fva"]
claim_discipline = outputs["claim_discipline"]
senior_review_gate = outputs["senior_review_gate"]
notebook_3_conclusion = outputs["notebook_3_conclusion"]


# ------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------

def fmt_units(value):
    if pd.isna(value):
        return "Not available"

    return f"{value:,.1f}"


def fmt_pct(value):
    if pd.isna(value):
        return "Not available"

    return f"{value:.1%}"


def format_percent_column(df, columns):
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")

    return out


SCENARIO_CLIENT_LABELS = {
    "Base case": "Base planning forecast",
    "Access downside": "Uniform access downside",
    "Earlier strong competitor pressure": "Stronger overlapping competitor pressure",
    "Epidemiology upside": "Eligible population upside",
    "Persistence downside": "Persistence downside",
    "Therapy D East supply constraint": "Therapy D East supply constraint",
    "Combined downside": "Combined downside planning case",
}


SCENARIO_CLIENT_EXPLANATIONS = {
    "Base case": (
        "Current governed planning forecast with no scenario adjustment."
    ),
    "Access downside": (
        "A simple stress test where reachable market access falls broadly."
    ),
    "Earlier strong competitor pressure": (
        "Stronger pressure against therapies with overlapping eligible patients."
    ),
    "Epidemiology upside": (
        "Higher eligible patient volume increases demand opportunity."
    ),
    "Persistence downside": (
        "Lower treatment continuation reduces patient-months and demand."
    ),
    "Therapy D East supply constraint": (
        "Observed sales are constrained for Therapy D in the East region."
    ),
    "Combined downside": (
        "Multiple risks are applied together to estimate downside exposure."
    ),
}


def add_client_scenario_labels(df):
    out = df.copy()

    if "scenario_name" in out.columns:
        out["scenario_label"] = out["scenario_name"].map(
            SCENARIO_CLIENT_LABELS
        ).fillna(out["scenario_name"])

        out["client_interpretation"] = out["scenario_name"].map(
            SCENARIO_CLIENT_EXPLANATIONS
        ).fillna("Scenario assumption from the saved output table.")

    return out


scenario_summary_display = add_client_scenario_labels(
    scenario_summary_with_uncertainty
)

therapy_scenario_display = add_client_scenario_labels(
    therapy_scenario_summary
)


# ------------------------------------------------------------
# Structured Analytics Router for RAG
# ------------------------------------------------------------

THERAPY_NAMES = {
    "therapy a": "Therapy A",
    "therapy b": "Therapy B",
    "therapy c": "Therapy C",
    "therapy d": "Therapy D",
}

SCENARIO_NAMES = {
    "base": "Base case",
    "base case": "Base case",
    "access": "Access downside",
    "access downside": "Access downside",
    "competitor": "Earlier strong competitor pressure",
    "competition": "Earlier strong competitor pressure",
    "earlier strong competitor pressure": "Earlier strong competitor pressure",
    "epidemiology": "Epidemiology upside",
    "epidemiology upside": "Epidemiology upside",
    "persistence": "Persistence downside",
    "persistence downside": "Persistence downside",
    "supply": "Therapy D East supply constraint",
    "supply constraint": "Therapy D East supply constraint",
    "therapy d east supply constraint": "Therapy D East supply constraint",
    "combined": "Combined downside",
    "combined downside": "Combined downside",
}


def detect_therapy(question):
    question_lower = question.lower()

    for key, therapy in THERAPY_NAMES.items():
        if key in question_lower:
            return therapy

    return None


def detect_scenario(question):
    question_lower = question.lower()

    for key in sorted(SCENARIO_NAMES.keys(), key=len, reverse=True):
        if key in question_lower:
            return SCENARIO_NAMES[key]

    return None


def detect_horizon(question):
    question_lower = question.lower()

    patterns = [
        r"(\d+)\s*month",
        r"(\d+)\s*months",
        r"horizon\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question_lower)

        if match:
            return int(match.group(1))

    return None


def question_mentions_structured_result(question):
    question_lower = question.lower()

    structured_terms = [
        "worst",
        "best",
        "biggest",
        "largest",
        "smallest",
        "highest",
        "lowest",
        "most affected",
        "least affected",
        "decline",
        "drop",
        "increase",
        "improve",
        "improvement",
        "added value",
        "add value",
        "fva",
        "wape",
        "mae",
        "bias",
        "which therapy",
        "which scenario",
        "which model",
        "when will",
        "when does",
        "p10",
        "p50",
        "p90",
        "uncertainty",
    ]

    return any(term in question_lower for term in structured_terms)


def build_therapy_scenario_context(question, therapy_scenario_summary):
    question_lower = question.lower()
    therapy = detect_therapy(question)
    scenario = detect_scenario(question)

    df = therapy_scenario_summary.copy()

    if therapy:
        df = df[df["therapy"] == therapy]

    if scenario:
        df = df[df["scenario_name"] == scenario]

    if df.empty:
        return None

    if any(
        term in question_lower
        for term in [
            "worst",
            "biggest decline",
            "largest decline",
            "drop",
            "most affected",
        ]
    ):
        selected = df.sort_values(
            "absolute_change_units",
            ascending=True,
        ).iloc[0]

        result_type = "largest negative change"

    elif any(
        term in question_lower
        for term in [
            "best",
            "biggest increase",
            "largest increase",
            "benefit",
            "upside",
        ]
    ):
        selected = df.sort_values(
            "absolute_change_units",
            ascending=False,
        ).iloc[0]

        result_type = "largest positive change"

    else:
        df["_abs_change"] = df["absolute_change_units"].abs()

        selected = df.sort_values(
            "_abs_change",
            ascending=False,
        ).iloc[0]

        result_type = "largest absolute change"

    context = f"""
STRUCTURED DATA RESULT:
Question type: Therapy-level scenario lookup
Result type: {result_type}

Therapy: {selected["therapy"]}
Scenario: {selected["scenario_name"]}
Client scenario label: {SCENARIO_CLIENT_LABELS.get(selected["scenario_name"], selected["scenario_name"])}

Baseline forecast units: {fmt_units(selected["baseline_forecast_units"])}
Scenario forecast units: {fmt_units(selected["scenario_forecast_units"])}
Absolute change units: {fmt_units(selected["absolute_change_units"])}
Percent change: {fmt_pct(selected["percent_change"])}

Interpretation instruction:
Explain that these values come from the saved therapy-level scenario output table.
Do not recalculate them.
Explain the commercial meaning in simple pharmaceutical planning language.
If the scenario is a uniform access downside, explain that similar percentage changes are expected by design and absolute unit impact is the more useful comparison.
"""

    return context


def build_overall_scenario_context(question, scenario_summary_with_uncertainty):
    question_lower = question.lower()
    scenario = detect_scenario(question)

    df = scenario_summary_with_uncertainty.copy()

    if scenario:
        df = df[df["scenario_name"] == scenario]

    if df.empty:
        return None

    if any(
        term in question_lower
        for term in [
            "worst",
            "biggest decline",
            "largest decline",
            "downside",
            "drop",
        ]
    ):
        selected = df.sort_values(
            "absolute_change_units",
            ascending=True,
        ).iloc[0]

        result_type = "largest overall downside"

    elif any(
        term in question_lower
        for term in [
            "best",
            "biggest increase",
            "largest increase",
            "upside",
        ]
    ):
        selected = df.sort_values(
            "absolute_change_units",
            ascending=False,
        ).iloc[0]

        result_type = "largest overall upside"

    else:
        selected = df.iloc[0]
        result_type = "selected scenario"

    context = f"""
STRUCTURED DATA RESULT:
Question type: Overall scenario lookup
Result type: {result_type}

Scenario: {selected["scenario_name"]}
Client scenario label: {SCENARIO_CLIENT_LABELS.get(selected["scenario_name"], selected["scenario_name"])}
Description: {selected.get("description", "not available")}

Baseline forecast units: {fmt_units(selected["baseline_forecast_units"])}
Scenario forecast units: {fmt_units(selected["scenario_forecast_units"])}
Absolute change units: {fmt_units(selected["absolute_change_units"])}
Percent change: {fmt_pct(selected["percent_change"])}

Conservative planning case, P10: {fmt_units(selected.get("p10"))}
Expected planning case, P50: {fmt_units(selected.get("p50"))}
Upside planning case, P90: {fmt_units(selected.get("p90"))}

Interpretation instruction:
Explain P10/P50/P90 in client-friendly language.
Clarify that these are simulated planning ranges from the scenario uncertainty engine, not clinical confidence intervals.
Do not recalculate the values.
"""

    return context


def build_model_performance_context(question, compact_metrics):
    question_lower = question.lower()
    horizon = detect_horizon(question)

    df = compact_metrics.copy()

    if horizon:
        df = df[df["horizon_months"] == horizon]

    if df.empty:
        return None

    if any(
        term in question_lower
        for term in ["worst", "highest wape", "least accurate"]
    ):
        selected = df.sort_values("WAPE", ascending=False).iloc[0]
        result_type = "worst model by WAPE"

    else:
        selected = df.sort_values("WAPE", ascending=True).iloc[0]
        result_type = "best model by WAPE"

    context = f"""
STRUCTURED DATA RESULT:
Question type: Model-performance lookup
Result type: {result_type}

Forecast horizon: {int(selected["horizon_months"])} months
Model: {selected["model"]}

WAPE: {fmt_pct(selected["WAPE"])}
MAE: {fmt_units(selected["MAE"])}
Bias: {fmt_pct(selected["Bias"])}

Interpretation instruction:
Explain that lower WAPE is better.
Explain why recent observed demand can be difficult to beat in a persistent oncology demand setting.
Do not claim that complex ML wins overall unless the structured result supports it.
"""

    return context


def build_regime_value_context(question, regime_positive_fva):
    question_lower = question.lower()

    if regime_positive_fva is None or regime_positive_fva.empty:
        return None

    model_terms = {
        "historical": "historical_xgboost",
        "patient driver": "patient_driver",
        "patient": "patient_driver",
        "hybrid": "hybrid_xgboost",
    }

    selected_model = None

    for term, model in model_terms.items():
        if term in question_lower:
            selected_model = model
            break

    if selected_model is None:
        return None

    df = regime_positive_fva.copy()

    fva_col = f"{selected_model}_fva_vs_naive"
    wape_col = f"{selected_model}_wape"

    if fva_col not in df.columns:
        return None

    df = df[df[fva_col] > 0].copy()

    if df.empty:
        return f"""
STRUCTURED DATA RESULT:
Question type: Regime Forecast Value Add lookup

Model checked: {selected_model}
Result: No positive Forecast Value Add windows were found for this model in the saved regime-positive table.

Interpretation instruction:
Explain that the selected model did not beat naive in the stored positive-FVA regime windows.
"""

    df = df.sort_values(fva_col, ascending=False)

    rows = []

    for _, row in df.head(5).iterrows():
        rows.append(
            f"- Horizon {int(row['horizon_months'])} months, "
            f"regime {row['target_market_regime']}: "
            f"{selected_model} WAPE {fmt_pct(row[wape_col])}, "
            f"naive WAPE {fmt_pct(row['naive_wape'])}, "
            f"FVA vs naive {fmt_pct(row[fva_col])}, "
            f"n forecasts {int(row['n_forecasts'])}"
        )

    context = f"""
STRUCTURED DATA RESULT:
Question type: Regime Forecast Value Add lookup

Model checked: {selected_model}
Positive value-add windows:
{chr(10).join(rows)}

Interpretation instruction:
Explain that patient/market or hybrid methods did not win overall, but did add value in selected event regimes.
Mention sample-size caution if n_forecasts is small.
"""

    return context


def build_structured_context_for_question(
    question,
    scenario_summary_with_uncertainty,
    therapy_scenario_summary,
    compact_metrics,
    regime_positive_fva=None,
):
    question_lower = question.lower()

    if not question_mentions_structured_result(question):
        return None

    if detect_therapy(question) is not None:
        context = build_therapy_scenario_context(
            question=question,
            therapy_scenario_summary=therapy_scenario_summary,
        )

        if context:
            return context

    if any(
        term in question_lower
        for term in [
            "model",
            "wape",
            "mae",
            "bias",
            "naive",
            "xgboost",
            "hybrid",
        ]
    ):
        context = build_model_performance_context(
            question=question,
            compact_metrics=compact_metrics,
        )

        if context:
            return context

    if any(
        term in question_lower
        for term in [
            "fva",
            "added value",
            "add value",
            "regime",
            "where does",
        ]
    ):
        context = build_regime_value_context(
            question=question,
            regime_positive_fva=regime_positive_fva,
        )

        if context:
            return context

    if any(
        term in question_lower
        for term in [
            "scenario",
            "p10",
            "p50",
            "p90",
            "uncertainty",
            "downside",
            "upside",
        ]
    ):
        context = build_overall_scenario_context(
            question=question,
            scenario_summary_with_uncertainty=scenario_summary_with_uncertainty,
        )

        if context:
            return context

    return None


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tab_summary, tab_performance, tab_scenarios, tab_rag, tab_governance = st.tabs(
    [
        "Executive Summary",
        "Model Performance",
        "Scenario Intelligence",
        "Project RAG",
        "Governance",
    ]
)


# ------------------------------------------------------------
# Executive Summary
# ------------------------------------------------------------

with tab_summary:
    st.header("Executive Summary")

    st.markdown(
        """
This project is a synthetic pharmaceutical analytics POC. It tests when a forecaster
should rely on recent observed demand and when patient or market signals add value.

The main finding is intentionally realistic: **recent demand was the strongest overall
forecasting benchmark**, because treated oncology demand is persistent. The patient and
market layer is most useful for **event interpretation and scenario planning**, not for
beating naive forecasting in every window.
"""
    )

    col1, col2, col3 = st.columns(3)

    total_forecasts = len(forecast_results) if forecast_results is not None else np.nan
    total_scenarios = scenario_summary_with_uncertainty["scenario_name"].nunique()
    total_therapies = therapy_scenario_summary["therapy"].nunique()

    col1.metric("Forecast Rows Evaluated", fmt_units(total_forecasts))
    col2.metric("Planning Scenarios", int(total_scenarios))
    col3.metric("Therapies", int(total_therapies))

    st.subheader("What The Project Proves")

    st.markdown(
        """
- Simple baselines matter. Complex models must earn their place.
- Forecast Information Set rules prevent future-data leakage.
- Patient and market signals help explain access, competition, persistence, epidemiology, and supply risks.
- Scenario intelligence creates the most client-facing value.
"""
    )

    if notebook_3_conclusion is not None:
        st.subheader("Notebook 3 Conclusion")
        st.dataframe(notebook_3_conclusion, use_container_width=True)


# ------------------------------------------------------------
# Model Performance
# ------------------------------------------------------------

with tab_performance:
    st.header("Model Performance")

    st.caption(
        "Lower WAPE and MAE are better. Bias below zero means under-forecasting."
    )

    display_metrics = compact_metrics.copy()

    st.subheader("Compact Model Comparison")
    st.dataframe(
        format_percent_column(display_metrics, ["WAPE", "Bias"]),
        use_container_width=True,
    )

    fig = px.line(
        compact_metrics,
        x="horizon_months",
        y="WAPE",
        color="model",
        markers=True,
        title="Forecast Accuracy by Horizon",
        labels={
            "horizon_months": "Forecast Horizon",
            "WAPE": "WAPE",
            "model": "Model",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    best_by_horizon = (
        compact_metrics
        .sort_values(["horizon_months", "WAPE"])
        .groupby("horizon_months", as_index=False)
        .first()
    )

    st.subheader("Best Model By Forecast Horizon")
    st.dataframe(
        format_percent_column(best_by_horizon, ["WAPE", "Bias"]),
        use_container_width=True,
    )

    st.info(
        "Interpretation: naive demand wins overall because recent sales are a strong "
        "proxy for active treated patient stock in a persistent oncology market. "
        "That is a defensible result, not a project failure."
    )

    if regime_positive_fva is not None:
        st.subheader("Event Windows Where Non-Naive Methods Added Value")
        st.caption(
            "This table is the more nuanced story: patient or hybrid signals can help "
            "inside specific market regimes even when naive wins overall."
        )
        st.dataframe(regime_positive_fva, use_container_width=True)


# ------------------------------------------------------------
# Scenario Intelligence
# ------------------------------------------------------------

with tab_scenarios:
    st.header("Scenario Intelligence")

    scenario_names = list(
        scenario_summary_with_uncertainty["scenario_name"].dropna().unique()
    )

    selected_scenario = st.selectbox(
        "Select scenario",
        scenario_names,
        format_func=lambda x: SCENARIO_CLIENT_LABELS.get(x, x),
    )

    selected_summary = scenario_summary_with_uncertainty[
        scenario_summary_with_uncertainty["scenario_name"] == selected_scenario
    ].iloc[0]

    st.subheader(SCENARIO_CLIENT_LABELS.get(selected_scenario, selected_scenario))
    st.caption(
        SCENARIO_CLIENT_EXPLANATIONS.get(
            selected_scenario,
            "Scenario result from the saved output table.",
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Baseline Demand Forecast",
        fmt_units(selected_summary["baseline_forecast_units"]),
    )

    col2.metric(
        "Scenario Demand Forecast",
        fmt_units(selected_summary["scenario_forecast_units"]),
    )

    col3.metric(
        "Change vs Baseline",
        fmt_units(selected_summary["absolute_change_units"]),
        delta=fmt_pct(selected_summary["percent_change"]),
    )

    col4.metric(
        "Expected Planning Case",
        fmt_units(selected_summary.get("p50")),
    )

    st.subheader("Planning Range")

    range_cols = st.columns(3)

    range_cols[0].metric(
        "Conservative Planning Case",
        fmt_units(selected_summary.get("p10")),
    )

    range_cols[1].metric(
        "Expected Planning Case",
        fmt_units(selected_summary.get("p50")),
    )

    range_cols[2].metric(
        "Upside Planning Case",
        fmt_units(selected_summary.get("p90")),
    )

    st.caption(
        "P10, P50, and P90 are simulated scenario planning ranges. "
        "They should be read as conservative, expected, and upside planning cases, "
        "not as clinical confidence intervals."
    )

    selected_therapy_rows = therapy_scenario_display[
        therapy_scenario_display["scenario_name"] == selected_scenario
    ].copy()

    st.subheader("Therapy-Level Impact")
    st.dataframe(
        format_percent_column(
            selected_therapy_rows[
                [
                    "scenario_label",
                    "therapy",
                    "baseline_forecast_units",
                    "scenario_forecast_units",
                    "absolute_change_units",
                    "percent_change",
                ]
            ],
            ["percent_change"],
        ),
        use_container_width=True,
    )

    fig = px.bar(
        selected_therapy_rows,
        x="therapy",
        y="absolute_change_units",
        title="Therapy-Level Change vs Baseline",
        labels={
            "therapy": "Therapy",
            "absolute_change_units": "Change in Forecast Units",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    if selected_scenario == "Access downside":
        st.info(
            "For a uniform access downside, similar percentage declines across "
            "therapies are expected by design. The more useful client insight is "
            "absolute unit exposure: high-volume therapies carry the largest "
            "commercial risk."
        )

    scenario_chart_df = scenario_summary_display[
        scenario_summary_display["scenario_name"] != "Base case"
    ].copy()

    st.subheader("Scenario Impact vs Baseline")

    fig = px.bar(
        scenario_chart_df.sort_values("absolute_change_units"),
        x="absolute_change_units",
        y="scenario_label",
        orientation="h",
        title="Portfolio-Level Scenario Impact",
        labels={
            "absolute_change_units": "Change in Forecast Units",
            "scenario_label": "Scenario",
        },
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Project RAG
# ------------------------------------------------------------

with tab_rag:
    st.header("Project RAG Assistant")

    st.caption(
        "Ask questions about the forecasting method, results, scenarios, uncertainty, "
        "assumptions, leakage controls, limitations, or RAG boundary. Numerical answers "
        "come from saved output tables; explanation comes from retrieved project evidence."
    )

    if not RAG_AVAILABLE:
        st.error("RAG utilities could not be imported.")
        st.exception(RAG_IMPORT_ERROR)
        st.stop()

    user_question = st.text_area(
        "Project question",
        value=(
            "Why was recent demand the strongest overall benchmark, "
            "and what limitations should we keep in mind?"
        ),
        height=90,
    )

    include_selected_scenario = st.checkbox(
        "Include selected scenario numbers when relevant",
        value=False,
    )

    rag_selected_scenario = st.selectbox(
        "Optional scenario context",
        scenario_names,
        format_func=lambda x: SCENARIO_CLIENT_LABELS.get(x, x),
        key="rag_scenario_select",
    )

    col1, col2 = st.columns(2)

    with col1:
        preview_clicked = st.button("Preview Retrieved Evidence")

    with col2:
        answer_clicked = st.button("Answer with Project RAG")

    if preview_clicked:
        if not user_question.strip():
            st.warning("Please enter a project question first.")

        else:
            with st.spinner("Retrieving evidence..."):
                preview = evidence_preview(
                    query=user_question,
                    k=4,
                )

            st.subheader("Retrieved Evidence Preview")
            st.text(preview)

    if answer_clicked:
        if not user_question.strip():
            st.warning("Please enter a project question.")

        else:
            structured_context = build_structured_context_for_question(
                question=user_question,
                scenario_summary_with_uncertainty=scenario_summary_with_uncertainty,
                therapy_scenario_summary=therapy_scenario_summary,
                compact_metrics=compact_metrics,
                regime_positive_fva=regime_positive_fva,
            )

            if include_selected_scenario and structured_context is None:
                scenario_question = f"Explain scenario {rag_selected_scenario}"

                structured_context = build_overall_scenario_context(
                    question=scenario_question,
                    scenario_summary_with_uncertainty=scenario_summary_with_uncertainty,
                )

            with st.spinner("Retrieving evidence and generating grounded answer..."):
                try:
                    result = answer_question_with_rag(
                        user_question=user_question,
                        structured_context=structured_context,
                        k=5,
                    )

                    st.subheader("Grounded Answer")
                    st.markdown(result["answer"])

                    if result.get("model_used"):
                        st.caption(f"Model used: {result['model_used']}")

                    if result.get("sources"):
                        st.caption(
                            "Evidence sources: "
                            + ", ".join(result["sources"])
                        )

                    if structured_context:
                        with st.expander("Structured data used for this answer"):
                            st.text(structured_context)

                except Exception as error:
                    st.error(
                        "The RAG answer could not be generated. "
                        "Check Streamlit secrets, Gemini model access, and vector store setup."
                    )
                    st.exception(error)

                    if structured_context:
                        st.subheader("Structured Result Available")
                        st.text(structured_context)


# ------------------------------------------------------------
# Governance
# ------------------------------------------------------------

with tab_governance:
    st.header("Governance and Claim Discipline")

    st.markdown(
        """
This app separates **calculation** from **explanation**.

- The notebooks and CSV outputs are the source of numerical truth.
- The structured router finds relevant saved results.
- RAG retrieves methodology and evidence documents.
- Gemini explains the result but does not create official numbers.
"""
    )

    if claim_discipline is not None:
        st.subheader("Claim Discipline")
        st.dataframe(claim_discipline, use_container_width=True)

    if senior_review_gate is not None:
        st.subheader("Senior Review Gate")
        st.dataframe(senior_review_gate, use_container_width=True)

    st.subheader("What This Project Can Claim")

    st.success(
        "Recent demand was the strongest overall benchmark in this synthetic oncology "
        "market, while patient and market signals added value for event interpretation "
        "and scenario planning."
    )

    st.subheader("Disclaimer")

    st.warning(
        "We do not claim clinical validation, real-world oncology proof, "
        "or that hybrid ML beats simpler methods across all horizons."
    )
