# ============================================================
# Patient-Informed Oncology Demand Forecasting App
# Streamlit + Scenario Intelligence + Project RAG
# ============================================================

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
    from rag.structured_router import (
        build_structured_context_for_question as route_structured_context,
    )
except Exception:
    route_structured_context = None

try:
    from rag.rag_utils import answer_question_with_rag, evidence_preview
    RAG_AVAILABLE = True
except Exception as import_error:
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = import_error

    def answer_question_with_rag(*args, **kwargs):
        return {
            "answer": (
                "The AI explanation service is temporarily unavailable. "
                "The validated forecast and scenario dashboards remain available."
            ),
            "model_used": "safe_fallback",
            "sources": [],
            "status": "rag_import_unavailable",
            "error_type": type(RAG_IMPORT_ERROR).__name__,
        }

    def evidence_preview(*args, **kwargs):
        return (
            "Evidence retrieval is temporarily unavailable. "
            "The numerical dashboards are unaffected."
        )


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

    data["forecast_metrics_by_therapy_horizon"] = read_csv_optional(
        "forecast_metrics_by_therapy_horizon.csv"
    )

    data["forecast_macro_metrics_by_horizon"] = read_csv_optional(
        "forecast_macro_metrics_by_horizon.csv"
    )

    data["forecast_model_bootstrap_intervals"] = read_csv_optional(
        "forecast_model_bootstrap_intervals.csv"
    )

    data["rag_evaluation_summary"] = read_csv_optional(
        "rag_evaluation_summary.csv"
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
forecast_metrics_by_therapy_horizon = outputs["forecast_metrics_by_therapy_horizon"]
forecast_macro_metrics_by_horizon = outputs["forecast_macro_metrics_by_horizon"]
forecast_model_bootstrap_intervals = outputs["forecast_model_bootstrap_intervals"]
rag_evaluation_summary = outputs["rag_evaluation_summary"]
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


def evaluation_metric(summary_df, metric_name):
    if summary_df is None or summary_df.empty:
        return None

    match = summary_df.loc[summary_df["metric"] == metric_name, "value"]
    if match.empty or pd.isna(match.iloc[0]):
        return None

    return float(match.iloc[0])


def fmt_evaluation_rate(value):
    return "Not run" if value is None else f"{value:.0%}"


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

    if forecast_macro_metrics_by_horizon is not None:
        with st.expander("Therapy-balanced performance view"):
            st.caption(
                "Portfolio WAPE weights high-volume therapies more heavily. Macro metrics "
                "average the four therapies equally, so they reveal whether the conclusion "
                "changes when every therapy receives the same weight."
            )
            macro_display = forecast_macro_metrics_by_horizon.copy()
            macro_display = format_percent_column(
                macro_display,
                ["macro_WAPE", "macro_Bias"],
            )
            st.dataframe(macro_display, use_container_width=True)

    if forecast_metrics_by_therapy_horizon is not None:
        with st.expander("Therapy-level forecast metrics"):
            therapy_metric_display = format_percent_column(
                forecast_metrics_by_therapy_horizon,
                ["WAPE", "Bias"],
            )
            st.dataframe(therapy_metric_display, use_container_width=True)

    if forecast_model_bootstrap_intervals is not None:
        with st.expander("Forecast-origin bootstrap evidence"):
            st.caption(
                "Each bootstrap sample resamples complete forecast origins, preserving the "
                "16 therapy-region forecasts within an origin. The interval tests whether a "
                "model's portfolio WAPE improvement over naive is stable across origins."
            )
            bootstrap_display = forecast_model_bootstrap_intervals[
                [
                    "horizon_months",
                    "model",
                    "n_origins",
                    "FVA_vs_naive",
                    "FVA_ci_lower",
                    "FVA_ci_upper",
                    "probability_model_beats_naive",
                ]
            ].copy()
            bootstrap_display = format_percent_column(
                bootstrap_display,
                [
                    "FVA_vs_naive",
                    "FVA_ci_lower",
                    "FVA_ci_upper",
                    "probability_model_beats_naive",
                ],
            )
            st.dataframe(bootstrap_display, use_container_width=True)


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

    st.subheader("AI Trust and Evaluation")
    if rag_evaluation_summary is None:
        st.info(
            "No saved RAG evaluation summary is available yet. Run "
            "`python rag/evaluate_rag.py --mode full` and commit the generated summary."
        )
    else:
        trust_cols = st.columns(5)
        trust_cols[0].metric(
            "Questions Tested",
            int(evaluation_metric(rag_evaluation_summary, "questions_evaluated") or 0),
        )
        trust_cols[1].metric(
            "Router Accuracy",
            fmt_evaluation_rate(
                evaluation_metric(rag_evaluation_summary, "router_intent_accuracy")
            ),
        )
        trust_cols[2].metric(
            "Retrieval Recall@5",
            fmt_evaluation_rate(
                evaluation_metric(rag_evaluation_summary, "source_recall_at_k")
            ),
        )
        trust_cols[3].metric(
            "Numerical Fidelity",
            fmt_evaluation_rate(
                evaluation_metric(rag_evaluation_summary, "numerical_exactness")
            ),
        )
        trust_cols[4].metric(
            "Boundary Accuracy",
            fmt_evaluation_rate(
                evaluation_metric(rag_evaluation_summary, "refusal_accuracy")
            ),
        )

        evaluation_mode = str(rag_evaluation_summary["evaluation_mode"].iloc[0])
        evaluation_split = (
            str(rag_evaluation_summary["evaluation_split"].iloc[0])
            if "evaluation_split" in rag_evaluation_summary.columns
            else "all"
        )
        generated_at = str(rag_evaluation_summary["generated_at_utc"].iloc[0])
        st.caption(
            f"Saved evaluation mode: {evaluation_mode}; split: {evaluation_split}. "
            f"Generated: {generated_at}. "
            "'Not run' means the API-dependent layer has not yet been measured; it is not a zero score."
        )

        with st.expander("Evaluation details and acceptance targets"):
            st.dataframe(rag_evaluation_summary, use_container_width=True)

    if not RAG_AVAILABLE:
        st.warning(
            "The AI explanation service is temporarily unavailable. "
            "The forecast and scenario tabs remain fully usable."
        )

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
            if route_structured_context is not None:
                structured_context = route_structured_context(
                    question=user_question,
                    scenario_summary_with_uncertainty=scenario_summary_with_uncertainty,
                    therapy_scenario_summary=therapy_scenario_summary,
                    compact_metrics=compact_metrics,
                    regime_positive_fva=regime_positive_fva,
                )
            else:
                structured_context = None

            if include_selected_scenario and structured_context is None:
                scenario_question = f"Explain scenario {rag_selected_scenario}"
                if route_structured_context is not None:
                    structured_context = route_structured_context(
                        question=scenario_question,
                        scenario_summary_with_uncertainty=scenario_summary_with_uncertainty,
                        therapy_scenario_summary=therapy_scenario_summary,
                        compact_metrics=compact_metrics,
                        regime_positive_fva=regime_positive_fva,
                    )

            with st.spinner("Retrieving evidence and generating grounded answer..."):
                try:
                    result = answer_question_with_rag(
                        user_question=user_question,
                        structured_context=structured_context,
                        k=5,
                    )

                    st.subheader("Grounded Answer")
                    if result.get("status") in {
                        "retrieval_unavailable",
                        "generation_unavailable",
                        "rag_import_unavailable",
                    }:
                        st.warning(result["answer"])
                    else:
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

                except Exception:
                    st.error(
                        "The generated explanation is temporarily unavailable. "
                        "Validated numerical outputs remain available in the dashboard."
                    )

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
