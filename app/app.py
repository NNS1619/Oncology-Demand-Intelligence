# ============================================================
# Patient-Informed Oncology Demand Forecasting
# Streamlit Decision-Support App
# ============================================================

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------
# App configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Oncology Demand Intelligence",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

try:
    from rag.rag_utils import evidence_preview, explain_scenario_with_rag
    RAG_AVAILABLE = True
    RAG_IMPORT_ERROR = None
except Exception as error:
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = str(error)
DATA_RAW = PROJECT_ROOT / "data" / "raw_synthetic"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_OUTPUTS = PROJECT_ROOT / "data" / "outputs"
RAG_DOCS = PROJECT_ROOT / "rag" / "evidence_docs"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.1%}"


def num(x):
    if pd.isna(x):
        return "NA"
    return f"{x:,.1f}"


def clean_model_name(name):
    return (
        str(name)
        .replace("_", " ")
        .replace("xgboost", "XGBoost")
        .title()
        .replace("Fva", "FVA")
    )


def require_data(df, name):
    if df.empty:
        st.error(f"Missing required file: {name}")
        st.stop()


def safe_columns(df, columns):
    return [col for col in columns if col in df.columns]

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
    "Base case": "Current governed planning forecast with no scenario adjustment.",
    "Access downside": "A simple stress test where market access falls equally across therapies and regions.",
    "Earlier strong competitor pressure": "Tests stronger competitive pressure against therapies serving overlapping biomarker-positive patients.",
    "Epidemiology upside": "Tests higher eligible patient volume.",
    "Persistence downside": "Tests lower treatment continuation and fewer active patient-months.",
    "Therapy D East supply constraint": "Tests limited observed sales for Therapy D in East only.",
    "Combined downside": "Tests multiple planning risks happening together.",
}


def add_client_scenario_labels(df):
    out = df.copy()

    out["client_scenario_name"] = out["scenario_name"].map(
        SCENARIO_CLIENT_LABELS
    ).fillna(out["scenario_name"])

    out["client_explanation"] = out["scenario_name"].map(
        SCENARIO_CLIENT_EXPLANATIONS
    ).fillna(out.get("description", ""))

    return out
# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

assumption_registry = load_csv(DATA_RAW / "assumption_registry.csv")
feature_provenance = load_csv(DATA_PROCESSED / "feature_provenance.csv")
feature_availability_audit = load_csv(DATA_PROCESSED / "feature_availability_audit.csv")

forecast_results = load_csv(DATA_OUTPUTS / "forecast_results.csv")
compact_metrics = load_csv(DATA_OUTPUTS / "compact_model_comparison.csv")
metrics_by_horizon = load_csv(DATA_OUTPUTS / "metrics_by_horizon.csv")
metrics_by_horizon_regime = load_csv(DATA_OUTPUTS / "metrics_by_horizon_regime.csv")
regime_positive_fva = load_csv(DATA_OUTPUTS / "regime_positive_fva.csv")

scenario_catalog = load_csv(DATA_OUTPUTS / "scenario_catalog.csv")
scenario_summary = load_csv(DATA_OUTPUTS / "scenario_summary_with_uncertainty.csv")
therapy_scenario_summary = load_csv(DATA_OUTPUTS / "therapy_scenario_summary.csv")

claim_discipline = load_csv(DATA_OUTPUTS / "claim_discipline.csv")
senior_review_gate = load_csv(DATA_OUTPUTS / "notebook_3_senior_review_gate.csv")
notebook_3_conclusion = load_csv(DATA_OUTPUTS / "notebook_3_conclusion.csv")

require_data(forecast_results, "data/outputs/forecast_results.csv")
require_data(compact_metrics, "data/outputs/compact_model_comparison.csv")
require_data(scenario_summary, "data/outputs/scenario_summary_with_uncertainty.csv")


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("Controls ⚙️")

available_horizons = sorted(forecast_results["horizon_months"].dropna().unique())
available_therapies = sorted(forecast_results["therapy"].dropna().unique())
available_regions = sorted(forecast_results["region"].dropna().unique())

selected_horizon = st.sidebar.selectbox(
    "Forecast horizon",
    available_horizons,
    index=min(2, len(available_horizons) - 1),
)

selected_therapy = st.sidebar.selectbox(
    "Therapy",
    available_therapies,
)

selected_region = st.sidebar.selectbox(
    "Region",
    available_regions,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This app reads saved notebook outputs. It does not retrain models."
)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("Patient-Informed Oncology Demand Forecasting")
st.caption(
    "Synthetic pharmaceutical demand forecasting and scenario-intelligence POC"
)

st.markdown(
    """
This POC evaluates when recent historical demand is enough, and when patient-flow,
market access, epidemiology, persistence, supply, and competition signals add value
for pharmaceutical planning.
"""
)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tab_summary, tab_forecast, tab_performance, tab_scenarios, tab_evidence, tab_governance = st.tabs(
    [
        "Executive Summary",
        "Forecast",
        "Model Performance",
        "Scenario Intelligence",
        "Evidence & Explanation",
        "Governance",
    ]
)


# ------------------------------------------------------------
# Tab 1: Executive Summary
# ------------------------------------------------------------

with tab_summary:
    st.subheader("Decision Summary")

    if not notebook_3_conclusion.empty:
        for _, row in notebook_3_conclusion.iterrows():
            st.markdown(f"**{row['question']}**")
            st.write(row["answer"])
    else:
        st.write(
            "Recent demand was strongest overall, while patient and market signals "
            "added value mainly for event-regime interpretation and scenario planning."
        )

    st.markdown("### Headline Findings")

    col1, col2, col3 = st.columns(3)

    naive_12 = compact_metrics.loc[
        (compact_metrics["horizon_months"] == 12)
        & (compact_metrics["model"] == "naive"),
        "WAPE",
    ]

    hybrid_3_persistence = pd.Series(dtype=float)

    if not regime_positive_fva.empty and "target_market_regime" in regime_positive_fva.columns:
        hybrid_3_persistence = regime_positive_fva.loc[
            (regime_positive_fva["horizon_months"] == 3)
            & (regime_positive_fva["target_market_regime"] == "persistence_change"),
            "hybrid_xgboost_fva_vs_naive",
        ]

    combined_downside = scenario_summary.loc[
        scenario_summary["scenario_name"] == "Combined downside",
        "percent_change",
    ]

    col1.metric(
        "Best Overall Benchmark",
        "Naive",
        "Recent demand dominated routine forecasting",
    )

    col2.metric(
        "12-Month Naive WAPE",
        pct(float(naive_12.iloc[0])) if len(naive_12) else "NA",
    )

    col3.metric(
        "Combined Downside Impact",
        pct(float(combined_downside.iloc[0])) if len(combined_downside) else "NA",
    )

    if len(hybrid_3_persistence):
        st.info(
            "Hybrid XGBoost added value in the 3-month persistence-change regime "
            f"with FVA vs naive of {pct(float(hybrid_3_persistence.iloc[0]))}."
        )

    st.markdown("### What This POC Is Actually For")

    st.write(
        """
The point is not to prove that one complex model always wins. The point is to
separate routine forecasting from planning intelligence. In stable periods,
recent sales can be very hard to beat. But for access changes, persistence shifts,
competition, epidemiology changes, and supply constraints, patient-informed
scenario analysis gives decision-makers a way to understand risk.
"""
    )


# ------------------------------------------------------------
# Tab 2: Forecast
# ------------------------------------------------------------

with tab_forecast:
    st.subheader("Forecast Explorer")

    filtered_forecast = forecast_results.loc[
        (forecast_results["horizon_months"] == selected_horizon)
        & (forecast_results["therapy"] == selected_therapy)
        & (forecast_results["region"] == selected_region)
    ].copy()

    filtered_forecast = filtered_forecast.sort_values("target_month_index")

    prediction_columns = [
        "actual_sales_units",
        "naive_prediction",
        "seasonal_naive_prediction",
        "historical_xgboost_prediction",
        "patient_driver_prediction",
        "hybrid_xgboost_prediction",
    ]

    plot_columns = safe_columns(filtered_forecast, prediction_columns)

    if filtered_forecast.empty:
        st.warning("No forecast rows found for this selection.")
    else:
        long_forecast = filtered_forecast[
            ["target_month_index"] + plot_columns
        ].melt(
            id_vars="target_month_index",
            var_name="series",
            value_name="units",
        )

        long_forecast["series"] = long_forecast["series"].map(clean_model_name)

        fig = px.line(
            long_forecast,
            x="target_month_index",
            y="units",
            color="series",
            markers=True,
            title=f"{selected_therapy} | {selected_region} | {int(selected_horizon)}-Month Horizon",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            filtered_forecast[
                safe_columns(
                    filtered_forecast,
                    [
                        "forecast_origin_month_index",
                        "target_month_index",
                        "actual_sales_units",
                        "naive_prediction",
                        "historical_xgboost_prediction",
                        "patient_driver_prediction",
                        "hybrid_xgboost_prediction",
                        "target_market_regime",
                    ],
                )
            ].round(3),
            use_container_width=True,
        )


# ------------------------------------------------------------
# Tab 3: Model Performance
# ------------------------------------------------------------

with tab_performance:
    st.subheader("Model Performance and Forecast Value Add")

    st.markdown(
        """
WAPE, MAE, and bias are evaluated using rolling-origin validation. Forecast Value
Add compares each method against the naive baseline.
"""
    )

    fig = px.line(
        compact_metrics,
        x="horizon_months",
        y="WAPE",
        color="model",
        markers=True,
        title="WAPE by Forecast Horizon",
    )

    fig.update_layout(
        xaxis_title="Forecast Horizon",
        yaxis_title="WAPE",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Compact Model Comparison")
    st.dataframe(compact_metrics.round(4), use_container_width=True)

    st.markdown("### Regime Windows with Positive FVA")
    if regime_positive_fva.empty:
        st.info("No regime-level positive FVA file found.")
    else:
        st.dataframe(regime_positive_fva.round(4), use_container_width=True)

    st.markdown("### Performance by Horizon and Regime")
    if metrics_by_horizon_regime.empty:
        st.info("Regime-level performance file not found.")
    else:
        st.dataframe(metrics_by_horizon_regime.round(4), use_container_width=True)


# ------------------------------------------------------------
# Tab 4: Scenario Intelligence
# ------------------------------------------------------------

with tab_scenarios:
    st.subheader("Scenario Intelligence")

    st.markdown(
        """
Scenario analysis answers planning questions that model accuracy alone cannot answer.
The numerical engine has already calculated these results. This tab translates them
into business planning language.
"""
    )

    scenario_summary_display = add_client_scenario_labels(scenario_summary)

    scenario_options = scenario_summary_display[
        ["scenario_name", "client_scenario_name"]
    ].drop_duplicates()

    selected_client_scenario = st.selectbox(
        "Planning scenario",
        scenario_options["client_scenario_name"].tolist(),
        index=0,
    )

    selected_scenario = scenario_options.loc[
        scenario_options["client_scenario_name"] == selected_client_scenario,
        "scenario_name",
    ].iloc[0]

    selected_scenario_row = scenario_summary_display.loc[
        scenario_summary_display["scenario_name"] == selected_scenario
    ].iloc[0]

    st.info(selected_scenario_row["client_explanation"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Baseline Demand Forecast",
        num(selected_scenario_row["baseline_forecast_units"]),
    )

    col2.metric(
        "Scenario Demand Forecast",
        num(selected_scenario_row["scenario_forecast_units"]),
        delta=num(selected_scenario_row["absolute_change_units"]),
    )

    col3.metric(
        "Change vs Baseline",
        pct(selected_scenario_row["percent_change"]),
    )

    st.markdown("### Planning Range")

    st.caption(
        "Conservative, expected, and upside planning estimates correspond to "
        "P10, P50, and P90 from the Monte Carlo uncertainty simulation."
    )

    range_col1, range_col2, range_col3 = st.columns(3)

    range_col1.metric(
        "Conservative Planning Case",
        num(selected_scenario_row["p10"]),
        help="P10: 10% of simulated outcomes were below this value.",
    )

    range_col2.metric(
        "Expected Planning Case",
        num(selected_scenario_row["p50"]),
        help="P50: median simulated outcome.",
    )

    range_col3.metric(
        "Upside Planning Case",
        num(selected_scenario_row["p90"]),
        help="P90: 90% of simulated outcomes were below this value.",
    )

    plot_df = scenario_summary_display.loc[
        scenario_summary_display["scenario_name"] != "Base case"
    ].copy()

    fig = px.bar(
        plot_df,
        x="absolute_change_units",
        y="client_scenario_name",
        orientation="h",
        title="Scenario Impact vs Baseline Demand Forecast",
        labels={
            "absolute_change_units": "Change in forecasted units",
            "client_scenario_name": "Planning scenario",
        },
    )

    fig.add_vline(x=0, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

    fig_uncertainty = go.Figure()

    fig_uncertainty.add_trace(
        go.Scatter(
            x=scenario_summary_display["client_scenario_name"],
            y=scenario_summary_display["p50"],
            mode="markers",
            error_y=dict(
                type="data",
                symmetric=False,
                array=scenario_summary_display["p90"] - scenario_summary_display["p50"],
                arrayminus=scenario_summary_display["p50"] - scenario_summary_display["p10"],
            ),
            name="Planning range",
        )
    )

    fig_uncertainty.update_layout(
        title="Scenario Planning Range",
        xaxis_title="Planning scenario",
        yaxis_title="Forecasted demand units",
    )

    st.plotly_chart(fig_uncertainty, use_container_width=True)

    st.markdown("### Therapy-Level Impact")

    if not therapy_scenario_summary.empty:
        therapy_rows = therapy_scenario_summary.loc[
            therapy_scenario_summary["scenario_name"] == selected_scenario
        ].copy()

        therapy_rows = therapy_rows.rename(
            columns={
                "therapy": "Therapy",
                "baseline_forecast_units": "Baseline forecast",
                "scenario_forecast_units": "Scenario forecast",
                "absolute_change_units": "Change in units",
                "percent_change": "Percent change",
            }
        )

        st.dataframe(
            therapy_rows.round(4),
            use_container_width=True,
        )

        st.caption(
            "For uniform access downside, percentage changes are expected to look similar "
            "across therapies because the same access multiplier is applied broadly. "
            "The more useful business signal is which therapy carries the largest absolute unit impact."
        )
    else:
        st.info("Therapy-level scenario output file not found.")

# ------------------------------------------------------------
# Tab 5: Evidence & Explanation
# ------------------------------------------------------------

with tab_evidence:
    st.subheader("Evidence & Explanation")

    st.markdown(
        """
The numerical engine has already calculated the forecasts and scenarios.
The RAG layer retrieves relevant assumption and methodology evidence.
The LLM explains the result, but it does not calculate or change official values.
"""
    )

    st.markdown("### RAG Boundary")

    boundary_table = pd.DataFrame(
        [
            {
                "Component": "Numerical engine",
                "Owns": "Forecasts, scenario calculations, uncertainty outputs",
                "Does not own": "Evidence retrieval or free-text explanation",
            },
            {
                "Component": "RAG retriever",
                "Owns": "Finding relevant assumption and methodology evidence",
                "Does not own": "Calculating forecast values",
            },
            {
                "Component": "LLM",
                "Owns": "Grounded business explanation",
                "Does not own": "Changing model outputs",
            },
        ]
    )

    st.dataframe(boundary_table, use_container_width=True)

    st.markdown("### Explain a Scenario")

    selected_for_explanation = st.selectbox(
        "Choose scenario to explain",
        scenario_summary["scenario_name"].tolist(),
        key="explain_scenario",
    )

    scenario_context_row = scenario_summary.loc[
        scenario_summary["scenario_name"] == selected_for_explanation
    ].iloc[0]

    therapy_context = ""

    if not therapy_scenario_summary.empty:
        therapy_rows = therapy_scenario_summary.loc[
            therapy_scenario_summary["scenario_name"] == selected_for_explanation
        ].copy()

        therapy_context = therapy_rows[
            [
                "therapy",
                "baseline_forecast_units",
                "scenario_forecast_units",
                "absolute_change_units",
                "percent_change",
            ]
        ].round(3).to_string(index=False)

    scenario_context = f"""
Scenario: {scenario_context_row['scenario_name']}
Description: {scenario_context_row.get('description', '')}

Baseline forecast: {scenario_context_row['baseline_forecast_units']:.1f} units
Scenario forecast: {scenario_context_row['scenario_forecast_units']:.1f} units
Absolute change: {scenario_context_row['absolute_change_units']:.1f} units
Percent change: {scenario_context_row['percent_change']:.1%}

Uncertainty:
P10: {scenario_context_row['p10']:.1f}
P50: {scenario_context_row['p50']:.1f}
P90: {scenario_context_row['p90']:.1f}

Therapy-level impact:
{therapy_context}
"""

    st.code(scenario_context, language="text")

    user_question = st.text_area(
    "Question for RAG explanation",
    value=(
        f"Explain the {selected_for_explanation} scenario in simple business language. "
        "First explain what this project is trying to do, then explain the forecasting method, "
        "why recent demand was the strongest overall benchmark, what the selected scenario changes, "
        "how the scenario result should be interpreted, what evidence or assumptions support the explanation, "
        "and what limitations a pharma client or senior data scientist should keep in mind."
    ),
    height=140,
)

    if not RAG_AVAILABLE:
        st.warning(
            "RAG utilities are not available yet. Check rag/rag_utils.py and requirements.txt."
        )

        if RAG_IMPORT_ERROR:
            st.code(RAG_IMPORT_ERROR)

    else:
        col1, col2 = st.columns(2)

        with col1:
            preview_clicked = st.button("Preview Retrieved Evidence")

        with col2:
            explain_clicked = st.button("Generate Grounded Explanation")

        if preview_clicked:
            with st.spinner("Retrieving evidence chunks..."):
                evidence_text = evidence_preview(
                    query=user_question,
                    k=4,
                )

            st.markdown("### Retrieved Evidence")
            st.text(evidence_text)

        if explain_clicked:
            with st.spinner(
                "Retrieving evidence and generating grounded explanation..."
            ):
                result = explain_scenario_with_rag(
                    scenario_context=scenario_context,
                    user_question=user_question,
                    k=4,
                )

            st.markdown("### Grounded Explanation")
            st.write(result["answer"])

            with st.expander("Retrieved evidence used"):
                st.text(result["evidence_text"])

    if RAG_DOCS.exists():
        md_files = sorted(RAG_DOCS.glob("*.md"))

        if md_files:
            st.markdown("### Evidence Documents Available")

            for file in md_files:
                st.write(f"- {file.name}")
        else:
            st.info("No evidence markdown files found yet.")
    else:
        st.info("RAG evidence folder has not been populated yet.")

# ------------------------------------------------------------
# Tab 6: Governance
# ------------------------------------------------------------

with tab_governance:
    st.subheader("Governance and Claim Discipline")

    st.markdown("### Senior Review Gate")

    if senior_review_gate.empty:
        st.info("Senior review gate file not found.")
    else:
        st.dataframe(senior_review_gate, use_container_width=True)

    st.markdown("### Claim Discipline")

    if claim_discipline.empty:
        st.info("Claim discipline file not found.")
    else:
        st.dataframe(claim_discipline, use_container_width=True)

    st.markdown("### Feature Provenance")

    if feature_provenance.empty:
        st.info("Feature provenance file not found.")
    else:
        st.dataframe(feature_provenance, use_container_width=True)

    st.markdown("### Assumption Registry")

    if assumption_registry.empty:
        st.info("Assumption registry file not found.")
    else:
        st.dataframe(assumption_registry, use_container_width=True)

    st.markdown("### Feature Availability Audit")

    if feature_availability_audit.empty:
        st.info("Feature availability audit file not found.")
    else:
        st.dataframe(feature_availability_audit, use_container_width=True)
