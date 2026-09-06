"""Deterministic routing for questions that require saved numerical results.

The router is intentionally separate from vector retrieval. Exact questions are
answered from governed CSV outputs; RAG supplies documentary evidence and the
LLM explains the already-selected facts.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


SCENARIO_CLIENT_LABELS = {
    "Base case": "Base planning forecast",
    "Access downside": "Uniform access downside",
    "Earlier strong competitor pressure": "Stronger overlapping competitor pressure",
    "Epidemiology upside": "Eligible population upside",
    "Persistence downside": "Persistence downside",
    "Therapy D East supply constraint": "Therapy D East supply constraint",
    "Combined downside": "Combined downside planning case",
}

THERAPY_NAMES = {
    "therapy a": "Therapy A",
    "therapy b": "Therapy B",
    "therapy c": "Therapy C",
    "therapy d": "Therapy D",
}

SCENARIO_NAMES = {
    "base planning forecast": "Base case",
    "base case": "Base case",
    "uniform access downside": "Access downside",
    "access downside": "Access downside",
    "access": "Access downside",
    "stronger overlapping competitor pressure": "Earlier strong competitor pressure",
    "earlier strong competitor pressure": "Earlier strong competitor pressure",
    "competitor pressure": "Earlier strong competitor pressure",
    "competition": "Earlier strong competitor pressure",
    "competitor": "Earlier strong competitor pressure",
    "eligible population upside": "Epidemiology upside",
    "epidemiology upside": "Epidemiology upside",
    "epidemiology": "Epidemiology upside",
    "persistence downside": "Persistence downside",
    "persistence": "Persistence downside",
    "therapy d east supply constraint": "Therapy D East supply constraint",
    "supply constraint": "Therapy D East supply constraint",
    "supply": "Therapy D East supply constraint",
    "combined downside planning case": "Combined downside",
    "combined downside": "Combined downside",
    "combined": "Combined downside",
}

MODEL_ALIASES = {
    "seasonal naive": "seasonal_naive",
    "historical xgboost": "historical_xgboost",
    "historical": "historical_xgboost",
    "patient driver": "patient_driver",
    "patient-driver": "patient_driver",
    "hybrid xgboost": "hybrid_xgboost",
    "hybrid": "hybrid_xgboost",
    "naive": "naive",
}

STRUCTURED_TERMS = {
    "worst", "best", "biggest", "largest", "smallest", "highest", "lowest",
    "most affected", "least affected", "decline", "drop", "increase",
    "improve", "improvement", "added value", "add value", "fva", "wape",
    "mae", "bias", "which therapy", "which scenario", "which model",
    "when will", "when does", "p10", "p50", "p90", "uncertainty",
    "scenario", "planning case", "forecast units", "change", "unchanged", "impact", "portfolio",
}


def _fmt_units(value: Any) -> str:
    return "Not available" if pd.isna(value) else f"{float(value):,.1f}"


def _fmt_pct(value: Any) -> str:
    return "Not available" if pd.isna(value) else f"{float(value):.1%}"


def detect_therapy(question: str) -> str | None:
    lower = question.lower()
    return next((therapy for key, therapy in THERAPY_NAMES.items() if key in lower), None)


def detect_scenario(question: str) -> str | None:
    lower = question.lower()
    for key in sorted(SCENARIO_NAMES, key=len, reverse=True):
        if key in lower:
            return SCENARIO_NAMES[key]
    return None


def detect_horizon(question: str) -> int | None:
    lower = question.lower()
    for pattern in (r"(\d+)\s*[- ]?months?", r"horizon\s*(?:of\s*)?(\d+)"):
        match = re.search(pattern, lower)
        if match:
            return int(match.group(1))
    return None


def detect_model(question: str) -> str | None:
    lower = question.lower()
    for key in sorted(MODEL_ALIASES, key=len, reverse=True):
        if key in lower:
            return MODEL_ALIASES[key]
    return None


def question_mentions_structured_result(question: str) -> bool:
    lower = question.lower()
    return any(term in lower for term in STRUCTURED_TERMS)


def _therapy_scenario_route(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    therapy = detect_therapy(question)
    scenario = detect_scenario(question)
    if therapy is None:
        return None

    work = df.loc[df["therapy"] == therapy].copy()
    if scenario:
        work = work.loc[work["scenario_name"] == scenario]
    if work.empty:
        return None

    lower = question.lower()
    if any(term in lower for term in ("worst", "decline", "drop", "most affected")):
        selected = work.nsmallest(1, "absolute_change_units").iloc[0]
        result_type = "largest negative change"
    elif any(term in lower for term in ("best", "increase", "upside", "benefit")):
        selected = work.nlargest(1, "absolute_change_units").iloc[0]
        result_type = "largest positive change"
    else:
        selected = work.loc[work["absolute_change_units"].abs().idxmax()]
        result_type = "largest absolute change"

    facts = {
        "therapy": selected["therapy"],
        "scenario_name": selected["scenario_name"],
        "baseline_forecast_units": float(selected["baseline_forecast_units"]),
        "scenario_forecast_units": float(selected["scenario_forecast_units"]),
        "absolute_change_units": float(selected["absolute_change_units"]),
        "percent_change": float(selected["percent_change"]),
    }
    context = f"""
STRUCTURED DATA RESULT:
Question type: Therapy-level scenario lookup
Result type: {result_type}

Therapy: {facts['therapy']}
Scenario: {facts['scenario_name']}
Client scenario label: {SCENARIO_CLIENT_LABELS.get(facts['scenario_name'], facts['scenario_name'])}
Baseline forecast units: {_fmt_units(facts['baseline_forecast_units'])}
Scenario forecast units: {_fmt_units(facts['scenario_forecast_units'])}
Absolute change units: {_fmt_units(facts['absolute_change_units'])}
Percent change: {_fmt_pct(facts['percent_change'])}

Use these saved values without recalculation. Explain the mechanism using retrieved evidence.
For a uniform access downside, similar percentage changes are expected by design;
absolute unit impact is the more useful cross-therapy comparison.
""".strip()
    return {"intent": "therapy_scenario", "context": context, "facts": facts}


def _cross_therapy_scenario_route(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    """Answer questions such as 'Which therapy loses most under access downside?'"""
    scenario = detect_scenario(question)
    lower = question.lower()
    if scenario is None or not any(
        term in lower for term in ("which therapy", "most affected", "least affected")
    ):
        return None
    work = df.loc[df["scenario_name"] == scenario].copy()
    if work.empty:
        return None
    if "least affected" in lower:
        selected = work.loc[work["absolute_change_units"].abs().idxmin()]
        result_type = "smallest absolute therapy change"
    elif any(term in lower for term in ("decline", "drop", "loses", "loss")):
        selected = work.nsmallest(1, "absolute_change_units").iloc[0]
        result_type = "largest negative therapy change"
    else:
        selected = work.loc[work["absolute_change_units"].abs().idxmax()]
        result_type = "largest absolute therapy change"
    facts = {
        "therapy": selected["therapy"],
        "scenario_name": selected["scenario_name"],
        "baseline_forecast_units": float(selected["baseline_forecast_units"]),
        "scenario_forecast_units": float(selected["scenario_forecast_units"]),
        "absolute_change_units": float(selected["absolute_change_units"]),
        "percent_change": float(selected["percent_change"]),
    }
    context = f"""
STRUCTURED DATA RESULT:
Question type: Cross-therapy scenario comparison
Result type: {result_type}

Scenario: {facts['scenario_name']}
Therapy: {facts['therapy']}
Baseline forecast units: {_fmt_units(facts['baseline_forecast_units'])}
Scenario forecast units: {_fmt_units(facts['scenario_forecast_units'])}
Absolute change units: {_fmt_units(facts['absolute_change_units'])}
Percent change: {_fmt_pct(facts['percent_change'])}

The comparison uses the saved therapy-level scenario output. Do not recalculate it.
""".strip()
    return {"intent": "therapy_scenario", "context": context, "facts": facts}


def _model_route(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    lower = question.lower()
    horizon = detect_horizon(question)
    model = detect_model(question)
    work = df.copy()
    if horizon is not None:
        work = work.loc[work["horizon_months"].astype(int) == horizon]
    if model is not None and not any(term in lower for term in ("best", "worst", "which model")):
        work = work.loc[work["model"] == model]
    if work.empty:
        return None

    if any(term in lower for term in ("worst", "highest wape", "least accurate")):
        selected = work.nlargest(1, "WAPE").iloc[0]
        result_type = "worst model by WAPE"
    else:
        selected = work.nsmallest(1, "WAPE").iloc[0]
        result_type = "best model by WAPE"

    facts = {
        "horizon_months": int(selected["horizon_months"]),
        "model": selected["model"],
        "WAPE": float(selected["WAPE"]),
        "MAE": float(selected["MAE"]),
        "Bias": float(selected["Bias"]),
    }
    context = f"""
STRUCTURED DATA RESULT:
Question type: Model-performance lookup
Result type: {result_type}

Forecast horizon: {facts['horizon_months']} months
Model: {facts['model']}
WAPE: {_fmt_pct(facts['WAPE'])}
MAE: {_fmt_units(facts['MAE'])}
Bias: {_fmt_pct(facts['Bias'])}

Lower WAPE is better. Do not claim that complex ML wins overall unless this stored result supports it.
""".strip()
    return {"intent": "model_performance", "context": context, "facts": facts}


def _regime_route(question: str, df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    model = detect_model(question)
    if model not in {"historical_xgboost", "patient_driver", "hybrid_xgboost"}:
        return None
    fva_col = f"{model}_fva_vs_naive"
    wape_col = f"{model}_wape"
    if fva_col not in df.columns or wape_col not in df.columns:
        return None
    work = df.loc[df[fva_col] > 0].sort_values(fva_col, ascending=False)
    if work.empty:
        context = (
            "STRUCTURED DATA RESULT:\n"
            f"Model checked: {model}\n"
            "No positive Forecast Value Add regime windows were found."
        )
        return {"intent": "regime_fva", "context": context, "facts": {"model": model, "rows": []}}

    rows = []
    lines = []
    for _, row in work.head(5).iterrows():
        fact = {
            "horizon_months": int(row["horizon_months"]),
            "target_market_regime": row["target_market_regime"],
            "model_wape": float(row[wape_col]),
            "naive_wape": float(row["naive_wape"]),
            "fva_vs_naive": float(row[fva_col]),
            "n_forecasts": int(row["n_forecasts"]),
        }
        rows.append(fact)
        lines.append(
            f"- Horizon {fact['horizon_months']} months, regime {fact['target_market_regime']}: "
            f"{model} WAPE {_fmt_pct(fact['model_wape'])}, naive WAPE {_fmt_pct(fact['naive_wape'])}, "
            f"FVA {_fmt_pct(fact['fva_vs_naive'])}, n={fact['n_forecasts']}"
        )
    context = (
        "STRUCTURED DATA RESULT:\nQuestion type: Regime Forecast Value Add lookup\n"
        f"Model checked: {model}\nPositive value-add windows:\n" + "\n".join(lines) +
        "\nMention sample-size caution when n is small."
    )
    return {"intent": "regime_fva", "context": context, "facts": {"model": model, "rows": rows}}


def _overall_scenario_route(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    scenario = detect_scenario(question)
    work = df.copy()
    if scenario:
        work = work.loc[work["scenario_name"] == scenario]
    if work.empty:
        return None

    lower = question.lower()
    if any(term in lower for term in ("worst", "decline", "drop", "loss")):
        selected = work.nsmallest(1, "absolute_change_units").iloc[0]
        result_type = "largest overall downside"
    elif any(term in lower for term in ("best", "largest increase", "biggest increase", "upside")):
        selected = work.nlargest(1, "absolute_change_units").iloc[0]
        result_type = "largest overall upside"
    else:
        selected = work.iloc[0]
        result_type = "selected scenario"

    facts = {
        "scenario_name": selected["scenario_name"],
        "baseline_forecast_units": float(selected["baseline_forecast_units"]),
        "scenario_forecast_units": float(selected["scenario_forecast_units"]),
        "absolute_change_units": float(selected["absolute_change_units"]),
        "percent_change": float(selected["percent_change"]),
        "p10": float(selected["p10"]),
        "p50": float(selected["p50"]),
        "p90": float(selected["p90"]),
    }
    context = f"""
STRUCTURED DATA RESULT:
Question type: Overall scenario lookup
Result type: {result_type}

Scenario: {facts['scenario_name']}
Client scenario label: {SCENARIO_CLIENT_LABELS.get(facts['scenario_name'], facts['scenario_name'])}
Baseline forecast units: {_fmt_units(facts['baseline_forecast_units'])}
Scenario forecast units: {_fmt_units(facts['scenario_forecast_units'])}
Absolute change units: {_fmt_units(facts['absolute_change_units'])}
Percent change: {_fmt_pct(facts['percent_change'])}
Conservative planning case, P10: {_fmt_units(facts['p10'])}
Expected planning case, P50: {_fmt_units(facts['p50'])}
Upside planning case, P90: {_fmt_units(facts['p90'])}

These are simulated planning ranges, not clinical confidence intervals. Do not recalculate them.
""".strip()
    return {"intent": "overall_scenario", "context": context, "facts": facts}


def route_structured_question(
    question: str,
    scenario_summary_with_uncertainty: pd.DataFrame,
    therapy_scenario_summary: pd.DataFrame,
    compact_metrics: pd.DataFrame,
    regime_positive_fva: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Return deterministic context and facts, or None for documentary questions."""
    if not question_mentions_structured_result(question):
        return None

    lower = question.lower()
    # Metric-definition and methodology questions need documentary evidence,
    # not an arbitrary numerical row selected from the results table.
    if (
        any(phrase in lower for phrase in ("what does", "what do", "define", "explain the metric"))
        and detect_horizon(question) is None
        and detect_scenario(question) is None
        and detect_therapy(question) is None
        and not any(term in lower for term in ("which model", "best", "worst", "highest", "lowest"))
    ):
        return None

    # Architecture and adversarial-boundary questions are answered from the
    # evidence corpus. They do not need a numerical row injected into context.
    if (
        ("llm" in lower and any(term in lower for term in ("calculate", "official")))
        or any(term in lower for term in ("invent", "ignore the saved", "ignore saved"))
    ):
        return None

    # A therapy name can be part of a portfolio-level scenario label. In that
    # case, route to the portfolio summary rather than a therapy-only row.
    if "portfolio" in lower:
        portfolio_route = _overall_scenario_route(question, scenario_summary_with_uncertainty)
        if portfolio_route is not None:
            return portfolio_route

    therapy_route = _therapy_scenario_route(question, therapy_scenario_summary)
    if therapy_route is not None:
        return therapy_route

    cross_therapy_route = _cross_therapy_scenario_route(question, therapy_scenario_summary)
    if cross_therapy_route is not None:
        return cross_therapy_route

    if any(term in lower for term in ("fva", "added value", "add value", "regime", "where does")):
        regime_route = _regime_route(question, regime_positive_fva)
        if regime_route is not None:
            return regime_route

    if any(term in lower for term in ("model", "wape", "mae", "bias", "naive", "xgboost", "hybrid")):
        model_route = _model_route(question, compact_metrics)
        if model_route is not None:
            return model_route

    if any(term in lower for term in ("scenario", "p10", "p50", "p90", "uncertainty", "downside", "upside", "drop", "worst", "best")):
        return _overall_scenario_route(question, scenario_summary_with_uncertainty)

    return None


def build_structured_context_for_question(
    question: str,
    scenario_summary_with_uncertainty: pd.DataFrame,
    therapy_scenario_summary: pd.DataFrame,
    compact_metrics: pd.DataFrame,
    regime_positive_fva: pd.DataFrame | None = None,
) -> str | None:
    route = route_structured_question(
        question=question,
        scenario_summary_with_uncertainty=scenario_summary_with_uncertainty,
        therapy_scenario_summary=therapy_scenario_summary,
        compact_metrics=compact_metrics,
        regime_positive_fva=regime_positive_fva,
    )
    return None if route is None else route["context"]
