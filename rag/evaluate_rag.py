"""Evaluate the oncology POC's analytics router, retriever, and grounded Q&A.

Usage
-----
python rag/evaluate_rag.py --mode static
python rag/evaluate_rag.py --mode retrieval
python rag/evaluate_rag.py --mode full

Static mode never calls Gemini. Retrieval mode calls the embedding service but
not the chat model. Full mode evaluates the complete RAG path. Results are
written to data/outputs as both question-level detail and a compact summary.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVALUATION_FILE = PROJECT_ROOT / "rag" / "evaluation_questions.csv"
EVIDENCE_DIR = PROJECT_ROOT / "rag" / "evidence_docs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
DETAIL_FILE = OUTPUT_DIR / "rag_evaluation_details.csv"
SUMMARY_FILE = OUTPUT_DIR / "rag_evaluation_summary.csv"

from rag.structured_router import route_structured_question  # noqa: E402


def _optional_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _split_expected(value: Any) -> list[str]:
    text = _optional_text(value)
    return [item.strip() for item in text.split(";") if item.strip()]


def _bool_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame | None]]:
    questions = pd.read_csv(EVALUATION_FILE)
    required = {
        "test_id", "split", "category", "question", "expected_intent",
        "expected_sources", "required_terms", "forbidden_terms",
        "expected_numbers", "should_refuse", "critical",
    }
    missing = sorted(required.difference(questions.columns))
    if missing:
        raise ValueError(f"Evaluation set is missing columns: {missing}")
    if questions["test_id"].duplicated().any():
        raise ValueError("Evaluation test_id values must be unique.")

    def read_optional(filename: str) -> pd.DataFrame | None:
        path = OUTPUT_DIR / filename
        return pd.read_csv(path) if path.exists() else None

    tables = {
        "scenario": read_optional("scenario_summary_with_uncertainty.csv"),
        "therapy": read_optional("therapy_scenario_summary.csv"),
        "compact": read_optional("compact_model_comparison.csv"),
        "regime": read_optional("regime_positive_fva.csv"),
    }
    for name in ("scenario", "therapy", "compact"):
        if tables[name] is None:
            raise FileNotFoundError(f"Required structured table is missing: {name}")
    return questions, tables


def audit_evidence_corpus() -> dict[str, Any]:
    files = sorted(EVIDENCE_DIR.glob("*.md"))
    texts = [path.read_text(encoding="utf-8") for path in files]
    hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    secret_pattern = re.compile(
        r"(?:AIza[0-9A-Za-z_-]{30,}|sk-[0-9A-Za-z_-]{20,}|OPENAI_API_KEY\s*=\s*[^\s]+)",
        re.IGNORECASE,
    )
    return {
        "evidence_document_count": len(files),
        "evidence_nonempty_rate": float(np.mean([bool(text.strip()) for text in texts])) if texts else 0.0,
        "evidence_unique_rate": len(set(hashes)) / len(hashes) if hashes else 0.0,
        "evidence_secret_scan_pass": float(not any(secret_pattern.search(text) for text in texts)),
    }


def evaluate_router(row: pd.Series, tables: dict[str, pd.DataFrame | None]) -> dict[str, Any]:
    route = route_structured_question(
        question=row["question"],
        scenario_summary_with_uncertainty=tables["scenario"],
        therapy_scenario_summary=tables["therapy"],
        compact_metrics=tables["compact"],
        regime_positive_fva=tables["regime"],
    )
    actual_intent = "" if route is None else route["intent"]
    expected_intent = _optional_text(row.get("expected_intent"))
    intent_pass = float(actual_intent == expected_intent)

    facts = {} if route is None else route.get("facts", {})
    dimension_checks = []
    for column, fact_name in (
        ("expected_therapy", "therapy"),
        ("expected_scenario", "scenario_name"),
        ("expected_horizon", "horizon_months"),
    ):
        raw_expected = row.get(column)
        expected = _optional_text(raw_expected)
        if not expected:
            continue
        actual = facts.get(fact_name)
        if fact_name == "horizon_months" and actual is not None:
            actual = str(int(actual))
            expected = str(int(float(raw_expected)))
        dimension_checks.append(str(actual) == expected)

    dimension_pass = float(all(dimension_checks)) if dimension_checks else np.nan
    return {
        "actual_intent": actual_intent,
        "router_intent_pass": intent_pass,
        "router_dimension_pass": dimension_pass,
        "structured_context": "" if route is None else route["context"],
    }


def _basename_sources(sources: list[str]) -> list[str]:
    return [Path(str(source)).name for source in sources]


def score_sources(actual_sources: list[str], expected_value: Any) -> tuple[float, float]:
    expected = _split_expected(expected_value)
    if not expected:
        return np.nan, np.nan
    actual = _basename_sources(actual_sources)
    hits = sum(source in actual for source in expected)
    recall = hits / len(expected)
    reciprocal_ranks = [1 / (actual.index(source) + 1) for source in expected if source in actual]
    mrr = max(reciprocal_ranks, default=0.0)
    return recall, mrr


def score_text(answer: str, row: pd.Series) -> dict[str, float]:
    lower = answer.lower()
    required = _split_expected(row.get("required_terms"))
    forbidden = _split_expected(row.get("forbidden_terms"))
    numbers = _split_expected(row.get("expected_numbers"))

    required_rate = (
        float(np.mean([term.lower() in lower for term in required])) if required else np.nan
    )
    forbidden_pass = (
        float(not any(term.lower() in lower for term in forbidden)) if forbidden else np.nan
    )

    actual_numbers = [
        float(item.replace(",", ""))
        for item in re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", answer)
    ]
    number_checks = []
    for expected_text in numbers:
        expected = float(expected_text.replace(",", ""))
        tolerance = max(0.2, abs(expected) * 0.002)
        number_checks.append(any(abs(actual - expected) <= tolerance for actual in actual_numbers))
    numerical_exactness = float(np.mean(number_checks)) if number_checks else np.nan

    return {
        "required_term_coverage": required_rate,
        "forbidden_term_pass": forbidden_pass,
        "numerical_exactness": numerical_exactness,
    }


def evaluate_questions(
    mode: str,
    split: str | None = None,
    delay_seconds: float = 0.0,
) -> pd.DataFrame:
    questions, tables = load_inputs()
    if split:
        questions = questions.loc[questions["split"] == split].copy()
        if questions.empty:
            raise ValueError(f"No evaluation questions found for split '{split}'.")
    rows: list[dict[str, Any]] = []

    if mode in {"retrieval", "full"}:
        from rag.rag_utils import answer_question_with_rag, retrieve_evidence

    for _, question_row in questions.iterrows():
        started = time.perf_counter()
        result = question_row.to_dict()
        result.update(evaluate_router(question_row, tables))
        result.update({
            "run_status": "static_only",
            "answer": "",
            "actual_sources": "",
            "source_recall_at_k": np.nan,
            "source_mrr": np.nan,
            "required_term_coverage": np.nan,
            "forbidden_term_pass": np.nan,
            "numerical_exactness": np.nan,
            "refusal_pass": np.nan,
            "response_success": np.nan,
        })

        try:
            if mode == "retrieval":
                documents = retrieve_evidence(question_row["question"], k=5)
                sources = [doc.metadata.get("source", "unknown") for doc in documents]
                recall, mrr = score_sources(sources, question_row.get("expected_sources"))
                result.update({
                    "run_status": "retrieval_success",
                    "actual_sources": ";".join(_basename_sources(sources)),
                    "source_recall_at_k": recall,
                    "source_mrr": mrr,
                    "response_success": 1.0,
                })
            elif mode == "full":
                rag_result = answer_question_with_rag(
                    user_question=question_row["question"],
                    structured_context=result["structured_context"] or None,
                    k=5,
                )
                answer = rag_result.get("answer", "")
                retrieved_documents = rag_result.get("retrieved_evidence", [])
                sources = [
                    document.metadata.get("source", "unknown")
                    for document in retrieved_documents
                ]
                if not sources:
                    sources = rag_result.get("sources", [])
                status = rag_result.get("status", "unknown")
                recall, mrr = score_sources(sources, question_row.get("expected_sources"))
                text_scores = score_text(answer, question_row)
                should_refuse = _bool_value(question_row.get("should_refuse"))
                refusal_pass = float(status == "out_of_scope") if should_refuse else float(status != "out_of_scope")
                result.update({
                    "run_status": status,
                    "answer": answer,
                    "actual_sources": ";".join(_basename_sources(sources)),
                    "source_recall_at_k": recall,
                    "source_mrr": mrr,
                    "refusal_pass": refusal_pass,
                    "response_success": float(status in {"success", "out_of_scope"}),
                    **text_scores,
                })
        except Exception as error:
            result["run_status"] = f"error:{type(error).__name__}"
            result["response_success"] = 0.0

        result["latency_seconds"] = round(time.perf_counter() - started, 4)
        rows.append(result)
        if mode == "full" and delay_seconds > 0:
            time.sleep(delay_seconds)

    return pd.DataFrame(rows)


def _mean_present(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def build_summary(details: pd.DataFrame, mode: str, split: str | None = None) -> pd.DataFrame:
    audit = audit_evidence_corpus()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = {
        "questions_evaluated": float(len(details)),
        "critical_questions": float(pd.to_numeric(details["critical"], errors="coerce").fillna(0).sum()),
        "router_intent_accuracy": _mean_present(details["router_intent_pass"]),
        "router_dimension_accuracy": _mean_present(details["router_dimension_pass"]),
        "source_recall_at_k": _mean_present(details["source_recall_at_k"]),
        "source_mrr": _mean_present(details["source_mrr"]),
        "required_term_coverage": _mean_present(details["required_term_coverage"]),
        "forbidden_term_pass_rate": _mean_present(details["forbidden_term_pass"]),
        "numerical_exactness": _mean_present(details["numerical_exactness"]),
        "refusal_accuracy": _mean_present(details["refusal_pass"]),
        "response_success_rate": _mean_present(details["response_success"]),
        "mean_latency_seconds": _mean_present(details["latency_seconds"]),
        **audit,
    }

    targets = {
        "router_intent_accuracy": 0.95,
        "router_dimension_accuracy": 0.95,
        "source_recall_at_k": 0.85,
        "source_mrr": 0.65,
        "required_term_coverage": 0.80,
        "forbidden_term_pass_rate": 1.0,
        "numerical_exactness": 0.95,
        "refusal_accuracy": 1.0,
        "response_success_rate": 0.95,
        "evidence_nonempty_rate": 1.0,
        "evidence_unique_rate": 1.0,
        "evidence_secret_scan_pass": 1.0,
    }
    count_metrics = {"questions_evaluated", "critical_questions", "evidence_document_count"}
    rows = []
    for name, value in metrics.items():
        target = targets.get(name, np.nan)
        if math.isnan(value):
            status = "NOT RUN"
        elif name in count_metrics or math.isnan(target):
            status = "INFORMATIONAL"
        else:
            status = "PASS" if value >= target else "REVIEW"
        rows.append({
            "metric": name,
            "value": value,
            "target": target,
            "status": status,
            "evaluation_mode": mode,
            "evaluation_split": split or "all",
            "generated_at_utc": now,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "retrieval", "full"], default="static")
    parser.add_argument("--split", choices=["development", "holdout"], default=None)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Pause between full-mode questions to respect provider request limits.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details = evaluate_questions(
        mode=args.mode,
        split=args.split,
        delay_seconds=args.delay_seconds,
    )
    summary = build_summary(details, args.mode, args.split)
    details.to_csv(DETAIL_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print(f"RAG evaluation mode: {args.mode}; split: {args.split or 'all'}")
    print(f"Question-level results: {DETAIL_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(summary[["metric", "value", "target", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
