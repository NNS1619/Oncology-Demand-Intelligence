# Patient-Informed Oncology Demand Forecasting and Scenario Intelligence

This project is a synthetic pharmaceutical analytics POC that evaluates when historical demand is sufficient and when patient-flow, epidemiology, market access, persistence, supply, and competitive signals add value for oncology demand planning.

The project uses a controlled synthetic oncology data-generating process, point-in-time feature governance, rolling-origin validation across 1/3/6/12-month horizons, forecast-value-add testing, Monte Carlo uncertainty, scenario analysis, and a bounded RAG/LLM explanation layer.

## Core Question

When should a pharmaceutical forecaster rely on historical demand versus patient and market information?

## Project Structure

- `notebooks/`: analytical notebooks
- `data/raw_synthetic/`: synthetic market-generation outputs
- `data/processed/`: SQL analytical mart and modeling datasets
- `data/outputs/`: forecasting, scenario, and review outputs
- `app/`: Streamlit decision-support app
- `rag/`: embedding-based evidence retrieval and explanation layer
- `docs/`: methodology and project documentation

## Important Limitation

This is a synthetic case study, not a clinically validated NSCLC model. Public epidemiology sources are used only as proxy anchors for selected assumptions. The project demonstrates forecasting methodology, pharmaceutical decision logic, feature governance, uncertainty analysis, scenario planning, and grounded explanation architecture.
# RAG architecture and validation

The RAG layer has three deliberately separate responsibilities:

1. `structured_router.py` selects exact facts from governed CSV outputs.
2. FAISS retrieves explanatory evidence from `evidence_docs/*.md` using Gemini embeddings.
3. Gemini writes a natural-language answer from the selected facts and retrieved evidence.

The LLM does not calculate official forecasts or scenario results.

## Build the vector store

Set `GOOGLE_API_KEY`, then run:

```bash
python rag/build_vector_store.py
```

The app can also build the index lazily on the first valid RAG question. Keep
`rag/vector_store/` out of Git unless you have intentionally decided to version
an index built from non-sensitive documents.

## Run validation

The evaluation set is `rag/evaluation_questions.csv`. It contains development
and holdout questions spanning methodology, governance, model results, scenario
lookups, uncertainty, out-of-scope requests, and adversarial requests.

```bash
# No API calls: corpus integrity and deterministic routing
python rag/evaluate_rag.py --mode static

# Embeddings/retrieval only: adds source Recall@5 and MRR
python rag/evaluate_rag.py --mode retrieval

# Complete path: adds grounded-answer, numerical, boundary, and availability checks
python rag/evaluate_rag.py --mode full --split holdout --delay-seconds 7
```

Each run creates:

- `data/outputs/rag_evaluation_details.csv`
- `data/outputs/rag_evaluation_summary.csv`

The Streamlit trust panel reads the summary. A value shown as **Not run** means
that the corresponding API-dependent layer has not been evaluated; it must not
be interpreted as either a pass or a zero score.

## Acceptance targets

| Check | Target |
|---|---:|
| Router intent accuracy | at least 95% |
| Router dimension accuracy | at least 95% |
| Expected-source Recall@5 | at least 85% |
| Mean reciprocal rank | at least 65% |
| Required-answer term coverage | at least 80% |
| Forbidden-claim pass rate | 100% |
| Numerical fidelity | at least 95% |
| Out-of-scope refusal accuracy | 100% |
| Successful answer/refusal rate | at least 95% |

These checks validate the POC implementation. They do not establish clinical
validity, external validity, or production readiness. A pharma deployment would
also require approved source ownership, access control, audit logging, monitoring,
security review, and validation on real company data.
