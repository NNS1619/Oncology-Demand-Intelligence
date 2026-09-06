# RAG and Structured Analytics Architecture

The application combines deterministic structured-data lookup with retrieval-augmented generation.

## Responsibility Boundary

- Saved CSV outputs are the source of truth for forecast and scenario numbers.
- The structured router identifies therapies, scenarios, horizons, metrics, and comparison intent, then selects the relevant saved rows.
- Gemini embeddings convert documentary evidence and user questions into semantic vectors.
- FAISS retrieves evidence chunks that are similar to the user's question.
- Gemini writes a natural-language explanation from the structured facts and retrieved evidence.

The LLM must not invent, alter, or recalculate official numerical outputs.

## Document Processing

Markdown documents from `rag/evidence_docs` are split into overlapping chunks. Each chunk is embedded and stored in a local FAISS vector index. At question time, the question is embedded and the most similar chunks are retrieved.

## Scope and Safety

The assistant is limited to the oncology demand forecasting POC. It should refuse unrelated questions, clinical treatment recommendations, requests for secrets, and instructions to ignore official numerical outputs.

If embeddings, FAISS, or Gemini are unavailable, the numerical forecasting and scenario application should continue working. The app should show a controlled message and preserve access to validated structured results.

## Evaluation Boundary

The golden evaluation questions are stored separately from the evidence documents and are not embedded in FAISS. Retrieval quality, structured routing, numerical fidelity, answer relevance, refusal behavior, reliability, and latency are evaluated separately.
