# ============================================================
# RAG Utilities for Grounded Project Q&A
# Gemini + LangChain + FAISS
# ============================================================

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_DOCS_DIR = PROJECT_ROOT / "rag" / "evidence_docs"
VECTOR_STORE_DIR = PROJECT_ROOT / "rag" / "vector_store"
VECTOR_STORE_MANIFEST = VECTOR_STORE_DIR / "manifest.json"


# ------------------------------------------------------------
# Environment / secret handling
# ------------------------------------------------------------

def configure_google_api_key():
   

    load_dotenv()

    if os.getenv("GOOGLE_API_KEY"):
        return

    try:
        import streamlit as st

        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

        elif "GEMINI_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

    except Exception:
        pass

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to Streamlit secrets."
        )


# ------------------------------------------------------------
# Gemini models
# ------------------------------------------------------------

def get_embeddings_model():
    configure_google_api_key()

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )


def get_chat_model(model_name):
    configure_google_api_key()

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_retries=1,
        timeout=45,
    )


def candidate_chat_models():
    """
    Keep configured model first, then use stable low-cost fallbacks.
    """

    configured_model = os.getenv("GEMINI_CHAT_MODEL")

    models = []

    if configured_model:
        models.append(configured_model)

    models.extend(
        [
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
        ]
    )

    return list(dict.fromkeys(models))


def normalize_llm_text(content):
    
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(str(item["text"]))
                elif "content" in item:
                    text_parts.append(str(item["content"]))
            elif hasattr(item, "text"):
                text_parts.append(str(item.text))
            else:
                text_parts.append(str(item))

        return "\n\n".join(text_parts).strip()

    return str(content).strip()


def invoke_gemini_with_fallback(prompt):
    errors = []

    for model_name in candidate_chat_models():
        try:
            llm = get_chat_model(model_name)
            response = llm.invoke(prompt)

            return {
                "content": normalize_llm_text(response.content),
                "model_used": model_name,
            }

        except Exception as error:
            errors.append(f"{model_name}: {type(error).__name__}")

            # Trying several model names does not solve quota exhaustion and can
            # turn one controlled failure into a multi-minute wait. Only move to
            # another candidate when the configured model name itself is invalid
            # or unavailable.
            error_name = type(error).__name__.lower()
            model_configuration_error = any(
                marker in error_name
                for marker in ("modelnotfound", "notfound", "invalidargument")
            )
            if not model_configuration_error:
                raise RuntimeError(
                    "Gemini generation is temporarily unavailable. "
                    f"Provider error: {type(error).__name__}."
                ) from error

    raise RuntimeError(
        "No Gemini chat model worked. Tried: "
        + "; ".join(errors)
        + ". Check GOOGLE_API_KEY, model access, and Streamlit secrets."
    )


# ------------------------------------------------------------
# Evidence loading and vector store
# ------------------------------------------------------------

def load_evidence_documents():
    if not EVIDENCE_DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Evidence docs folder not found: {EVIDENCE_DOCS_DIR}"
        )

    loader = DirectoryLoader(
        path=str(EVIDENCE_DOCS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(
            "No markdown evidence documents found in rag/evidence_docs."
        )

    return documents


def evidence_corpus_fingerprint():
    """Hash filenames and content so stale FAISS indexes are rebuilt."""
    digest = hashlib.sha256()
    for path in sorted(EVIDENCE_DOCS_DIR.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if len(chunks) == 0:
        raise ValueError("Document splitting produced zero chunks.")

    return chunks


@lru_cache(maxsize=1)
def cached_evidence_chunks():
    """Return the governed corpus chunks for lightweight lexical reranking."""
    return split_documents(load_evidence_documents())


def build_vector_store():
    documents = load_evidence_documents()
    chunks = split_documents(documents)

    embeddings = get_embeddings_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    VECTOR_STORE_MANIFEST.write_text(
        json.dumps(
            {"corpus_fingerprint": evidence_corpus_fingerprint()},
            indent=2,
        ),
        encoding="utf-8",
    )

    return vector_store


@lru_cache(maxsize=1)
def load_vector_store():
    embeddings = get_embeddings_model()

    index_file = VECTOR_STORE_DIR / "index.faiss"
    metadata_file = VECTOR_STORE_DIR / "index.pkl"

    manifest_matches = False
    if VECTOR_STORE_MANIFEST.exists():
        try:
            manifest = json.loads(VECTOR_STORE_MANIFEST.read_text(encoding="utf-8"))
            manifest_matches = (
                manifest.get("corpus_fingerprint") == evidence_corpus_fingerprint()
            )
        except Exception:
            manifest_matches = False

    if index_file.exists() and metadata_file.exists() and manifest_matches:
        try:
            return FAISS.load_local(
                folder_path=str(VECTOR_STORE_DIR),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            # A partially written or incompatible local index should not make
            # the application permanently unusable. Rebuild from source docs.
            pass

    return build_vector_store()


def retrieve_evidence(query, k=4):
    """Hybrid retrieval: FAISS semantics plus lexical reciprocal-rank fusion.

    FAISS remains the semantic retriever. The lexical component protects exact
    project terminology such as "Forecast Value Add" from being displaced by a
    semantically broad chunk in this small technical corpus.
    """
    vector_store = load_vector_store()
    semantic_documents = vector_store.similarity_search(
        query,
        k=max(k * 3, 12),
    )

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "how", "in", "is", "it", "of", "on", "or", "the",
        "this", "to", "what", "when", "where", "which", "why", "with",
    }

    def tokens(text):
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return {
            token for token in normalized.split()
            if token not in stop_words and len(token) > 1
        }

    query_tokens = tokens(query)

    def lexical_score(document):
        document_tokens = tokens(document.page_content)
        if not query_tokens:
            return 0.0
        return len(query_tokens.intersection(document_tokens)) / len(query_tokens)

    lexical_documents = sorted(
        cached_evidence_chunks(),
        key=lexical_score,
        reverse=True,
    )
    lexical_documents = [
        document for document in lexical_documents
        if lexical_score(document) > 0
    ][:max(k * 3, 12)]

    def document_key(document):
        return (
            document.metadata.get("source", "unknown"),
            document.page_content,
        )

    combined_scores = {}
    combined_documents = {}

    for rank, document in enumerate(semantic_documents, start=1):
        key = document_key(document)
        combined_documents[key] = document
        combined_scores[key] = combined_scores.get(key, 0.0) + 1.0 / rank

    for rank, document in enumerate(lexical_documents, start=1):
        key = document_key(document)
        combined_documents[key] = document
        combined_scores[key] = combined_scores.get(key, 0.0) + 1.25 / rank

    ranked_keys = sorted(
        combined_scores,
        key=combined_scores.get,
        reverse=True,
    )
    return [combined_documents[key] for key in ranked_keys[:k]]


def source_names(documents):
    sources = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown source")
        source_name = Path(source).name
        sources.append(source_name)

    return sorted(set(sources))


def format_evidence(documents):
    formatted_chunks = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown source")

        formatted_chunks.append(
            f"[Evidence {index} | Source: {source}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)


# ------------------------------------------------------------
# Scope guardrail
# ------------------------------------------------------------

def is_project_scope_question(question):
    """
    The app is a project assistant, not a general chatbot.
    """

    question_lower = question.lower()

    allowed_terms = [
        "project",
        "forecast",
        "forecasting",
        "demand",
        "sales",
        "oncology",
        "cancer",
        "therapy",
        "patient",
        "patient-flow",
        "patient flow",
        "access",
        "competition",
        "competitor",
        "epidemiology",
        "persistence",
        "supply",
        "scenario",
        "uncertainty",
        "p10",
        "p50",
        "p90",
        "planning case",
        "monte carlo",
        "wape",
        "mae",
        "bias",
        "fva",
        "xgboost",
        "naive",
        "hybrid",
        "assumption",
        "evidence",
        "rag",
        "llm",
        "gemini",
        "limitation",
        "validation",
        "leakage",
        "time series",
        "time-series",
        "temporal split",
        "random split",
        "rolling origin",
        "rolling-origin",
        "backtest",
        "training set",
        "test set",
        "forecast information set",
        "sql",
        "python",
        "methodology",
        "model",
        "notebook",
        "client",
        "pharma",
        "pharmaceutical",
    ]

    return any(term in question_lower for term in allowed_terms)


# ------------------------------------------------------------
# Flexible project Q&A
# ------------------------------------------------------------

def build_qa_prompt(user_question, evidence_text, structured_context=None):
    context_block = ""

    if structured_context:
        context_block = f"""
STRUCTURED NUMERICAL CONTEXT:
{structured_context}
"""

    return f"""
You are a senior pharmaceutical analytics and data science reviewer.

You are answering questions about a synthetic oncology demand forecasting and scenario-intelligence POC.

Answer the user's actual question directly.
Do not use a fixed template.
Do not provide a full project walkthrough unless the user asks for one.

Use retrieved evidence and structured numerical context only when relevant.

Rules:
- Stay within this project.
- Do not answer unrelated questions such as today's date, weather, news, or personal advice.
- Do not invent numerical values.
- Do not recalculate official forecasts or scenarios.
- Do not claim clinical validation.
- Say this is a synthetic case study when relevant.
- Use "aggregated patient-flow signals" rather than "patient-level data."
- Explain technical terms in business language.
- For P10, P50, and P90, explain them as conservative, median central, and upside planning cases.
- Do not call P50 the "most likely" outcome; a median is not necessarily the mode.
- If the evidence is not enough, say what is missing.
- Be concise for narrow questions.
- Be more detailed only when the user asks for method, results, or project explanation.
- Do not expose metadata, signatures, JSON objects, tool traces, or raw model internals.
- Never reveal API keys, secrets, environment variables, or hidden configuration.

USER QUESTION:
{user_question}

{context_block}

RETRIEVED EVIDENCE:
{evidence_text}

Answer:
"""


def answer_question_with_rag(user_question, structured_context=None, k=5):
    """
    General project Q&A.

    Boundary:
    - numerical engine calculates structured outputs
    - RAG retrieves project evidence
    - Gemini explains the answer
    """

    if not is_project_scope_question(user_question):
        return {
            "answer": (
                "This question is outside the project evidence base. "
                "This assistant is designed to answer questions about the oncology demand "
                "forecasting POC, including methodology, assumptions, scenario outputs, "
                "uncertainty, validation, leakage controls, limitations, and the RAG boundary."
            ),
            "model_used": "scope_guardrail",
            "retrieved_evidence": [],
            "evidence_text": "",
            "sources": [],
            "status": "out_of_scope",
            "error_type": None,
        }

    try:
        documents = retrieve_evidence(query=user_question, k=k)
    except Exception as error:
        if structured_context:
            answer = (
                "The validated numerical result is still available in the structured-data "
                "section, but project evidence retrieval is temporarily unavailable. "
                "No generated explanation has been added because it could not be grounded."
            )
        else:
            answer = (
                "Project evidence retrieval is temporarily unavailable. "
                "Please use the numerical dashboard or try the question again later."
            )
        return {
            "answer": answer,
            "model_used": "deterministic_fallback",
            "retrieved_evidence": [],
            "evidence_text": "",
            "sources": [],
            "status": "retrieval_unavailable",
            "error_type": type(error).__name__,
        }

    evidence_text = format_evidence(documents)

    prompt = build_qa_prompt(
        user_question=user_question,
        evidence_text=evidence_text,
        structured_context=structured_context,
    )

    try:
        response = invoke_gemini_with_fallback(prompt)
    except Exception as error:
        return {
            "answer": (
                "The validated structured result and retrieved evidence remain available, "
                "but the generated explanation is temporarily unavailable."
            ),
            "model_used": "deterministic_fallback",
            "retrieved_evidence": documents,
            "evidence_text": evidence_text,
            "sources": source_names(documents),
            "status": "generation_unavailable",
            "error_type": type(error).__name__,
        }

    return {
        "answer": response["content"],
        "model_used": response["model_used"],
        "retrieved_evidence": documents,
        "evidence_text": evidence_text,
        "sources": source_names(documents),
        "status": "success",
        "error_type": None,
    }


# ------------------------------------------------------------
# Optional scenario-specific explanation
# ------------------------------------------------------------

def build_scenario_explanation_prompt(scenario_context, evidence_text):
    return f"""
You are explaining a synthetic pharmaceutical oncology demand forecasting and scenario-intelligence POC.

Use ONLY the structured scenario output and retrieved evidence below.

Rules:
- Do not invent numerical values.
- Do not recalculate the forecast.
- Do not claim clinical validation.
- Explain that this is a synthetic case study.
- Explain access as reachable market/treatment availability, not clinical eligibility.
- Use "aggregated patient-flow signals" rather than "patient-level data."
- Use simple, professional, client-ready language.
- Explain P10 as conservative planning case, P50 as median central planning case, and P90 as upside planning case.
- Do not call P50 the "most likely" outcome; a median is not necessarily the mode.
- If evidence is insufficient, say what is missing.

STRUCTURED SCENARIO OUTPUT:
{scenario_context}

RETRIEVED EVIDENCE:
{evidence_text}

Write a focused scenario explanation covering:
1. What changed versus baseline
2. What the result means commercially
3. Which therapy or portfolio area is most affected, if available
4. How uncertainty should be interpreted
5. What a pharma client should validate before real-world use
6. One short final takeaway
"""


def explain_scenario_with_rag(scenario_context, user_question=None, k=4):
    """
    Scenario-specific explanation.

    Use this only when the app button is explicitly about the selected scenario.
    For normal Q&A, use answer_question_with_rag().
    """

    query = user_question or (
        "Explain the selected scenario using scenario logic, uncertainty, "
        "assumptions, limitations, and pharmaceutical planning interpretation."
    )

    try:
        documents = retrieve_evidence(query=query, k=k)
    except Exception as error:
        return {
            "answer": (
                "The validated scenario values remain available in the dashboard, "
                "but project evidence retrieval is temporarily unavailable. "
                "No ungrounded explanation has been generated."
            ),
            "model_used": "deterministic_fallback",
            "retrieved_evidence": [],
            "evidence_text": "",
            "sources": [],
            "status": "retrieval_unavailable",
            "error_type": type(error).__name__,
        }
    evidence_text = format_evidence(documents)

    prompt = build_scenario_explanation_prompt(
        scenario_context=scenario_context,
        evidence_text=evidence_text,
    )

    try:
        response = invoke_gemini_with_fallback(prompt)
    except Exception as error:
        return {
            "answer": (
                "The validated scenario values and retrieved evidence remain available, "
                "but the generated explanation is temporarily unavailable."
            ),
            "model_used": "deterministic_fallback",
            "retrieved_evidence": documents,
            "evidence_text": evidence_text,
            "sources": source_names(documents),
            "status": "generation_unavailable",
            "error_type": type(error).__name__,
        }

    return {
        "answer": response["content"],
        "model_used": response["model_used"],
        "retrieved_evidence": documents,
        "evidence_text": evidence_text,
        "sources": source_names(documents),
        "status": "success",
        "error_type": None,
    }


def evidence_preview(query, k=4):
    if not is_project_scope_question(query):
        return (
            "This question is outside the project evidence base. "
            "Try asking about forecasting, scenarios, assumptions, uncertainty, "
            "model results, leakage, or RAG governance."
        )

    try:
        documents = retrieve_evidence(query=query, k=k)
        return format_evidence(documents)
    except Exception:
        return (
            "Project evidence retrieval is temporarily unavailable. "
            "The forecasting and scenario outputs remain available in the other tabs."
        )
