# ============================================================
# RAG Utilities for Grounded Project Q&A
# Gemini + LangChain + FAISS
# ============================================================

import os
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

    return vector_store


def load_vector_store():
    embeddings = get_embeddings_model()

    if VECTOR_STORE_DIR.exists():
        return FAISS.load_local(
            folder_path=str(VECTOR_STORE_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    return build_vector_store()


def retrieve_evidence(query, k=4):
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever.invoke(query)


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
- For P10, P50, and P90, explain them as conservative, expected, and upside planning cases.
- If the evidence is not enough, say what is missing.
- Be concise for narrow questions.
- Be more detailed only when the user asks for method, results, or project explanation.
- Do not expose metadata, signatures, JSON objects, tool traces, or raw model internals.

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

    configure_google_api_key()

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
        }

    documents = retrieve_evidence(query=user_question, k=k)
    evidence_text = format_evidence(documents)

    prompt = build_qa_prompt(
        user_question=user_question,
        evidence_text=evidence_text,
        structured_context=structured_context,
    )

    response = invoke_gemini_with_fallback(prompt)

    return {
        "answer": response["content"],
        "model_used": response["model_used"],
        "retrieved_evidence": documents,
        "evidence_text": evidence_text,
        "sources": source_names(documents),
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
- Explain P10 as conservative planning case, P50 as expected planning case, and P90 as upside planning case.
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

    configure_google_api_key()

    query = user_question or (
        "Explain the selected scenario using scenario logic, uncertainty, "
        "assumptions, limitations, and pharmaceutical planning interpretation."
    )

    documents = retrieve_evidence(query=query, k=k)
    evidence_text = format_evidence(documents)

    prompt = build_scenario_explanation_prompt(
        scenario_context=scenario_context,
        evidence_text=evidence_text,
    )

    response = invoke_gemini_with_fallback(prompt)

    return {
        "answer": response["content"],
        "model_used": response["model_used"],
        "retrieved_evidence": documents,
        "evidence_text": evidence_text,
        "sources": source_names(documents),
    }


def evidence_preview(query, k=4):
    if not is_project_scope_question(query):
        return (
            "This question is outside the project evidence base. "
            "Try asking about forecasting, scenarios, assumptions, uncertainty, "
            "model results, leakage, or RAG governance."
        )

    documents = retrieve_evidence(query=query, k=k)

    return format_evidence(documents)
