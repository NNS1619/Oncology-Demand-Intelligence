# ============================================================
# RAG Utilities for Grounded Scenario Explanation
# Gemini + LangChain + FAISS
# ============================================================

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


def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )


def get_chat_model():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
    )


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
    """
    Builds and saves FAISS from rag/evidence_docs/*.md.

    This creates real vector embeddings using Gemini.
    """

    load_dotenv()

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
    """
    Loads FAISS if present. If absent, builds it from evidence docs.

    On Streamlit Cloud, this means the first RAG click may take longer.
    """

    load_dotenv()

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


def format_evidence(documents):
    formatted_chunks = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown source")

        formatted_chunks.append(
            f"[Evidence {index} | Source: {source}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def build_scenario_explanation_prompt(scenario_context, evidence_text):
    return f"""
You are explaining a pharmaceutical oncology demand forecasting scenario.

Use ONLY the structured scenario output and retrieved evidence below.
Do not invent numerical values.
Do not recalculate the forecast.
Do not claim clinical validation.
If evidence is insufficient, say what is missing.

STRUCTURED SCENARIO OUTPUT:
{scenario_context}

RETRIEVED EVIDENCE:
{evidence_text}

Write a clear business explanation with:
1. What changed versus baseline
2. Which assumptions drove the change
3. Which therapies or patient segments are most affected, if available
4. Why this matters for pharmaceutical planning
5. What a human reviewer should validate before using this in a real client setting
"""


def explain_scenario_with_rag(scenario_context, user_question=None, k=4):
    """
    Boundary:
    - numerical engine calculates scenario_context
    - RAG retrieves evidence
    - Gemini explains the already-calculated result
    """

    load_dotenv()

    query = user_question or (
        "Explain the scenario result using assumptions, methodology, "
        "forecast information set, scenario logic, and model results."
    )

    documents = retrieve_evidence(query=query, k=k)
    evidence_text = format_evidence(documents)

    prompt = build_scenario_explanation_prompt(
        scenario_context=scenario_context,
        evidence_text=evidence_text,
    )

    llm = get_chat_model()

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "retrieved_evidence": documents,
        "evidence_text": evidence_text,
    }


def evidence_preview(query, k=4):
    """
    Retrieves evidence without generating an LLM answer.
    """

    documents = retrieve_evidence(query=query, k=k)

    return format_evidence(documents)
