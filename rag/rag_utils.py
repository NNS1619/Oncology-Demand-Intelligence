# ============================================================
# RAG Utilities for Grounded Scenario Explanation
# ============================================================

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_STORE_DIR = PROJECT_ROOT / "rag" / "vector_store"


def load_vector_store():
    """
    Loads the saved FAISS vector store.

    This assumes rag/build_vector_store.py has already been run.
    """

    load_dotenv()

    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found. Run rag/build_vector_store.py first."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.load_local(
        folder_path=str(VECTOR_STORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def retrieve_evidence(query, k=4):
    """
    Retrieves the most relevant evidence chunks.

    Inputs:
    - query: user or app-generated question
    - k: number of chunks to retrieve

    Output:
    - list of retrieved documents
    """

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    documents = retriever.invoke(query)

    return documents


def format_evidence(documents):
    """
    Converts retrieved documents into a readable evidence block.
    """

    formatted_chunks = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown source")

        formatted_chunks.append(
            f"[Evidence {index} | Source: {source}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def build_scenario_explanation_prompt(scenario_context, evidence_text):
    """
    Builds a controlled prompt for scenario explanation.

    Important:
    The LLM receives numerical outputs from the model.
    It should explain those outputs, not recalculate them.
    """

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
    Generates a grounded scenario explanation.

    Boundary:
    - Numerical engine calculates scenario_context.
    - RAG retrieves evidence.
    - LLM explains.
    """

    load_dotenv()

    if user_question is None:
        query = (
            "Explain the scenario result using assumptions, methodology, "
            "forecast information set, scenario logic, and model results."
        )
    else:
        query = user_question

    documents = retrieve_evidence(query=query, k=k)
    evidence_text = format_evidence(documents)

    prompt = build_scenario_explanation_prompt(
        scenario_context=scenario_context,
        evidence_text=evidence_text,
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "retrieved_evidence": documents,
        "evidence_text": evidence_text,
    }


def evidence_preview(query, k=4):
    """
    Retrieves evidence without calling the LLM.

    Useful for debugging retrieval quality.
    """

    documents = retrieve_evidence(query=query, k=k)

    return format_evidence(documents)
