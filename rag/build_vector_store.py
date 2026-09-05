# ============================================================
# Build FAISS Vector Store for RAG Evidence Retrieval
# ============================================================

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_DOCS_DIR = PROJECT_ROOT / "rag" / "evidence_docs"
VECTOR_STORE_DIR = PROJECT_ROOT / "rag" / "vector_store"


def load_evidence_documents():
    """
    Loads markdown evidence documents.

    Inputs:
    - rag/evidence_docs/*.md

    Output:
    - LangChain Document objects with source metadata
    """

    if not EVIDENCE_DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Evidence docs folder not found: {EVIDENCE_DOCS_DIR}"
        )

    loader = DirectoryLoader(
        path=str(EVIDENCE_DOCS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(
            "No markdown evidence documents found in rag/evidence_docs."
        )

    return documents


def split_documents(documents):
    """
    Splits documents into smaller chunks for retrieval.

    Why chunking matters:
    - Very large documents retrieve too much irrelevant text.
    - Very tiny chunks lose context.
    - This size is a practical starting point for methodology documents.
    """

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
    Builds and saves the FAISS vector store.

    The OpenAI API key must be available in the environment as:
    OPENAI_API_KEY
    """

    load_dotenv()

    documents = load_evidence_documents()
    chunks = split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(VECTOR_STORE_DIR))

    print("FAISS vector store created successfully.")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Vector store saved to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_vector_store()
