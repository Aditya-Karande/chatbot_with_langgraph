from typing import TypedDict

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_postgres import PGVector
from langgraph.graph import START, END, StateGraph

from backend.config import embeddings,DB_URL, LLM

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="study_notes",
    connection= DB_URL,
    use_jsonb=True,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def _clean_text(text: str) -> str:
    """Strip NUL bytes and other control characters that Postgres text
    columns can't store (common artifact of PDF text extraction)."""
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch.isprintable())

def _load_text_file(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = _clean_text(f.read())
    return [
        Document(page_content=chunk, metadata={"source": path})
        for chunk in splitter.split_text(text)
    ]

def _load_pdf_file(path: str) -> list[Document]:
    try:
        pages = PyPDFLoader(path).load()
    except Exception as e:
        print(f"Skipping {path}: could not read PDF ({e})")
        return []
    docs = []
    for page in pages:
        cleaned = _clean_text(page.page_content)
        if not cleaned.strip():
            continue  # skip blank/scanned pages with no extractable text
        for chunk in splitter.split_text(cleaned):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": path, "page": page.metadata.get("page")},
                )
            )
    return docs

import hashlib

def ingest_files(paths: list[str]) -> int:
    """Chunk + embed the given .md/.txt/.pdf files into the RAG vector store.
    Skips chunks whose exact text is already stored, to avoid duplicate
    entries when the same file gets uploaded/ingested more than once.
    Returns the number of NEW chunks actually ingested."""
    documents: list[Document] = []
    for path in paths:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            documents.extend(_load_pdf_file(path))
        elif ext in (".md", ".txt"):
            documents.extend(_load_text_file(path))
        else:
            print(f"Skipping unsupported file type: {path}")

    new_documents = []
    for doc in documents:
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        existing = vector_store.similarity_search(doc.page_content, k=1)
        if existing and existing[0].page_content == doc.page_content:
            continue  # already stored, skip
        doc.metadata["content_hash"] = content_hash
        new_documents.append(doc)

    if new_documents:
        vector_store.add_documents(new_documents)
    return len(new_documents)

class RAGState(TypedDict):
    query: str
    documents: list[str]
    context: str

def retrieve_node(state:RAGState) -> dict:
    # results = vector_store.similarity_search(state["query"], k=4)
    # return {"documents":[d.page_content for d in results]}
    query = state.get("query", "")
    if not query or not query.strip():
        return {"documents": []}
    results = vector_store.similarity_search(query, k=4)
    return {"documents": [d.page_content for d in results]}


def grade_node(state: RAGState) -> dict:
    """Corrective-RAG step: ask the LLM to drop chunks that don't actually
    answer the query, instead of blindly stuffing everything into context.
    Defensive: some extracted text (especially from PDFs) can contain
    characters that break message serialization to certain providers — if
    grading a chunk fails, we keep it rather than crash the whole search."""
    kept = []
    for doc in state["documents"]:
        # Coerce to plain str and strip characters that sometimes cause
        # malformed API requests (control chars, null bytes, etc.)
        clean_doc = "".join(ch for ch in str(doc) if ch.isprintable() or ch in "\n\t")

        try:
            verdict = (
                LLM.invoke(
                    [
                        HumanMessage(
                            content=(
                                "Does this passage help answer the query? "
                                "Reply with only YES or NO.\n\n"
                                f"Query: {state['query']}\n\nPassage: {clean_doc}"
                            )
                        )
                    ]
                )
                .content.strip()
                .upper()
            )
        except Exception as e:
            print(f"grade_node: skipping grading for one chunk due to error: {e}")
            kept.append(doc)  # fail open — keep it rather than lose it
            continue

        if verdict.startswith("YES"):
            kept.append(doc)
    return {"documents": kept}

def generate_context_node(state: RAGState) -> dict:
    if not state["documents"]:
        return{"context":"No relevant notes were found for this query."}

    return{"context":"\n\n---\n\n".join(state["documents"])}

# building sub-graph
rag_builder = StateGraph(RAGState)

rag_builder.add_node("retreive",retrieve_node) 
rag_builder.add_node("grade",grade_node)
rag_builder.add_node("generate_context",generate_context_node)

rag_builder.add_edge(START,"retreive")
rag_builder.add_edge("retreive","grade")
rag_builder.add_edge("grade","generate_context")
rag_builder.add_edge("generate_context",END)

rag_subgraph = rag_builder.compile()

# converting into a tool.
@tool
def search_study_notes(query: str) -> str:
    """Search the user's own study notes / documents for relevant information.
    Use this for questions about the user's personal material, not general
    knowledge (use wikipedia_tool for that)."""
    if not query or not query.strip():
        return "No search query was provided, so no notes were searched."

    try:
        result = rag_subgraph.invoke({"query": query, "documents": [], "context": ""})
    except Exception as e:
        print(f"search_study_notes: RAG subgraph failed: {e}")
        return "Something went wrong while searching your notes. Please try rephrasing your question."

    return result.get("context", "No relevant notes were found for this query.")