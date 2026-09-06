"""Build the document index used by the LangGraph RAG workflow."""

from langchain_ingest import build_index as build_langchain_index


# LangGraph handles the application workflow; ingestion still uses the same
# FAISS/BM25-compatible document preparation and embedding process.
def build_index() -> None:
    build_langchain_index(
        index_dir="vector_store/langgraph_faiss",
        documents_file="vector_store/langgraph_documents.json",
    )


if __name__ == "__main__":
    build_index()
