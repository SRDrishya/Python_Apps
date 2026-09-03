"""Build a persistent LangChain FAISS index from the documents directory."""

import csv
import json
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import API_KEY


SUPPORTED_SUFFIXES = {".csv", ".html", ".htm", ".json", ".md", ".pdf", ".txt"}


def load_documents(input_dir: str | Path = "documents") -> list[Document]:
    """Load supported local files while retaining useful source metadata."""
    documents = []
    input_path = Path(input_dir)

    # Convert every supported file into LangChain's common Document type so
    # the splitter and vector store can process all formats consistently.
    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            # PDFs are handled page by page so answers can cite the page that
            # supplied the evidence instead of only citing the whole file.
            for page_number, page in enumerate(PdfReader(str(file_path)).pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path.name, "file_type": "pdf", "page": page_number},
                    ))
            continue

        text = file_path.read_text(encoding="utf-8")
        if suffix in {".html", ".htm"}:
            # Embedding HTML markup is noisy; retain its readable text only.
            text = BeautifulSoup(text, "html.parser").get_text("\n", strip=True)
        elif suffix == ".csv":
            # Turn each CSV row into labeled text so natural-language queries
            # can match column names as well as their values.
            rows = list(csv.DictReader(text.splitlines()))
            text = "\n\n".join(
                "\n".join(f"{key}: {value}" for key, value in row.items() if value)
                for row in rows
            )
        elif suffix == ".json":
            # Pretty-print JSON to preserve structure and improve chunking.
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)

        if text.strip():
            documents.append(Document(
                page_content=text.strip(),
                metadata={
                    "source": file_path.name,
                    "file_type": suffix.removeprefix("."),
                    "page": None,
                },
            ))

    if not documents:
        raise ValueError(f"No supported documents found in {input_path}")
    return documents


def build_index(
    input_dir: str | Path = "documents",
    index_dir: str | Path = "vector_store/langchain_faiss",
    documents_file: str | Path = "vector_store/langchain_documents.json",
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 500,
    chunk_overlap: int = 75,
) -> None:
    """Split documents, embed them, and persist the FAISS store plus BM25 corpus."""
    source_documents = load_documents(input_dir)

    # Recursive splitting prefers paragraph and line boundaries before it
    # falls back to words, keeping related information in the same chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(source_documents)
    for chunk_id, chunk in enumerate(chunks):
        # This stable identifier makes individual chunks traceable during
        # debugging and can be used later in source displays or evaluations.
        chunk.metadata["chunk_id"] = chunk_id

    # FAISS handles semantic similarity: it finds passages with related
    # meaning even when the question uses different words.
    # Pass the project's validated key explicitly instead of relying only on
    # LangChain's automatic environment-variable lookup.
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=API_KEY)
    vector_store = FAISS.from_documents(chunks, embeddings)
    Path(index_dir).parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))

    # Save the same chunks separately because BM25 needs the original text
    # corpus at query time, while FAISS stores vectors in its own files.
    serializable = [
        {"page_content": document.page_content, "metadata": document.metadata}
        for document in chunks
    ]
    Path(documents_file).write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Indexed {len(chunks)} chunks in {index_dir}")


if __name__ == "__main__":
    build_index()
