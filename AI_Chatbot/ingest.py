import os
import json

import faiss
import numpy as np

from pypdf import PdfReader
from openai import OpenAI

from config import API_KEY


DOCUMENTS_DIR = "documents"
VECTOR_STORE_DIR = "vector_store"

INDEX_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "faiss_index.bin"
)

CHUNKS_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "chunks.json"
)

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


client = OpenAI(
    api_key=API_KEY
)


def load_pdf(file_path):

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if not text:
            continue

        pages.append({
            "text": text,
            "page": page_number + 1
        })

    return pages


def load_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    return [{
        "text": text,
        "page": None
    }]


def load_document(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":

        return load_pdf(file_path)

    elif extension == ".txt":

        return load_txt(file_path)

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )


def chunk_text(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_all_documents():

    documents = []

    for filename in os.listdir(DOCUMENTS_DIR):

        file_path = os.path.join(
            DOCUMENTS_DIR,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        print(
            f"Loading: {filename}"
        )

        pages = load_document(
            file_path
        )

        for page in pages:

            chunks = chunk_text(
                page["text"]
            )

            for chunk in chunks:

                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "page": page["page"]
                    }
                })

    return documents


def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for {len(texts)} chunks..."
    )

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    # Normalize because your retrieval code
    # also normalizes the query vector.
    faiss.normalize_L2(
        embeddings
    )

    return embeddings


def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def save_vector_store(index, chunks):

    os.makedirs(
        VECTOR_STORE_DIR,
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_FILE
    )

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved FAISS index to {INDEX_FILE}"
    )

    print(
        f"Saved chunks to {CHUNKS_FILE}"
    )


def main():

    chunks = load_all_documents()

    print(
        f"Total chunks: {len(chunks)}"
    )

    embeddings = create_embeddings(
        chunks
    )

    index = build_faiss_index(
        embeddings
    )

    save_vector_store(
        index,
        chunks
    )

    print(
        "Ingestion complete."
    )


if __name__ == "__main__":
    main()