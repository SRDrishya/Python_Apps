from openai import OpenAI
from config import API_KEY

import faiss
import numpy as np
import json
import os
import re


# -----------------------------------
# Configuration
# -----------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 30
CHUNK_OVERLAP = 1

VECTOR_STORE_DIR = "vector_store"
DOCUMENT_FILE = "document.txt"

INDEX_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "faiss_index.bin"
)

CHUNKS_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "chunks.json"
)


# -----------------------------------
# Sentence splitting
# -----------------------------------

def split_into_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return sentences


# -----------------------------------
# Chunking
# -----------------------------------

def chunk_sentences(
    text,
    chunk_size=30,
    overlap=1
):

    sentences = split_into_sentences(text)

    chunks = []

    current_chunk = []

    for sentence in sentences:

        sentence_words = sentence.split()

        current_word_count = sum(
            len(s.split())
            for s in current_chunk
        )

        if (
            current_word_count
            + len(sentence_words)
            <= chunk_size
        ):

            current_chunk.append(sentence)

        else:

            if current_chunk:
                chunks.append(
                    " ".join(current_chunk)
                )

            current_chunk = current_chunk[-overlap:]

            current_chunk.append(sentence)

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


def build_index(document_file=DOCUMENT_FILE):
    client = OpenAI(api_key=API_KEY)

    # Read the knowledge source from a plain-text file so it can be replaced
    # without changing application code.
    with open(document_file, "r", encoding="utf-8") as document_handle:
        document = document_handle.read()

    # Chunking keeps retrieved context small enough for the model while
    # retaining nearby sentences that explain the same fact.
    chunks = chunk_sentences(document, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("The document does not contain any text to index.")

    print("Chunks:")
    for chunk_number, chunk in enumerate(chunks):
        print(f"\nChunk {chunk_number}:\n{chunk}")

    # Embeddings turn each chunk into a vector whose distance represents
    # semantic similarity, allowing retrieval by meaning instead of keywords.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks
    )
    embedding_matrix = np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )

    # Unit-length vectors make inner-product search equivalent to cosine
    # similarity, which is a useful measure for text embeddings.
    faiss.normalize_L2(embedding_matrix)

    # FAISS provides a fast in-memory vector index for nearest-neighbor
    # retrieval at question-answering time.
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as chunks_handle:
        json.dump(chunks, chunks_handle, ensure_ascii=False, indent=2)

    print("\nVectors stored:", index.ntotal)
    print("Index saved to:", INDEX_FILE)
    print("Chunks saved to:", CHUNKS_FILE)


if __name__ == "__main__":
    build_index()