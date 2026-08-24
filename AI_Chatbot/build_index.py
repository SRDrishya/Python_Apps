from openai import OpenAI
from config import API_KEY

import faiss
import numpy as np
import json
import os
import re


client = OpenAI(api_key=API_KEY)


# -----------------------------------
# Configuration
# -----------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 30
CHUNK_OVERLAP = 1

VECTOR_STORE_DIR = "vector_store"

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


# -----------------------------------
# Document
# -----------------------------------

document = """
The project named Aurora was created by Dr. Maya Chen in 2024.
Aurora was designed to help researchers analyze climate data.
The project uses Python for its data processing pipeline.
The initial version was developed at the Northstar Research Lab.
Aurora is currently used by several research teams for climate analysis.
"""


# -----------------------------------
# Create chunks
# -----------------------------------

chunks = chunk_sentences(
    document,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
)


print("Chunks:")

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i}:")
    print(chunk)


# -----------------------------------
# Create embeddings
# -----------------------------------

response = client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=chunks
)


chunk_vectors = [
    item.embedding
    for item in response.data
]


# -----------------------------------
# Convert to NumPy
# -----------------------------------

embedding_matrix = np.array(
    chunk_vectors,
    dtype="float32"
)


# -----------------------------------
# Normalize vectors
# -----------------------------------

faiss.normalize_L2(
    embedding_matrix
)


# -----------------------------------
# Create FAISS index
# -----------------------------------

dimension = embedding_matrix.shape[1]

index = faiss.IndexFlatIP(
    dimension
)


# -----------------------------------
# Add vectors
# -----------------------------------

index.add(
    embedding_matrix
)


print(
    "\nVectors stored:",
    index.ntotal
)


# -----------------------------------
# Create directory
# -----------------------------------

os.makedirs(
    VECTOR_STORE_DIR,
    exist_ok=True
)


# -----------------------------------
# Save FAISS index
# -----------------------------------

faiss.write_index(
    index,
    INDEX_FILE
)


# -----------------------------------
# Save chunks
# -----------------------------------

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
    "\nIndex saved to:",
    INDEX_FILE
)

print(
    "Chunks saved to:",
    CHUNKS_FILE
)