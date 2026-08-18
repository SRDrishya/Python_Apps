from openai import OpenAI
from config import API_KEY
import math
import re
import numpy as np
import faiss

client = OpenAI(api_key=API_KEY)


def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


def split_into_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return sentences


def chunk_sentences(text, chunk_size=40, overlap=1):

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
            current_word_count + len(sentence_words)
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


document = """
The project named Aurora was created by Dr. Maya Chen in 2024.
Aurora was designed to help researchers analyze climate data.
The project uses Python for its data processing pipeline.
The initial version was developed at the Northstar Research Lab.
"""


# -----------------------------
# 1. Chunk the document
# -----------------------------

chunks = chunk_sentences(
    document,
    chunk_size=30,
    overlap=1,
)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks,
)
chunk_vectors = [
    item.embedding
    for item in response.data
]
print("Chunks:")
embedding_matrix = np.array(
    chunk_vectors,
    dtype="float32"
)

dimension = embedding_matrix.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)





# -----------------------------
# 3. Create query embedding
# -----------------------------

query = "Who was the CEO of the Aurora project?"

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query,
)

query_vector = query_response.data[0].embedding

query_vector_np = np.array(
    [query_vector],
    dtype="float32"
)
top_k = 2

distances, indices = index.search(
    query_vector_np,
    top_k
)

print("\nRetrieved chunks:")

for i in indices[0]:

    print(
        f"\nChunk {i}:"
    )

    print(chunks[i])