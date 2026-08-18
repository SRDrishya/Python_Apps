from openai import OpenAI
from config import API_KEY
import math
import re


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


print("Chunks:")

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i}:")
    print(chunk)

    print(
        "Word count:",
        len(chunk.split())
    )


# -----------------------------
# 2. Create embeddings
# -----------------------------

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks,
)


chunk_vectors = [
    item.embedding
    for item in response.data
]


# -----------------------------
# 3. Create query embedding
# -----------------------------

query = "Who was the CEO of the Aurora project?"

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query,
)

query_vector = query_response.data[0].embedding


# -----------------------------
# 4. Compare query with chunks
# -----------------------------

results = []

for chunk, vector in zip(
    chunks,
    chunk_vectors,
):

    score = cosine_similarity(
        query_vector,
        vector,
    )

    results.append({
        "chunk": chunk,
        "score": score,
    })


# -----------------------------
# 5. Sort results
# -----------------------------

results.sort(
    key=lambda x: x["score"],
    reverse=True,
)

top_k = 2

top_results = results[:top_k]

# -----------------------------
# 6. Display results
# -----------------------------

print("\nQuery:")
print(query)

print("\nResults:")

for result in top_results:

    print(
        f"{result['score']:.4f} -> "
        f"{result['chunk']}"
    )

context = "\n\n".join(
    result["chunk"]
    for result in top_results
)

response = client.responses.create(
    model="gpt-4.1-mini",
    input=f"""
Answer the user's question using only the provided context.

Context:
{context}

Question:
{query}
"""
)

answer = response.output_text

print("\nAnswer:")
print(answer)