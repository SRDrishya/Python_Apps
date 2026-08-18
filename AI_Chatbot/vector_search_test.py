from openai import OpenAI
from config import API_KEY
import math


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

    return dot_product / (magnitude_a * magnitude_b)

def chunk_text(text, chunk_size=20, overlap=5):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

document= """
Python is a high-level, interpreted programming language known for its simplicity and readability. 
It was created by Guido van Rossum and first released in 1991. 
Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. 
It has a large standard library that provides tools suited to many tasks, making it a versatile choice for developers.
"""

documents = [
    "Python lists store multiple values in a single variable.",
    "Python dictionaries store data using key-value pairs.",
    "Python functions allow us to organize reusable code.",
    "Machine learning allows computers to learn patterns from data.",
    "Neural networks are commonly used in deep learning.",
]
chunks = chunk_text(
    document,
    chunk_size=20,
    overlap=5,
)
for i, chunk in enumerate(chunks):

    print(f"\nChunk {i}:")
    print(chunk)

# Create embeddings for the documents
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents,
)

document_vectors = [
    item.embedding
    for item in response.data
]


query = "How do I store multiple items in Python?"


# Create embedding for the query
query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query,
)

query_vector = query_response.data[0].embedding


# Compare query with every document
results = []

for document, vector in zip(documents, document_vectors):

    score = cosine_similarity(
        query_vector,
        vector,
    )

    results.append({
        "document": document,
        "score": score,
    })


# Highest similarity first
results.sort(
    key=lambda x: x["score"],
    reverse=True,
)


print("Query:")
print(query)

print("\nResults:")

for result in results:
    print(
        f"{result['score']:.4f} -> "
        f"{result['document']}"
    )