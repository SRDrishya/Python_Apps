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

    return dot_product / (magnitude_a * magnitude_b)


# Previous conversation = candidate memories
memories = [
    "My name is Drishya.",
    "I am learning Python.",
    "I want to build an AI chatbot.",
    "I live in Chennai.",
]

# New user message = query
query = "What is my name?"


# Create embedding for the query
query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query,
)

query_vector = query_response.data[0].embedding


# Create embeddings for all memories
memory_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=memories,
)


# Compare query against every memory
results = []

for i, item in enumerate(memory_response.data):
    score = cosine_similarity(
        query_vector,
        item.embedding,
    )

    results.append({
        "memory": memories[i],
        "score": score,
    })


# Sort highest similarity first
results.sort(
    key=lambda x: x["score"],
    reverse=True,
)


print("Query:")
print(query)

print("\nRelevant memories:")

for result in results:
    print(
        f"{result['score']:.4f} -> {result['memory']}"
    )