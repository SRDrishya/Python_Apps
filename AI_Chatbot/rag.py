from openai import OpenAI
from config import API_KEY

import faiss
import numpy as np
import json


class RAG:

    def __init__(
        self,
        index_file="vector_store/faiss_index.bin",
        chunks_file="vector_store/chunks.json",
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4.1-mini",
        top_k=2,
    ):

        self.client = OpenAI(
            api_key=API_KEY
        )

        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.top_k = top_k

        # Load FAISS index
        self.index = faiss.read_index(
            index_file
        )

        # Load chunks
        with open(
            chunks_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.chunks = json.load(f)


    def retrieve(self, question):

        # Create query embedding
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=question
        )

        query_vector = response.data[0].embedding

        # Convert to NumPy
        query_vector = np.array(
            [query_vector],
            dtype="float32"
        )

        # Normalize
        faiss.normalize_L2(
            query_vector
        )

        # Search
        scores, indices = self.index.search(
            query_vector,
            self.top_k
        )

        results = []

        for score, index_number in zip(
            scores[0],
            indices[0]
        ):

            results.append({
                "score": float(score),
                "chunk": self.chunks[index_number]
            })

        return results


    def ask(self, question):

        # Retrieve relevant chunks
        results = self.retrieve(
            question
        )

        # Create context
        context = "\n\n".join(
            result["chunk"]
            for result in results
        )

        # Prompt
        prompt = f"""
You are a helpful assistant answering
questions using the provided context.

Rules:

1. Answer using only the provided context.
2. Do not invent information.
3. If the answer is not present in the context,
   say that you don't have enough information.

Context:

{context}

Question:

{question}
"""

        # Generate answer
        response = self.client.responses.create(
            model=self.chat_model,
            input=prompt
        )

        return {
            "answer": response.output_text,
            "sources": results
        }