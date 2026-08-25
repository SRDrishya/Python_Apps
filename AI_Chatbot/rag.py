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
        min_score=0.40
    ):

        self.client = OpenAI(
            api_key=API_KEY
        )

        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.top_k = top_k
        self.min_score = min_score

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

        # Retrieval: embed the question so FAISS can find semantically similar
        # document chunks rather than relying on exact word matches.
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

        # Search the vector index for the most relevant chunks.
        scores, indices = self.index.search(
            query_vector,
            self.top_k
        )

        results = []

        for score, index_number in zip(
            scores[0],
            indices[0]
        ):

            if index_number < 0:
                continue
            score = float(score)

            # Reject weak / irrelevant matches
            if score < self.min_score:
                continue
            results.append({
                "score": float(score),
                "text": self.chunks[index_number]["text"],
                "metadata": self.chunks[index_number]["metadata"]
            })

        return results


    def ask(self, question):

        # Retrieval: find the document passages that are most relevant to the
        # user's question.
        results = self.retrieve(
            question
        )

        # Augmentation: combine retrieved passages into the context supplied
        # to the language model, grounding its answer in the document.
        context = "\n\n".join(
            result["text"]
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

        # Generation: ask the language model to answer using only that context
        # so unsupported facts are less likely to be introduced.
        response = self.client.responses.create(
            model=self.chat_model,
            input=prompt
        )

        return {
            "answer": response.output_text,
            "sources": results
        }


def ask(question):
    """Answer a question and expose retrieved chunks for the example script."""
    result = RAG().ask(question)
    return {
        "answer": result["answer"],
        # Keep the old example-script field name while RAG internally uses
        # ``text`` for every retrieved document passage.
        "chunks": [
            {
                "score": source["score"],
                "chunk": source["text"],
                "metadata": source["metadata"],
            }
            for source in result["sources"]
        ]
    }