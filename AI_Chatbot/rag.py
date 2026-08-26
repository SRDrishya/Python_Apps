from openai import OpenAI
from config import API_KEY

import faiss
import numpy as np
import json
import re
import logging
from rank_bm25 import BM25Okapi


logger = logging.getLogger(__name__)


class RAG:

    def __init__(
        self,
        index_file="vector_store/faiss_index.bin",
        chunks_file="vector_store/chunks.json",
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4.1-mini",
        top_k=2,
        min_score=0.40,
        rrf_k=60,
        candidate_k=20,
    ):

        self.client = OpenAI(
            api_key=API_KEY
        )

        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.top_k = top_k
        self.min_score = min_score
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

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

        self.bm25 = BM25Okapi(
            [
                self._tokenize(chunk["text"])
                for chunk in self.chunks
            ]
        )


    @staticmethod
    def _tokenize(text):
        """Use the same simple normalization for queries and documents."""
        return re.findall(r"\w+", text.lower())


    def _semantic_search(self, question):

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=question
        )

        query_vector = np.array(
            [response.data[0].embedding],
            dtype="float32"
        )

        faiss.normalize_L2(query_vector)

        candidate_count = min(
            max(self.candidate_k, self.top_k),
            self.index.ntotal
        )
        scores, indices = self.index.search(
            query_vector,
            candidate_count
        )

        ranked = []

        for rank, (score, index_number) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):
            if index_number < 0 or score < self.min_score:
                continue

            ranked.append({
                "index": int(index_number),
                "rank": rank,
                "score": float(score),
            })

        return ranked


    def _bm25_search(self, question):

        scores = self.bm25.get_scores(
            self._tokenize(question)
        )
        candidate_count = min(
            max(self.candidate_k, self.top_k),
            len(scores)
        )
        ranked_indices = [
            index_number
            for index_number in np.argsort(scores)[::-1]
            if scores[index_number] > 0
        ][:candidate_count]

        return [
            {
                "index": int(index_number),
                "rank": rank,
                "score": float(scores[index_number]),
            }
            for rank, index_number in enumerate(ranked_indices, start=1)
        ]


    def _rrf(self, semantic_results, bm25_results):

        fused = {}
        branch_scores = {}

        for result in semantic_results:
            index_number = result["index"]
            fused[index_number] = fused.get(index_number, 0.0) + (
                1.0 / (self.rrf_k + result["rank"])
            )
            branch_scores.setdefault(index_number, {})[
                "semantic_score"
            ] = result["score"]

        for result in bm25_results:
            index_number = result["index"]
            fused[index_number] = fused.get(index_number, 0.0) + (
                1.0 / (self.rrf_k + result["rank"])
            )
            branch_scores.setdefault(index_number, {})[
                "bm25_score"
            ] = result["score"]

        ranked = sorted(
            fused,
            key=fused.get,
            reverse=True
        )[:self.top_k]

        return [
            {
                "citation": f"S{position}",
                "score": fused[index_number],
                "rrf_score": fused[index_number],
                **branch_scores[index_number],
                "text": self.chunks[index_number]["text"],
                "metadata": self.chunks[index_number]["metadata"],
            }
            for position, index_number in enumerate(ranked, start=1)
        ]


    def retrieve(self, question):
        # RRF combines rank positions, so BM25 and cosine scores do not need
        # to be calibrated onto the same numeric scale.
        semantic_results = self._semantic_search(question)
        bm25_results = self._bm25_search(question)
        return self._rrf(semantic_results, bm25_results)


    @staticmethod
    def format_context(results):
        """Format retrieved passages with stable citation labels and metadata."""
        context_parts = []

        for result in results:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page")
            location = f"{source}, page {page}" if page else source
            context_parts.append(
                f"[{result['citation']}] Source: {location}\n"
                f"Text: {result['text']}"
            )

        return "\n\n".join(context_parts)


    def ask(self, question):

        # Retrieval: find the document passages that are most relevant to the
        # user's question.
        results = self.retrieve(
            question
        )

        if not results:
            logger.info("No retrieval results for question=%r", question)
            return {
                "answer": "I don't have that information in the provided documents.",
                "sources": [],
            }

        # Augmentation: combine retrieved passages into the context supplied
        # to the language model, grounding its answer in the document.
        context = self.format_context(results)

        # Prompt
        prompt = f"""
You are a helpful assistant answering
questions using the provided context.

Rules:

1. Answer using only the provided context.
2. Do not invent information.
3. Cite every factual claim with the matching source label, such as [S1].
4. If the answer is not present in the context,
   say exactly: "I don't have that information in the provided documents."

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