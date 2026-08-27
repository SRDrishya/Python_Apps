import json
import re
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import API_KEY
import documents_loader


class Ingestion:

    def __init__(
        self,
        input_file="documents",
        index_file="vector_store/faiss_index.bin",
        chunks_file="vector_store/chunks.json",
        embedding_model="text-embedding-3-small",
        chunk_size=300,
        chunk_overlap=50,
    ):

        self.client = OpenAI(api_key=API_KEY)

        self.input_file = Path(input_file)
        self.index_file = Path(index_file)
        self.chunks_file = Path(chunks_file)

        self.embedding_model = embedding_model

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.encoding = tiktoken.get_encoding(
            "cl100k_base"
        )

    # =========================================================
    # TOKEN COUNTING
    # =========================================================

    def count_tokens(self, text):

        return len(
            self.encoding.encode(text)
        )

    # =========================================================
    # LOAD DOCUMENTS
    # =========================================================

    def load_documents(self):
        """Load documents through the shared document loader module."""
        return documents_loader.load_documents(self)

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    def normalize_text(self, text):

        text = text.replace(
            "\r\n",
            "\n"
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # =========================================================
    # HEADING DETECTION
    # =========================================================

    def is_heading(self, line):

        line = line.strip()

        if not line:
            return False

        # Very long lines are probably normal content.
        if len(line) > 80:
            return False

        # Ignore lines ending like normal sentences.
        if line.endswith(
            (".", "!", "?", ":")
        ):
            return False

        words = line.split()

        # Headings are usually short.
        if len(words) > 8:
            return False

        # Known heading patterns in the current corpus.
        heading_patterns = [
            r".*Policy$",
            r".*Leave$",
            r".*Equipment$",
            r"Office and Conduct",
            r"Travel",
            r"Company Handbook",
        ]

        for pattern in heading_patterns:

            if re.fullmatch(
                pattern,
                line,
                re.IGNORECASE
            ):

                return True

        # Title Case heuristic.
        #
        # Example:
        # "Parental Leave"
        # "Remote Equipment"
        #
        # But avoid ordinary sentence-like text.
        if len(words) <= 4:

            title_case_words = 0

            for word in words:

                if word[0].isupper():
                    title_case_words += 1

            if title_case_words >= len(words) * 0.75:
                return True

        return False

    # =========================================================
    # SPLIT INTO SENTENCES
    # =========================================================

    def split_into_sentences(self, text):

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip()
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # =========================================================
    # BUILD HIERARCHY
    # =========================================================

    def build_sections(self, text):

        text = self.normalize_text(text)

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        sections = []

        current_heading = None
        current_lines = []

        for line in lines:

            if self.is_heading(line):

                # Save previous section.
                if current_lines:

                    sections.append({
                        "heading": current_heading,
                        "text": " ".join(
                            current_lines
                        )
                    })

                current_heading = line
                current_lines = []

            else:

                current_lines.append(line)

        # Save final section.
        if current_lines:

            sections.append({
                "heading": current_heading,
                "text": " ".join(
                    current_lines
                )
            })

        return sections

    # =========================================================
    # CHUNK SECTION BY SENTENCES
    # =========================================================

    def chunk_section(
        self,
        heading,
        text
    ):

        sentences = self.split_into_sentences(
            text
        )

        chunks = []

        current_sentences = []
        current_tokens = 0

        for sentence in sentences:

            sentence_tokens = self.count_tokens(
                sentence
            )

            # -------------------------------------------------
            # Handle a single very large sentence
            # -------------------------------------------------

            if sentence_tokens > self.chunk_size:

                if current_sentences:

                    chunks.append(
                        self.format_chunk(
                            heading,
                            current_sentences
                        )
                    )

                    current_sentences = []
                    current_tokens = 0

                token_ids = self.encoding.encode(
                    sentence
                )

                start = 0

                while start < len(token_ids):

                    end = start + self.chunk_size

                    piece = self.encoding.decode(
                        token_ids[start:end]
                    ).strip()

                    chunks.append(
                        self.format_chunk(
                            heading,
                            [piece]
                        )
                    )

                    if end >= len(token_ids):
                        break

                    start = end - self.chunk_overlap

                continue

            # -------------------------------------------------
            # Add sentence to current chunk
            # -------------------------------------------------

            if (
                current_tokens
                + sentence_tokens
                <= self.chunk_size
            ):

                current_sentences.append(
                    sentence
                )

                current_tokens += sentence_tokens

            else:

                # Save current chunk.
                if current_sentences:

                    chunks.append(
                        self.format_chunk(
                            heading,
                            current_sentences
                        )
                    )

                # Start new chunk with overlap.
                overlap_sentences = []

                overlap_tokens = 0

                for previous_sentence in reversed(
                    current_sentences
                ):

                    previous_tokens = self.count_tokens(
                        previous_sentence
                    )

                    if (
                        overlap_tokens
                        + previous_tokens
                        > self.chunk_overlap
                    ):
                        break

                    overlap_sentences.insert(
                        0,
                        previous_sentence
                    )

                    overlap_tokens += previous_tokens

                current_sentences = (
                    overlap_sentences
                    + [sentence]
                )

                current_tokens = (
                    overlap_tokens
                    + sentence_tokens
                )

        # Save final chunk.
        if current_sentences:

            chunks.append(
                self.format_chunk(
                    heading,
                    current_sentences
                )
            )

        return chunks

    # =========================================================
    # FORMAT CHUNK
    # =========================================================

    def format_chunk(
        self,
        heading,
        sentences
    ):

        content = " ".join(
            sentences
        ).strip()

        if heading:

            return (
                f"{heading}\n\n"
                f"{content}"
            )

        return content

    # =========================================================
    # CREATE ALL CHUNKS
    # =========================================================

    def create_chunks(
        self,
        documents
    ):

        all_chunks = []

        for document in documents:

            metadata = document.get(
                "metadata",
                {}
            )

            source = metadata.get(
                "source",
                "unknown"
            )

            page = metadata.get(
                "page"
            )

            page_start = metadata.get(
                "page_start"
            )

            page_end = metadata.get(
                "page_end"
            )

            file_type = metadata.get(
                "file_type"
            )

            sheet = metadata.get(
                "sheet"
            )

            slide = metadata.get(
                "slide"
            )

            row = metadata.get(
                "row"
            )

            text = document.get(
                "text",
                ""
            )

            if not text.strip():
                continue

            sections = self.build_sections(
                text
            )

            for section_index, section in enumerate(
                sections
            ):

                heading = section["heading"]

                section_text = section["text"]

                chunks = self.chunk_section(
                    heading,
                    section_text
                )

                for chunk_index, chunk in enumerate(
                    chunks
                ):

                    all_chunks.append({

                        "text": chunk,

                        "metadata": {

                            "source": source,

                            "file_type": file_type,

                            "page": page,

                            "page_start": page_start,

                            "page_end": page_end,

                            "sheet": sheet,

                            "slide": slide,

                            "row": row,

                            "section": heading,

                            "section_index":
                                section_index,

                            "chunk_index":
                                chunk_index
                        }
                    })

        return all_chunks

    # =========================================================
    # CREATE EMBEDDINGS
    # =========================================================

    def create_embeddings(
        self,
        chunks
    ):

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print(
            f"Creating embeddings for "
            f"{len(texts)} chunks..."
        )

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        vectors = [
            item.embedding
            for item in response.data
        ]

        return np.array(
            vectors,
            dtype="float32"
        )

    # =========================================================
    # BUILD FAISS INDEX
    # =========================================================

    def build_faiss_index(
        self,
        embeddings
    ):

        # Normalize vectors.
        #
        # Inner product between normalized vectors
        # is equivalent to cosine similarity.

        faiss.normalize_L2(
            embeddings
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )

        return index

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        index,
        chunks
    ):

        faiss.write_index(
            index,
            str(self.index_file)
        )

        with open(
            self.chunks_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Saved FAISS index: "
            f"{self.index_file}"
        )

        print(
            f"Saved chunks: "
            f"{self.chunks_file}"
        )

    # =========================================================
    # INGEST
    # =========================================================

    def ingest(self):

        print(
            "Loading documents..."
        )

        documents = self.load_documents()

        print(
            f"Loaded {len(documents)} documents."
        )

        print(
            "\nCreating hierarchical chunks..."
        )

        chunks = self.create_chunks(
            documents
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        # -----------------------------------------------------
        # Show chunks for inspection
        # -----------------------------------------------------

        for i, chunk in enumerate(
            chunks
        ):

            print(
                f"\n{'=' * 60}"
            )

            print(
                f"CHUNK {i}"
            )

            print(
                f"Source: "
                f"{chunk['metadata']['source']}"
            )

            print(
                f"Section: "
                f"{chunk['metadata']['section']}"
            )

            print(
                f"Tokens: "
                f"{self.count_tokens(chunk['text'])}"
            )

            print(
                f"\n{chunk['text']}"
            )

        # -----------------------------------------------------
        # Embeddings
        # -----------------------------------------------------

        embeddings = self.create_embeddings(
            chunks
        )

        # -----------------------------------------------------
        # FAISS
        # -----------------------------------------------------

        index = self.build_faiss_index(
            embeddings
        )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        self.save(
            index,
            chunks
        )

        print(
            "\nIngestion complete."
        )


if __name__ == "__main__":

    ingestion = Ingestion(
        chunk_size=300,
        chunk_overlap=50
    )

    ingestion.ingest()