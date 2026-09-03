"""LangChain hybrid retrieval and grounded answer generation."""

import json
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import API_KEY


ABSTENTION = "I don't have that information in the provided documents."


class LangChainRAG:
    def __init__(
        self,
        index_dir: str | Path = "vector_store/langchain_faiss",
        documents_file: str | Path = "vector_store/langchain_documents.json",
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4.1-mini",
        top_k: int = 4,
    ):
        # Reload the chunk documents for lexical retrieval and source details.
        records = json.loads(Path(documents_file).read_text(encoding="utf-8"))
        documents = [Document(**record) for record in records]

        # FAISS performs semantic search over the vectors created by ingestion.
        vector_store = FAISS.load_local(
            str(index_dir),
            OpenAIEmbeddings(model=embedding_model, api_key=API_KEY),
            allow_dangerous_deserialization=True,
        )
        semantic = vector_store.as_retriever(search_kwargs={"k": top_k})

        # BM25 performs exact keyword matching, which is especially useful for
        # names, IDs, product codes, and other terms embeddings may blur.
        lexical = BM25Retriever.from_documents(documents, k=top_k)

        # EnsembleRetriever merges both ranked lists. The larger semantic
        # weight favors meaning while BM25 preserves precise keyword matches.
        self.retriever = EnsembleRetriever(
            retrievers=[semantic, lexical],
            weights=[0.65, 0.35],
        )
        self.llm = ChatOpenAI(model=chat_model, temperature=0, api_key=API_KEY)

    @staticmethod
    def _prompt() -> ChatPromptTemplate:
        # The system message limits generation to retrieved evidence and
        # requires the model to expose which labeled passage supports a claim.
        return ChatPromptTemplate.from_messages([
            ("system", """You answer questions using only the supplied document context.
Do not guess or use outside knowledge. If the context does not support the answer,
say exactly: \"I don't have that information in the provided documents.\"
Cite every factual claim with the matching source label, such as [S1].

Context:
{context}"""),
            ("human", "{input}"),
        ])

    @staticmethod
    def _label_documents(documents: list[Document]) -> list[Document]:
        labeled = []
        seen = set()
        for document in documents:
            # Hybrid retrieval can return the same passage twice. Deduplicate
            # it before assigning citations so each label is unambiguous.
            identity = (document.page_content, tuple(sorted(document.metadata.items())))
            if identity in seen:
                continue
            seen.add(identity)
            metadata = dict(document.metadata)
            metadata["citation"] = f"S{len(labeled) + 1}"
            labeled.append(Document(page_content=document.page_content, metadata=metadata))
        return labeled

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        parts = []
        for document in documents:
            metadata = document.metadata
            location = metadata.get("source", "unknown")
            if metadata.get("page"):
                location += f", page {metadata['page']}"
            # Put citation labels directly beside their evidence so the model
            # can reliably reproduce them in the answer.
            parts.append(f"[{metadata['citation']}] Source: {location}\nText: {document.page_content}")
        return "\n\n".join(parts)

    def retrieve(self, question: str) -> list[Document]:
        # invoke() runs the configured semantic and lexical retrievers and
        # returns their combined ranking for this question.
        return self._label_documents(self.retriever.invoke(question))

    def ask(self, question: str) -> dict:
        documents = self.retrieve(question)
        if not documents:
            # Avoid an unsupported model answer when retrieval found no
            # evidence at all.
            return {"answer": ABSTENTION, "sources": []}

        # Format the labeled evidence into the prompt, then let the chat model
        # produce a grounded answer using the same prompt contract every time.
        messages = self._prompt().format_messages(
            context=self._format_context(documents),
            input=question,
        )
        response = self.llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": documents,
        }
