"""A stateful LangGraph implementation of the grounded RAG workflow."""

import json
from pathlib import Path
from typing import TypedDict

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

from config import API_KEY


ABSTENTION = "I don't have that information in the provided documents."


class RAGState(TypedDict, total=False):
    """Data passed between graph nodes during one question-answer turn."""

    question: str
    documents: list[Document]
    answer: str


class LangGraphRAG:
    def __init__(
        self,
        index_dir: str | Path = "vector_store/langgraph_faiss",
        documents_file: str | Path = "vector_store/langgraph_documents.json",
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4.1-mini",
        top_k: int = 4,
    ):
        records = json.loads(Path(documents_file).read_text(encoding="utf-8"))
        documents = [Document(**record) for record in records]

        # The graph's retrieval node combines semantic and exact keyword
        # search, preserving the hybrid behavior of the original RAG system.
        vector_store = FAISS.load_local(
            str(index_dir),
            OpenAIEmbeddings(model=embedding_model, api_key=API_KEY),
            allow_dangerous_deserialization=True,
        )
        semantic = vector_store.as_retriever(search_kwargs={"k": top_k})
        lexical = BM25Retriever.from_documents(documents, k=top_k)
        self.retriever = EnsembleRetriever(
            retrievers=[semantic, lexical],
            weights=[0.65, 0.35],
        )
        self.llm = ChatOpenAI(model=chat_model, temperature=0, api_key=API_KEY)
        self.graph = self._build_graph()

    @staticmethod
    def _prompt() -> ChatPromptTemplate:
        # The prompt is the generation node's grounding boundary: the model
        # receives only the labeled passages selected by the retrieval node.
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
            # Both retrievers may return the same chunk, so deduplicate before
            # assigning labels that the model will use in its answer.
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
            parts.append(
                f"[{metadata['citation']}] Source: {location}\n"
                f"Text: {document.page_content}"
            )
        return "\n\n".join(parts)

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Graph node that retrieves and labels evidence for the question."""
        documents = self.retriever.invoke(state["question"])
        return {"documents": self._label_documents(documents)}

    def _generate_node(self, state: RAGState) -> RAGState:
        """Graph node that generates an answer from retrieved evidence only."""
        documents = state.get("documents", [])
        if not documents:
            return {"answer": ABSTENTION}

        messages = self._prompt().format_messages(
            context=self._format_context(documents),
            input=state["question"],
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _build_graph(self):
        # StateGraph makes each RAG phase explicit and leaves room for future
        # branches such as query rewriting, validation, or human approval.
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def ask(self, question: str) -> dict:
        """Run one question through the compiled retrieval-generation graph."""
        result = self.graph.invoke({"question": question})
        return {
            "answer": result.get("answer", ABSTENTION),
            "sources": result.get("documents", []),
        }

    def retrieve(self, question: str) -> list[Document]:
        """Expose retrieval separately for evaluation and debugging."""
        return self._label_documents(self.retriever.invoke(question))
