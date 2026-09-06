"""A stateful LangGraph implementation of the grounded RAG workflow."""

import json
import re
import uuid
from pathlib import Path
from typing import Any, TypedDict

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from config import API_KEY


ABSTENTION = "I don't have that information in the provided documents."


class RAGState(TypedDict, total=False):
    """Data passed between nodes during one question-answer turn."""

    question: str
    plan: list[str]
    step_index: int
    step_question: str
    query: str
    documents: list[Document]
    step_documents: list[Document]
    step_answers: list[str]
    answer: str
    retrieval_attempt: int
    generation_attempt: int
    max_retries: int
    approval_required: bool
    approval: Any
    validation_errors: list[str]


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

    @staticmethod
    def _parse_plan(text: str, question: str) -> list[str]:
        steps = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned and len(cleaned) > 8:
                steps.append(cleaned)
        return steps[:4] or [question]

    @staticmethod
    def _merge_documents(existing: list[Document], new: list[Document]) -> list[Document]:
        merged = []
        seen = set()
        for document in [*existing, *new]:
            identity = (document.page_content, tuple(sorted(document.metadata.items())))
            if identity in seen:
                continue
            seen.add(identity)
            metadata = dict(document.metadata)
            metadata["citation"] = f"S{len(merged) + 1}"
            merged.append(Document(page_content=document.page_content, metadata=metadata))
        return merged

    def _plan_node(self, state: RAGState) -> RAGState:
        prompt = [
            ("system", """Break the user's request into the smallest useful ordered research steps.
Return one step per line, with no explanation. Use one line when the request is already simple."""),
            ("human", state["question"]),
        ]
        try:
            response = self.llm.invoke(prompt)
            plan = self._parse_plan(str(response.content), state["question"])
        except Exception:
            plan = [state["question"]]
        return {
            "plan": plan,
            "step_index": 0,
            "step_question": plan[0],
            "query": plan[0],
            "documents": [],
            "step_documents": [],
            "step_answers": [],
            "retrieval_attempt": 0,
            "generation_attempt": 0,
            "validation_errors": [],
        }

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieve evidence for the current agent step."""
        documents = self._label_documents(self.retriever.invoke(state["query"]))
        return {
            "documents": self._merge_documents(state.get("documents", []), documents),
            "step_documents": documents,
        }

    def _rewrite_query_node(self, state: RAGState) -> RAGState:
        prompt = [
            ("system", "Rewrite the query for document retrieval. Return only the rewritten query."),
            ("human", state["step_question"]),
        ]
        try:
            response = self.llm.invoke(prompt)
            query = str(response.content).strip()
        except Exception:
            query = f"{state['step_question']} relevant details"
        return {
            "query": query or state["step_question"],
            "retrieval_attempt": state.get("retrieval_attempt", 0) + 1,
        }

    def _generate_node(self, state: RAGState) -> RAGState:
        """Generate one grounded answer for the current agent step."""
        documents = state.get("step_documents", [])
        if not documents:
            return {"answer": ABSTENTION, "generation_attempt": 0}

        messages = self._prompt().format_messages(
            context=self._format_context(documents),
            input=state["step_question"],
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _validate_node(self, state: RAGState) -> RAGState:
        answer = state.get("answer", "").strip()
        errors = []
        if not answer:
            errors.append("The answer is empty.")
        if state.get("step_documents") and answer != ABSTENTION and not re.search(r"\[S\d+\]", answer):
            errors.append("Every grounded answer must cite at least one source label.")
        return {"validation_errors": errors}

    def _retry_generation_node(self, state: RAGState) -> RAGState:
        return {"generation_attempt": state.get("generation_attempt", 0) + 1}

    def _advance_step_node(self, state: RAGState) -> RAGState:
        answers = [*state.get("step_answers", []), state.get("answer", ABSTENTION)]
        next_index = state["step_index"] + 1
        plan = state["plan"]
        return {
            "step_answers": answers,
            "step_index": next_index,
            "step_question": plan[next_index],
            "query": plan[next_index],
            "step_documents": [],
            "retrieval_attempt": 0,
            "generation_attempt": 0,
            "validation_errors": [],
        }

    def _fallback_step_node(self, state: RAGState) -> RAGState:
        return {"answer": ABSTENTION}

    def _approval_node(self, state: RAGState) -> RAGState:
        decision = interrupt({
            "type": "human_approval",
            "question": state["question"],
            "answer": state.get("answer", ABSTENTION),
            "sources": len(state.get("documents", [])),
        })
        return {"approval": decision}

    def _finalize_node(self, state: RAGState) -> RAGState:
        answers = [*state.get("step_answers", []), state.get("answer", ABSTENTION)]
        if len(answers) == 1:
            return {"answer": answers[0]}
        prompt = [
            ("system", """Combine the step answers into one concise answer.
Use only the supplied step answers and preserve their source citations. If they do not answer
the question, say exactly: I don't have that information in the provided documents."""),
            ("human", f"Question: {state['question']}\nStep answers:\n" + "\n".join(answers)),
        ]
        response = self.llm.invoke(prompt)
        return {"answer": response.content}

    @staticmethod
    def _reject_node(state: RAGState) -> RAGState:
        return {"answer": ABSTENTION}

    @staticmethod
    def _route_retrieval(state: RAGState) -> str:
        if state.get("step_documents") or state.get("retrieval_attempt", 0) >= state.get("max_retries", 2):
            return "generate"
        return "rewrite_query"

    @staticmethod
    def _route_validation(state: RAGState) -> str:
        if state.get("validation_errors") and state.get("generation_attempt", 0) < state.get("max_retries", 2):
            return "retry_generation"
        if state.get("validation_errors"):
            return "fallback_step"
        if state["step_index"] + 1 < len(state["plan"]):
            return "advance_step"
        return "approval" if state.get("approval_required") else "finalize"

    @staticmethod
    def _route_approval(state: RAGState) -> str:
        decision = state.get("approval") or {}
        return "finalize" if decision.get("approved", True) else "reject"

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("retry_generation", self._retry_generation_node)
        workflow.add_node("advance_step", self._advance_step_node)
        workflow.add_node("fallback_step", self._fallback_step_node)
        workflow.add_node("approval", self._approval_node)
        workflow.add_node("reject", self._reject_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_conditional_edges("retrieve", self._route_retrieval)
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate", "validate")
        workflow.add_conditional_edges(
            "validate",
            self._route_validation,
            {
                "retry_generation": "retry_generation",
                "fallback_step": "fallback_step",
                "advance_step": "advance_step",
                "approval": "approval",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("retry_generation", "generate")
        workflow.add_edge("advance_step", "retrieve")
        workflow.add_conditional_edges(
            "fallback_step",
            lambda state: "approval" if state.get("approval_required") else "finalize",
            {"approval": "approval", "finalize": "finalize"},
        )
        workflow.add_conditional_edges(
            "approval",
            lambda state: "approval" if state.get("approval_required") and not state.get("approval") else self._route_approval(state),
            {"approval": "approval", "finalize": "finalize", "reject": "reject"},
        )
        workflow.add_edge("reject", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile(checkpointer=MemorySaver())

    def ask(
        self,
        question: str,
        *,
        require_approval: bool = False,
        approval: dict | None = None,
        thread_id: str | None = None,
        max_retries: int = 2,
    ) -> dict:
        """Run the agent, optionally pausing for human approval.

        Resume an approval pause by calling this method with the returned
        ``thread_id`` and ``approval={"approved": True}`` or ``False``.
        """
        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        if approval is not None:
            result = self.graph.invoke(Command(resume=approval), config)
        else:
            result = self.graph.invoke({
                "question": question,
                "approval_required": require_approval,
                "max_retries": max(0, max_retries),
            }, config)
        # A resumed interrupt may return only the latest node update. Read the
        # checkpointed state so sources and the final answer are preserved.
        checkpoint = self.graph.get_state(config)
        state = checkpoint.values if checkpoint else result
        pending = result.get("__interrupt__", [])
        return {
            "answer": state.get("answer", result.get("answer", ABSTENTION)),
            "sources": state.get("documents", result.get("documents", [])),
            "thread_id": thread_id,
            "awaiting_approval": bool(pending),
            "approval_request": pending[0].value if pending else None,
            "validation_errors": state.get("validation_errors", result.get("validation_errors", [])),
        }

    def retrieve(self, question: str) -> list[Document]:
        """Expose retrieval separately for evaluation and debugging."""
        return self._label_documents(self.retriever.invoke(question))
