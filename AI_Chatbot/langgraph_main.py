"""Command-line entry point for the LangGraph RAG chatbot."""

import os

from langgraph_rag import LangGraphRAG


if __name__ == "__main__":
    # Build the index first with: python langgraph_ingest.py
    rag = LangGraphRAG()
    require_approval = os.getenv("REQUIRE_HUMAN_APPROVAL", "false").lower() in {"1", "true", "yes"}
    print("LangGraph RAG ready. Type 'quit' or 'exit' to stop.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        # Approval is checkpointed; the second call resumes the same graph run.
        result = rag.ask(question, require_approval=require_approval)
        if result["awaiting_approval"]:
            print("Approval required before returning this answer.")
            decision = input("Approve? [y/N]: ").strip().lower() == "y"
            result = rag.ask(
                question,
                approval={"approved": decision},
                thread_id=result["thread_id"],
            )
        print(f"Assistant: {result['answer']}")
        print("Sources:")
        for document in result["sources"]:
            metadata = document.metadata
            location = metadata.get("source", "unknown document")
            if metadata.get("page"):
                location += f", page {metadata['page']}"
            print(f"- [{metadata['citation']}] {location}")
