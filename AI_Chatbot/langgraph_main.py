"""Command-line entry point for the LangGraph RAG chatbot."""

from langgraph_rag import LangGraphRAG


if __name__ == "__main__":
    # Build the index first with: python langgraph_ingest.py
    rag = LangGraphRAG()
    print("LangGraph RAG ready. Type 'quit' or 'exit' to stop.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        # The graph runs retrieval and grounded generation in sequence.
        result = rag.ask(question)
        print(f"Assistant: {result['answer']}")
        print("Sources:")
        for document in result["sources"]:
            metadata = document.metadata
            location = metadata.get("source", "unknown document")
            if metadata.get("page"):
                location += f", page {metadata['page']}"
            print(f"- [{metadata['citation']}] {location}")
