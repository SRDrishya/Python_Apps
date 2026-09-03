"""Command-line entry point for the LangChain RAG chatbot."""

from langchain_rag import LangChainRAG


if __name__ == "__main__":
    # The index must be built first with: python langchain_ingest.py
    rag = LangChainRAG()
    print("LangChain RAG ready. Type 'quit' or 'exit' to stop.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        # ask() performs retrieval, grounding, and generation as one workflow.
        result = rag.ask(question)
        print(f"Assistant: {result['answer']}")
        print("Sources:")
        for document in result["sources"]:
            metadata = document.metadata
            location = metadata.get("source", "unknown document")
            if metadata.get("page"):
                location += f", page {metadata['page']}"
            # Show the same citation label that appears in the answer.
            print(f"- [{metadata['citation']}] {location}")
