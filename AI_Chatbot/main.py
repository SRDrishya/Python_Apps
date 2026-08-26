from chatbot import ChatBot
import time
import logging


logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

chatbot = ChatBot()

if __name__ == "__main__":
    print("AI Chatbot ready. Type 'quit' or 'exit' to stop.")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        print("Assistant:", end=" ", flush=True)
        for chunk in chatbot.chat(user_input, stream=True):
            print(chunk, end="", flush=True)
            time.sleep(0.05)  # Simulate streaming delay
        print()

        # Show the document metadata used to ground the answer so the user
        # can verify which file and page supplied the retrieved context.
        print("Sources:")
        displayed_sources = set()
        for source in chatbot.last_sources:
            metadata = source.get("metadata", {})
            source_name = metadata.get("source", "unknown document")
            page = metadata.get("page")
            source_label = f"{source_name}, page {page}" if page else source_name

            if source_label in displayed_sources:
                continue

            displayed_sources.add(source_label)
            citation = source.get("citation", "")
            print(
                f"- [{citation}] {source_label} "
                f"(score: {source['score']:.4f})"
            )


