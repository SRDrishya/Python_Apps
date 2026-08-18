from chatbot import ChatBot
import time

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

results = chatbot.retrieve_memories(
    "What is my name?"
)

print("\nRetrieved memories:")

for memory in results:
    print(
        f"{memory['score']:.4f} -> "
        f"{memory['content']}"
    )
