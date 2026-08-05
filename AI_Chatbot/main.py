from chatbot import ChatBot

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

        response = chatbot.chat(user_input)
        print("Assistant:", response)




