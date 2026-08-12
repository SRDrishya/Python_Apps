import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4.1-mini")

messages = [
    "Hello, my name is Drishya.",
    "I want to learn Python.",
    "What is a list?"
]

total_tokens = 0

for message in messages:
    tokens = encoding.encode(message)

    print("Message:", message)
    print("Tokens:", tokens)
    print("Count:", len(tokens))
    print()

    total_tokens += len(tokens)

print("Estimated total:", total_tokens)