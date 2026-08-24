from rag import ask


question = "Who created the Aurora project?"


result = ask(question)


print("\nQuestion:")
print(question)


print("\nAnswer:")
print(result["answer"])


print("\nRetrieved chunks:")

for item in result["chunks"]:

    print(
        f"\nScore: {item['score']:.4f}"
    )

    print(
        item["chunk"]
    )