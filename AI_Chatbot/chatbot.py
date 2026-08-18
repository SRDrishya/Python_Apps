from email.mime import message, text

from openai import OpenAI
from config import API_KEY
import json
from pathlib import Path
import tiktoken

class ChatBot:
    def __init__(self, model="gpt-4.1-mini", temperature=0.7, system_prompt="You are a helpful AI assistant.",context_budget=4000,
    max_output_tokens=500):
        self.client = OpenAI(api_key=API_KEY)
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.context_budget = context_budget
        self.max_output_tokens = max_output_tokens
        self.history = []
        self.chathistory = Path("chathistory.json")
        self.memory_file = Path("memory.json")
        self.load_history()

    def add_user_message(self, message):
        self.history.append({
            "role": "user",
            "content": message,
        })

    def add_assistant_message(self, message):
        self.history.append({
            "role": "assistant",
            "content": message,
        })

    def build_conversation(self, memories=None):
        conversation = []
    
        system_tokens = 0
    
        if self.system_prompt:
            conversation.append({
                "role": "system",
                "content": self.system_prompt,
            })
    
            system_tokens = self.count_tokens(self.system_prompt)
        if memories is None:
            memories = []
        memory_text = ""
        memory_tokens = 0
        if memories:
            memory_text = "Relevant long-term memories:\n"

            for memory in memories:
                memory_text += f"- {memory['content']}\n"

            memory_tokens = self.count_memory_tokens(memories)
        previous_history = self.history[:-1]
        current_message = self.history[-1]
    
        current_message_tokens = self.count_tokens(
            current_message["content"]
        )
    
        history_budget = (
            self.context_budget
            - system_tokens
            - memory_tokens
            - current_message_tokens
            - self.max_output_tokens
        )
    
        recent_history, history_tokens = self.get_recent_history(
            previous_history,
            history_budget
        )

        # Add memory context before the current message
        if memory_text:
            conversation.append({
                "role": "system",
                "content": memory_text,
            })

        conversation.append(current_message)
    
        print("Context budget:", self.context_budget)
        print("System tokens:", system_tokens)
        print("Current message tokens:", current_message_tokens)
        print("Max output tokens:", self.max_output_tokens)
        print("History budget:", history_budget)
        print("Selected history tokens:", history_tokens)
    
        return conversation

    def clear_history(self):
        self.history = []

    def chat(self, message, stream=False):
        """Send prompt and either stream chunks or return full response.

        - If `stream=True` returns the generator from `stream_response()`.
        - Otherwise collects the generator's chunks and returns the final string.
        """
        if stream:
            return self.stream_response(message)

        # Non-stream: collect chunks produced by the canonical streamer
        chunks = []
        for chunk in self.stream_response(message):
            chunks.append(chunk)

        return "".join(chunks).strip()

    def stream_response(self, message):
        self.add_user_message(message)

         # Retrieve relevant long-term memories
        memories = self.retrieve_memories(message, top_k=3)

        request_input = self.build_conversation(memories=memories)

        stream = self.client.responses.create(
        model=self.model,
        input=request_input,
        temperature=self.temperature,
        max_output_tokens=self.max_output_tokens,
        stream=True,
        )

        full_text = ""

        for event in stream:
            if event.type == "response.output_text.delta":
                full_text += event.delta
                yield event.delta

        self.add_assistant_message(full_text.strip())

        memory = self.extract_memory(message)

        if memory:
            self.save_memory(memory)

        self.save_history()


    def load_history(self):
        """Load a list of messages into the chatbot's history."""
        
        if not self.chathistory.exists():
            self.history = []
            return
        with open(self.chathistory, "r") as f:
            self.history = json.load(f)

    def save_history(self):
        """Return the current conversation history."""
        with open(self.chathistory, "w") as f:
            json.dump(self.history, f)
    

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def get_recent_history(self, history, max_tokens):
        selected_history = []
        token_count = 0

        for message in reversed(history):
            message_tokens = self.count_tokens(message["content"])

            if token_count + message_tokens > max_tokens:
                break

            selected_history.insert(0, message)
            token_count += message_tokens

        return selected_history, token_count

    def save_memory(self, message):
        """Save a message as long-term memory if it doesn't already exist."""

        memories = []

        if self.memory_file.exists():
            with open(self.memory_file, "r") as f:
                memories = json.load(f)

        # Prevent exact duplicates
        for memory in memories:
            if memory["message"].strip().lower() == message.strip().lower():
                return

        # Create embedding
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=message,
        )

        embedding = response.data[0].embedding

        memory = {
            "message": message,
            "embedding": embedding,
        }

        memories.append(memory)

        with open(self.memory_file, "w") as f:
            json.dump(memories, f)

    def cosine_similarity(self, vector_a, vector_b):
        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = sum(
            a * a for a in vector_a
        ) ** 0.5

        magnitude_b = sum(
            b * b for b in vector_b
        ) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)
    
    def retrieve_memories(self, query, top_k=3):
        """Return the most relevant stored memories for a query."""

        # 1. Create an embedding for the current query
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )

        query_vector = response.data[0].embedding

        # 2. Load stored memories
        if not self.memory_file.exists():
            return []

        with open(self.memory_file, "r") as f:
            memories = json.load(f)

        # 3. Calculate similarity for every memory
        results = []

        for memory in memories:
            score = self.cosine_similarity(
                query_vector,
                memory["embedding"],
            )

            results.append({
                "content": memory["message"],
                "score": score,
            })

        # 4. Highest similarity first
        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        # 5. Return only the top results
        return results[:top_k]
    
    def count_memory_tokens(self, memories):
        total = 0

        for memory in memories:
            total += self.count_tokens(memory["content"])

        return total

    def should_save_memory(self, message):
        """Determine whether a user message looks like useful long-term memory."""

        keywords = [
            "my name is",
            "i am",
            "i live in",
            "i work as",
            "i work at",
            "i like",
            "i love",
            "i prefer",
            "i want to",
            "my goal is",
            "my age is"
        ]

        message_lower = message.lower()

        for keyword in keywords:
            if keyword in message_lower:
                return True

        return False

    def extract_memory(self, message):
        """Ask the model whether the user's message contains a useful long-term memory."""

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a memory extraction system for an AI chatbot. "
                        "Identify durable information about the user that may be "
                        "useful in future conversations.\n\n"
                        "Return ONLY the memory itself if there is one. "
                        "If there is no useful long-term memory, return: NONE\n\n"
                        "Examples:\n"
                        "User: My name is Drishya.\n"
                        "Output: My name is Drishya.\n\n"
                        "User: I am 37 years old.\n"
                        "Output: Drishya is 37 years old.\n\n"
                        "User: I want to become an AI engineer.\n"
                        "Output: Drishya wants to become an AI engineer.\n\n"
                        "User: What is Python?\n"
                        "Output: NONE"
                    ),
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            temperature=0,
            max_output_tokens=100,
        )

        memory = response.output_text.strip()

        if memory.upper() == "NONE":
            return None

        return memory