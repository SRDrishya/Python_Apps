from email.mime import text

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

    def build_conversation(self):
        conversation = []
    
        system_tokens = 0
    
        if self.system_prompt:
            conversation.append({
                "role": "system",
                "content": self.system_prompt,
            })
    
            system_tokens = self.count_tokens(self.system_prompt)
    
        previous_history = self.history[:-1]
        current_message = self.history[-1]
    
        current_message_tokens = self.count_tokens(
            current_message["content"]
        )
    
        history_budget = (
            self.context_budget
            - system_tokens
            - current_message_tokens
            - self.max_output_tokens
        )
    
        recent_history, history_tokens = self.get_recent_history(
            previous_history,
            history_budget
        )
    
        conversation.extend(recent_history)
    
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
        request_input = self.build_conversation()
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