from openai import OpenAI
from config import API_KEY
import json
from pathlib import Path

class ChatBot:
    def __init__(self, model="gpt-4.1-mini", temperature=0.7, system_prompt="You are a helpful AI assistant."):
        self.client = OpenAI(api_key=API_KEY)
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
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
        if self.system_prompt:
            conversation.append({
                "role": "system",
                "content": self.system_prompt,
            })
        conversation.extend(self.history)
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
        print("DEBUG HISTORY:")
        print(self.history)
        stream = self.client.responses.create(
        model=self.model,
        input=request_input,
        temperature=self.temperature,
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
    

