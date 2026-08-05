from openai import OpenAI
from config import API_KEY

class ChatBot:
    def __init__(self, model="gpt-4.1-mini", temperature=0.7, system_prompt="You are a helpful AI assistant."):
        self.client = OpenAI(api_key=API_KEY)
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.history = []

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

    def chat(self, message):
        """Send a prompt to the OpenAI API and return the text response."""
        self.add_user_message(message)
        request_input = self.build_conversation()

        response = self.client.responses.create(
            model=self.model,
            input=request_input,
            temperature=self.temperature,
        )

        output_text = ""
        if hasattr(response, "output_text") and response.output_text:
            output_text = response.output_text.strip()
        else:
            for item in getattr(response, "output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        output_text += content.get("text", "")
            output_text = output_text.strip()

        self.add_assistant_message(output_text)
        return output_text
