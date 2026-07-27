from openai import OpenAI
from config import API_KEY

class ChatBot:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)

    def chat(self, message, model="gpt-4.1-mini", temperature=0.7):
        """Send a prompt to the OpenAI API and return the text response."""
        response = self.client.responses.create(
            model=model,
            input=message,
            temperature=temperature,
        )

        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        output_text = ""
        for item in getattr(response, "output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    output_text += content.get("text", "")

        return output_text.strip()
