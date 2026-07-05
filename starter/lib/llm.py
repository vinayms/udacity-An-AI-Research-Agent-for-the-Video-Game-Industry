# starter/lib/llm.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment or .env file.")
        
        # Configure client with the Vocareum API endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openai.vocareum.com/v1/"
        )

    def get_completion(self, messages, model="gpt-4o-mini", temperature=0.0, max_tokens=1000):
        """
        Sends a list of message objects to the OpenAI Chat Completions API.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def get_embedding(self, text, model="text-embedding-3-small"):
        """
        Generates an embedding vector for the provided text.
        """
        response = self.client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
