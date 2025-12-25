import os

from ollama import Client


class OllamaClient:
    def __init__(self, host: str = None):
        """
        Initialize the Ollama client.

        Args:
            host (str, optional): The URL of the Ollama server.
                                  Defaults to OLLAMA_HOST env var or http://localhost:11434.
        """
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = Client(host=self.host)

    def chat(
        self, model: str, messages: list, format: str = None, options: dict = None
    ):
        """
        Send a chat request to the Ollama model.
        """
        try:
            return self.client.chat(  # pyright: ignore[reportCallIssue]
                model=model, messages=messages, format=format, options=options
            )
        except Exception as e:
            print(f"Error communicating with Ollama at {self.host}: {e}")
            raise

    def pull_model(self, model_name: str):
        """
        Pull a model if it doesn't exist.
        """
        try:
            # Check if model exists (list local models)
            # This is a bit of a heuristic, pull is idempotent anyway but can be slow
            # self.client.pull(model_name)
            # For now, we assume the user might want explicitly pull or we just let it fail/auto-pull if configured
            print(f"Requesting pull for model: {model_name}...")
            self.client.pull(model_name)
            print(f"Model {model_name} pulled successfully.")
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}")
