import json

from .client import OllamaClient


class StructureEngine:
    def __init__(self, client: OllamaClient, model_name: str = "llama3.1"):
        self.client = client
        self.model_name = model_name

    def extract(self, text_content: str, spec: dict) -> dict:
        """
        Extract structured data from text based on the provided spec.
        """
        schema_str = json.dumps(spec, indent=2, ensure_ascii=False)

        prompt = f"""
You are a data extraction assistant. 
Extract information from the provided text content based on the following JSON schema specification.
The values in the schema describe what to extract.

SCHEMA:
{schema_str}

TEXT CONTENT:
{text_content}

INSTRUCTIONS:
1. Return purely valid JSON matching the keys in the schema.
2. If a field is not found, set it to null or an empty string.
3. Do not include markdown formatting (like ```json), just the raw JSON string.
"""

        messages = [{"role": "user", "content": prompt}]

        # Enforce JSON mode if supported by the model/Ollama,
        # but standard chat with strong prompting usually works for Llama 3.1
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            format="json",  # Enable Ollama's JSON mode
            options={"temperature": 0},
        )

        result_content = response["message"]["content"]

        try:
            return json.loads(result_content)
        except json.JSONDecodeError:
            # Fallback cleanup if model returns markdown ticks despite instructions
            cleaned = result_content.strip().strip("`").replace("json\n", "", 1)
            return json.loads(cleaned)
