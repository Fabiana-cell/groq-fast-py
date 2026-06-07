import requests
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Generator

class ChatChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None

class GroqResponse(BaseModel):
    id: str
    model: str
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]] = Field(default_factory=dict)

class GroqFastClient:
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        if not api_key:
            raise ValueError("API Key cannot be empty. Please provide a valid Groq API Key.")
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_chat_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> GroqResponse:
        """Envia mensagens e retorna a resposta completa de uma vez (Validada)."""
        url = f"{self.base_url}/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
        
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return GroqResponse(**response.json())
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Groq API Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")

    def create_chat_stream(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> Generator[str, None, None]:
        """Envia mensagens e vai entregando a resposta palavra por palavra (Streaming)."""
        url = f"{self.base_url}/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True}
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        content = decoded_line[6:]
                        if content == "[DONE]":
                            break
                        # Aqui extraímos o texto da palavra que acabou de chegar
                        import json
                        chunk = json.loads(content)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
        except Exception as e:
            raise RuntimeError(f"Streaming failed: {e}")
