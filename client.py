import requests
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# 1. Modelo de dados para garantir que a resposta da IA venha formatada perfeitamente
class ChatChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None

class GroqResponse(BaseModel):
    id: str
    model: str
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]] = Field(default_factory=dict)

# 2. O Cliente principal da nossa SDK
class GroqFastClient:
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        """
        Inicializa o cliente para conexão ultra-rápica com a API da Groq.
        """
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_chat_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> GroqResponse:
        """
        Envia um histórico de mensagens para o modelo de IA da Groq e retorna a resposta validada.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status() # Dispara um erro automático se a API da Groq falhar
        
        # O Pydantic valida o JSON da Groq e o transforma em um objeto Python limpo
        return GroqResponse(**response.json())
