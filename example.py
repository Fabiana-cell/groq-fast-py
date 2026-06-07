import os
from client import GroqFastClient

# O usuário vai colar a própria chave aqui ou o sistema vai puxar direto do PC dele
API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY_HERE")

try:
    # Inicializa o cliente tunado de alta performance
    client = GroqFastClient(api_key=API_KEY)
    
    messages = [
        {"role": "user", "content": "Tell me a 2-sentence joke about Python programming."}
    ]
    
    # --- MODO 1: RESPOSTA COMPLETA ---
    print("--- Testing Standard Response ---")
    response = client.create_chat_completion(model="llama-3.1-8b-instant", messages=messages)
    print("Response:", response.choices[0].message["content"])
    print("\n" + "="*40 + "\n")
    
    # --- MODO 2: STREAMING (Tempo Real) ---
    print("--- Testing Streaming Response (Real-time) ---")
    print("Response: ", end="", flush=True)
    for chunk in client.create_chat_stream(model="llama-3.1-8b-instant", messages=messages):
        print(chunk, end="", flush=True)
    print("\n")

except Exception as e:
    print(f"Execution failed: {e}")
