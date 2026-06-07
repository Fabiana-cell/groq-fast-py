import os
from client import GroqFastClient

# Substitua pela sua chave real da Groq para testar na sua máquina se quiser!
API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY_HERE")

try:
    # Inicializa o nosso cliente tunado
    client = GroqFastClient(api_key=API_KEY)
    
    messages = [
        {"role": "user", "content": "Tell me a 2-sentence joke about Python programming."}
    ]
    
    # --- MODO 1: RESPOSTA COMPLETA ---
    print("--- Testing Standard Response ---")
    response = client.create_chat_completion(model="llama3-8b-8192", messages=messages)
    print("Response:", response.choices[0].message["content"])
    print("\n" + "="*40 + "\n")
    
    # --- MODO 2: STREAMING (Palavra por Palavra) ---
    print("--- Testing Streaming Response (Real-time) ---")
    print("Response: ", end="", flush=True)
    for chunk in client.create_chat_stream(model="llama3-8b-8192", messages=messages):
        print(chunk, end="", flush=True)
    print("\n")

except Exception as e:
    print(f"Execution failed: {e}")
