"""
Example Python client for Qwen3-0.6B Local API.

Make sure the API server is running: python src/api.py
"""

import requests

API_URL = "http://127.0.0.1:8000"


def chat(message: str, conversation_id: int = None, temperature: float = 0.7):
    """
    Send a message to the chat API.

    Args:
        message: Your message
        conversation_id: Optional conversation ID (creates new if None)
        temperature: Sampling temperature

    Returns:
        Tuple of (conversation_id, response_text)
    """
    payload = {
        "message": message,
        "temperature": temperature,
        "max_tokens": 512,
    }

    if conversation_id is not None:
        payload["conversation_id"] = conversation_id

    response = requests.post(f"{API_URL}/chat", json=payload)
    response.raise_for_status()

    data = response.json()
    return data["conversation_id"], data["answer"]


def chat_stream(message: str, conversation_id: int = None, temperature: float = 0.7):
    """Streaming SSE : affiche les chunks au fur et à mesure. curl équivalent : curl -N POST /chat/stream."""
    import json
    payload = {"message": message, "temperature": temperature, "max_tokens": 512}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    full = ""
    cid = conversation_id
    with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if "conversation_id" in obj:
                cid = obj["conversation_id"]
            if "chunk" in obj:
                print(obj["chunk"], end="", flush=True)
                full += obj["chunk"]
    print()
    return cid, full


def list_conversations():
    """List all conversations."""
    response = requests.get(f"{API_URL}/conversations")
    response.raise_for_status()
    return response.json()


def delete_conversation(conversation_id: int):
    """Delete a conversation."""
    response = requests.delete(f"{API_URL}/conversation/{conversation_id}")
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("Qwen3-0.6B Chat Client")
    print("=" * 40)

    cid = None

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        cid, response = chat(user_input, conversation_id=cid)
        print(f"\nAssistant: {response}")
        print(f"(Conversation ID: {cid})")
