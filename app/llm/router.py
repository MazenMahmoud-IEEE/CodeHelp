# ============================================================
# 🔧 Router LLM — Intent Classifier (via OpenRouter)
# ============================================================
import os
import requests

def llm_router(
    prompt: str,
    model: str = "deepseek/deepseek-r1",
    max_tokens: int = 256,
    temperature: float = 0.2
) -> str:
    """
    Uses OpenRouter LLM to classify the user's intent into:
    - generate
    - explain
    - chat
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ Please set your OpenRouter API key as 'OPENROUTER_API_KEY' environment variable.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        import time
        time.sleep(2)  # wait 2 seconds between calls
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip().lower()
        return content
    except requests.exceptions.Timeout:
        print("⚠️ Router Timeout: The request took too long.")
    except requests.exceptions.RequestException as e:
        print("❌ Router Network/API error:", e)
    except (KeyError, IndexError):
        print("⚠️ Router Unexpected response format:", response.text)

    return "chat"  # fallback

def intent_router(user_task, llm_router):
    system_prompt = """You are a routing model for a Python assistant.
Your task is to classify the user's intent into one of the following categories:

1. generate → when the user is asking to WRITE, CREATE, or MODIFY Python code.
2. explain  → when the user is asking to DESCRIBE, ANALYZE, or EXPLAIN code behavior.
3. chat     → when the user is making general conversation, not code related.

Return ONLY one word: generate, explain, or chat.
"""

    full_prompt = f"{system_prompt}\n\nUser query: {user_task}\n\nIntent:"
    try:
        response = llm_router(full_prompt).strip().lower()
    except Exception as e:
        print("⚠️ Router error:", e)
        response = "chat"  # fallback

    if "generate" in response:
        return "generate"
    elif "explain" in response:
        return "explain"
    else:
        return "chat"
