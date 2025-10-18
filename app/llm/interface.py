import os
import time
import requests
from app.llm.router import llm_router, intent_router

def query_openrouter_llm(
    user_task: str,
    retrieved_docs=None,
    model: str = "deepseek/deepseek-r1",
    max_tokens: int = 512,
    temperature: float = 0.2
) -> str:
    """
    Queries OpenRouter LLM for code generation or explanation.
    Automatically handles DeepSeek models that return 'reasoning' instead of 'content'.
    """

    # System prompts
    system_prompt_generation = """You are a professional Python coding assistant specializing in code generation, debugging, and algorithmic problem solving.

Your responses must:
- Be professional, concise, and logically structured.
- Follow clean coding practices (PEP8 compliant).
- Include exactly ONE full function implementation unless otherwise stated.
- Avoid explanations, markdown syntax, or conversational filler.
- Never include system messages, RAG context, or examples in the final code.
- Assume the user has intermediate coding knowledge.
"""

    system_prompt_explanation = """You are a helpful AI coding tutor.
Explain code, algorithms, and debugging steps clearly and concisely.
Do not include your reasoning process or chain-of-thought unless it's part of the final answer.
"""

    system_prompt_chat = "You are a friendly and knowledgeable AI assistant."

    # Intent routing
    intent = intent_router(user_task, llm_router)
    system_prompt = {
        "explain": system_prompt_explanation,
        "generate": system_prompt_generation
    }.get(intent, system_prompt_chat)

    print(f"🧠 Intent detected: {intent}")

    # Prompt construction
    if retrieved_docs:
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Context:\n{context_text}\n\nUser Task:\n{user_task}"
    else:
        prompt = user_task

    # API setup
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ Please set OPENROUTER_API_KEY in your environment.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # API call
    try:
        time.sleep(2)
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Handle errors
        if "error" in data:
            msg = data["error"].get("message", "Unknown error")
            print("❌ OpenRouter error:", msg)
            return f"⚠️ API error: {msg}"

        if not data.get("choices"):
            print("⚠️ Empty choices:", data)
            return "⚠️ No valid response from model."

        message = data["choices"][0].get("message", {})
        content = message.get("content", "")
        reasoning = message.get("reasoning", "")

        # ✅ Smart fallback: use reasoning only if content is missing
        if not content.strip() and reasoning.strip():
            print("ℹ️ Using reasoning as fallback (content empty).")
            content = reasoning.strip()

        if not content.strip():
            print("⚠️ Model returned no usable output:", data)
            return "⚠️ Model returned no usable output."

        # Optional: Trim reasoning-like internal chatter if too verbose
        if "Okay," in content and "Let's" in content[:100]:
            # Try to extract concise final paragraph
            parts = content.split("\n\n")
            if len(parts) > 1:
                content = parts[-1].strip()

        print(f"✅ LLM returned {len(content)} characters for intent '{intent}'.")
        print(f"🪶 Sample output:\n{content[:200]}...\n")
        return content

    except requests.exceptions.Timeout:
        return "⚠️ Timeout: The request took too long."

    except requests.exceptions.RequestException as e:
        print("❌ Network or API error:", e)
        return f"⚠️ Network or API error: {e}"

    except Exception as e:
        print("❌ Unexpected error:", e)
        return f"⚠️ Unexpected error: {e}"
