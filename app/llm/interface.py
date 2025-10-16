import os
import requests
from app.llm.router import llm_router, intent_router

# ============================================================
# 🔧 Unified LLM Code Generation + Explanation (via OpenRouter)
# ============================================================

def query_openrouter_llm(
    user_task: str,
    retrieved_docs=None,
    model: str = "deepseek/deepseek-r1:free",
    max_tokens: int = 512,
    temperature: float = 0.2
) -> str:
    """
    Queries OpenRouter LLM (e.g., DeepSeek-R1) for code generation or explanation
    based on the detected intent and optional retrieved documents.
    """

    # ✅ 1. Define system prompts
    system_prompt_generation = """You are a professional Python coding assistant specializing in code generation, debugging, and algorithmic problem solving.

Your responses must:
- Be professional, concise, and logically structured.
- Follow clean coding practices (PEP8 compliant).
- Include exactly ONE full function implementation unless otherwise stated.
- Avoid explanations, markdown syntax, or conversational filler.
- Never include system messages, RAG context, or examples in the final code.
- Assume the user has intermediate coding knowledge.

You have expertise in:
- Python algorithms and data structures.
- Mathematics, recursion, and optimization.
- Writing robust, correct, and efficient Python code.
"""

    system_prompt_explanation = """You are a helpful AI coding tutor.
Your goal is to explain code, algorithms, and debugging steps in a clear, concise, and educational manner.

Guidelines:
- Give step-by-step reasoning.
- Be concise but informative.
- Use plain English (no over-technical jargon unless necessary).
- Do not generate new code unless explicitly asked.
"""

    system_prompt_chat = """You are a friendly and knowledgeable AI assistant."""
    
    # ✅ 2. Choose system prompt
    intent = intent_router(user_task, llm_router)
    if intent == "explain":
        system_prompt = system_prompt_explanation
    elif intent == "generate":
        system_prompt = system_prompt_generation
    else:
        system_prompt = system_prompt_chat

    # ✅ 3. Construct context-aware prompt
    if retrieved_docs and len(retrieved_docs) > 0:
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""Context (retrieved knowledge):
{context_text}

User Task:
{user_task}
"""
    else:
        prompt = user_task

    # ✅ 4. Prepare API request
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # ✅ 5. Send request and handle response
    try:
        import time
        time.sleep(2)  # wait 2 seconds between calls
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content

    except requests.exceptions.Timeout:
        print("⚠️ Timeout: The request took too long. Try reducing 'max_tokens'.")
    except requests.exceptions.RequestException as e:
        print("❌ Network or API error:", e)
    except (KeyError, IndexError):
        print("⚠️ Unexpected response format:", response.text)

    return ""