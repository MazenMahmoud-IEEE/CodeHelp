def get_generation_prompt(user_task, context_text, memory_context):
    """
    Build a clean generation prompt with optional memory context.
    """
    prompt = (
        "You are a professional Python coding assistant.\n"
        "Your task is to generate efficient, correct, and PEP8-compliant Python code.\n\n"
    )

    if context_text:
        prompt += f"### Context:\n{context_text}\n\n"

    if memory_context:
        # Extract only the last 10 lines from memory context
        history_lines = memory_context.strip().splitlines()[-10:]
        history_text = "\n".join(history_lines)
        prompt += f"### Relevant previous examples:\n{history_text}\n\n"

    prompt += f"### Task:\n{user_task}\n\n### Output:\n"
    return prompt

def get_explanation_prompt(user_task: str, context_text: str = "") -> str:
    """
    Builds a prompt for Python code explanation or teaching.
    Used when the intent is to 'explain' code or concepts.
    """
    return f"""
You are a friendly and expert Python instructor.
Your goal is to help the user understand code or programming concepts clearly.

# User Query:
{user_task}

# Helpful Context:
{context_text}

# Output Requirements:
- Explain step-by-step using simple, clear English.
- Include short code snippets *only if* they help illustrate the concept.
- Avoid overwhelming detail or repetition.
- Assume the user has intermediate programming knowledge.

### Explanation:
"""