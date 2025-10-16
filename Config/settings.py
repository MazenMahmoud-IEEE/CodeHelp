# config/settings.py
import os

# Read OpenRouter API key from environment variable or fallback to None
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
