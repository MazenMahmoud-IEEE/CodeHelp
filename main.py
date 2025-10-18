from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
from langchain.schema import Document
import os
import time
import logging

# === Imports from your app ===
from app.utils.langgraph_setup import langgraph_agent
from app.memory.session_memory import session_memory
from app.memory.chroma_memory import init_chroma_memory, reload_chroma_vectorstore

# ==========================================================
# ⚙️ Setup FastAPI App
# ==========================================================
app = FastAPI(title="CodeHelp")

# Mount static folder for CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory API key store
user_api_key_store = {}

# Keep last 8 chat messages for UI
chat_history = []

# ==========================================================
# 📁 Logging Setup
# ==========================================================
LOGS_DIR = "logs"
GEN_DIR = os.path.join(LOGS_DIR, "generation")
EXP_DIR = os.path.join(LOGS_DIR, "explanation")
CHAT_DIR = os.path.join(LOGS_DIR, "chat")

# Ensure directories exist
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(CHAT_DIR, exist_ok=True)

# ==========================================================
# 🏠 Home Page
# ==========================================================
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat_history": chat_history, "api_key_set": "key" in user_api_key_store}
    )

# ==========================================================
# 🔑 Save API Key
# ==========================================================
@app.post("/set_api_key")
async def set_api_key(api_key: str = Form(...)):
    user_api_key_store["key"] = api_key
    os.environ["OPENROUTER_API_KEY"] = api_key
    return {"status": "ok"}

# ==========================================================
# 💬 Chat Endpoint
# ==========================================================
@app.post("/chat", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    global chat_history

    if "key" not in user_api_key_store:
        return HTMLResponse("❌ Please set your API key first!")

    user_task = user_input.strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ✅ Trim memory (keep short context to avoid repeated responses)
    try:
        if len(session_memory.chat_memory.messages) > 4:
            session_memory.chat_memory.messages = session_memory.chat_memory.messages[-4:]
    except Exception as e:
        print(f"⚠️ Could not trim session memory: {e}")

    # Small polite delay
    for _ in range(3):
        time.sleep(0.2)

    # ======================================================
    # 🔮 Run LangGraph agent
    # ======================================================
    try:
        result = langgraph_agent.invoke({"user_task": user_task})
        response = result.get("response", "").strip()
        intent = result.get("intent", "chat")

        if not response:
            print("⚠️ Empty response returned from LangGraph agent:", result)
            response = "⚠️ The model returned an empty response."

        # 🧹 Remove possible duplication if model repeats last answer
        if len(chat_history) > 0 and chat_history[-1]["bot"].strip() == response.strip():
            print("⚠️ Detected repeated response; ignoring duplicate context.")
            response = "⚠️ Please rephrase or ask a different question."
    except Exception as e:
        response = f"❌ LangGraph execution failed: {e}"
        intent = "chat"

    # ======================================================
    # 🧠 Save Context to Memory + Chroma
    # ======================================================
    try:
        session_memory.chat_memory.add_user_message(user_task)
        session_memory.chat_memory.add_ai_message(response)

        memory, memory_vectorstore = init_chroma_memory()
        memory.save_context({"input": user_task}, {"output": response})
    except Exception as e:
        print(f"⚠️ Memory persistence failed: {e}")

    try:
        chroma_collection = reload_chroma_vectorstore()
        doc = Document(page_content=response, metadata={"intent": intent, "query": user_task})
        chroma_collection.add_documents([doc])
    except Exception as e:
        print(f"⚠️ Failed to save to Chroma: {e}")

    # ======================================================
    # 🪵 Save to Logs Based on Intent
    # ======================================================
    try:
        if intent == "generate":
            log_dir = GEN_DIR
        elif intent == "explain":
            log_dir = EXP_DIR
        else:
            log_dir = CHAT_DIR

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{timestamp}.txt")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"User Query:\n{user_task}\n\nModel Response:\n{response}\n")

        print(f"🪵 Logged {intent} query → {log_path}")
    except Exception as e:
        print(f"⚠️ Failed to write log file: {e}")

    # ======================================================
    # 🧾 Update Chat History for Frontend
    # ======================================================
    chat_history.append({"user": user_task, "bot": response})
    chat_history = chat_history[-8:]  # keep last 8 messages

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat_history": chat_history, "api_key_set": True}
    )

# ==========================================================
# 🚀 Run Locally (for development)
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
