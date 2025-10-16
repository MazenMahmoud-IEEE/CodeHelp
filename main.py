from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
import os
import time

from app.utils.langgraph_setup import langgraph_agent
from app.memory.session_memory import session_memory
from app.memory.chroma_memory import init_chroma_memory, reload_chroma_vectorstore

app = FastAPI(title="CodeHelp")

# Mount static folder for CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory API key store (replace with DB/session if needed)
user_api_key_store = {}

# Keep last 8 messages
chat_history = []

# --- Home page ---
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat_history": chat_history, "api_key_set": "key" in user_api_key_store}
    )

# --- Save user API key ---
@app.post("/set_api_key")
async def set_api_key(api_key: str = Form(...)):
    user_api_key_store["key"] = api_key
    os.environ["OPENROUTER_API_KEY"] = api_key
    return {"status": "ok"}

# --- Chat endpoint ---
@app.post("/chat", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    global chat_history

    if "key" not in user_api_key_store:
        return HTMLResponse("❌ Please set your API key first!")

    user_task = user_input.strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # LangGraph processing
    for _ in range(3):
        time.sleep(0.2)

    try:
        result = langgraph_agent.invoke({"user_task": user_task})
        response = result["response"]
        intent = result.get("intent", "chat")
    except Exception as e:
        response = f"❌ LangGraph execution failed: {e}"
        intent = "chat"

    # Save to memory
    try:
        session_memory.chat_memory.add_user_message(user_task)
        session_memory.chat_memory.add_ai_message(response)
        memory = init_chroma_memory()
        memory.save_context({"input": user_task}, {"output": response})
    except Exception as e:
        print(f"⚠️ Memory persistence failed: {e}")

    # Save to Chroma
    try:
        chroma_collection = reload_chroma_vectorstore()
        chroma_collection.add(
            documents=[response],
            metadatas=[{"intent": intent, "query": user_task}],
            ids=[f"{intent}_{timestamp}"]
        )
    except Exception as e:
        print(f"⚠️ Failed to save to Chroma: {e}")

    # Update chat history
    chat_history.append({"user": user_task, "bot": response})
    chat_history = chat_history[-8:]  # keep last 8 messages

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat_history": chat_history, "api_key_set": True}
    )

# --- Optional: run directly ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
