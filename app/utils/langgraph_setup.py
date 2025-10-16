from langgraph.graph import StateGraph, END, START
from app.llm.router import llm_router
from app.llm.interface import query_openrouter_llm
from app.retrieval.retriever import retrieve_context_from_chroma
from app.prompts.prompts import get_generation_prompt, get_explanation_prompt
from app.retrieval.embeddings import reload_chroma_vectorstore
from app.llm.router import intent_router

chroma_collection = reload_chroma_vectorstore()

# 🧩 1. Define state structure
class AgentState(dict):
    user_task: str = ""
    intent: str = ""
    response: str = ""


# 🧠 2. Define the node functions
def node_generate(state: AgentState):
    user_task = state["user_task"]
    context_text = retrieve_context_from_chroma(user_task, chroma_collection, k=8)
    final_prompt = get_generation_prompt(user_task, context_text, "")
    response = query_openrouter_llm(final_prompt)
    state["response"] = response
    print("🧠 Generated code:\n", response)
    return state

def node_explain(state: AgentState):
    user_task = state["user_task"]
    context_text = retrieve_context_from_chroma(user_task, chroma_collection, k=8)
    final_prompt = get_explanation_prompt(user_task, context_text)
    response = query_openrouter_llm(final_prompt)
    state["response"] = response
    print("📘 Explanation:\n", response)
    return state

def node_chat(state: AgentState):
    user_task = state["user_task"]
    response = query_openrouter_llm(user_task)
    state["response"] = response
    print("💬 Chat reply:\n", response)
    return state

# 🧭 3. Define router
def router_node(state: AgentState):
    user_task = state["user_task"]
    intent = intent_router(user_task, llm_router)
    state["intent"] = intent
    print(f"⚙️ Intent detected → {intent.upper()}")
    return state

# 🕸️ 4. Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("generate", node_generate)
graph.add_node("explain", node_explain)
graph.add_node("chat", node_chat)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    lambda state: state["intent"],
    {
        "generate": "generate",
        "explain": "explain",
        "chat": "chat"
    }
)
graph.add_edge("generate", END)
graph.add_edge("explain", END)
graph.add_edge("chat", END)

langgraph_agent = graph.compile()

# 🚀 5. Define a simple runner
def run_langgraph_agent():
    print("🤖 LangGraph Agent — type 'exit' to quit\n")
    while True:
        user_task = input("🧑 You: ").strip()
        if user_task.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        state = {"user_task": user_task}
        result = langgraph_agent.invoke(AgentState({"user_task": user_task}))
        print("\n🤖 Assistant:", result["response"])
        print("\n" + "-" * 60 + "\n")