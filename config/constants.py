# config/constants.py
import os

# Base logs folder
LOGS_DIR = "logs"

# Subdirectories
GEN_DIR = os.path.join(LOGS_DIR, "generation")
EXP_DIR = os.path.join(LOGS_DIR, "explanation")
CHAT_DIR = os.path.join(LOGS_DIR, "chat")

# Ensure folders exist
for folder in [GEN_DIR, EXP_DIR, CHAT_DIR]:
    os.makedirs(folder, exist_ok=True)

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
COMBINED_RAG_CORPUS = f"{KNOWLEDGE_BASE_DIR}/combined_rag_corpus.csv"
GROUND_TRUTH_JSON = f"{KNOWLEDGE_BASE_DIR}/ground_truth_ids_for_task.json"
CHROMA_EMBEDDINGS_DIR = "data/chroma/chroma_embeddings"
CHROMA_MEMORY_DIR = "data/chroma/chroma_memory"