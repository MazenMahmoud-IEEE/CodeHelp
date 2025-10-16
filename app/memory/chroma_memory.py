import os
from config.constants import CHROMA_EMBEDDINGS_DIR
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from app.retrieval.embeddings import reload_chroma_vectorstore
from config.constants import CHROMA_MEMORY_DIR

def init_chroma_memory(embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                       memory_dir: str = CHROMA_MEMORY_DIR,
                       k: int = 5):
    """
    Initialize persistent Chroma-based long-term memory for the assistant.
    Returns the LangChain memory object and the underlying Chroma vectorstore.
    """
    os.makedirs(memory_dir, exist_ok=True)

    # Reload embedding function (can reuse the same embeddings as main Chroma)
    _, _, embedding_function = reload_chroma_vectorstore(embed_model_name=embed_model_name)

    # Load or create Chroma memory vectorstore
    memory_vectorstore = Chroma(
        persist_directory=memory_dir,
        embedding_function=embedding_function
    )

    # Wrap in LangChain retriever memory
    memory = VectorStoreRetrieverMemory(
        retriever=memory_vectorstore.as_retriever(search_kwargs={"k": k})
    )

    print("✅ Conversational memory initialized and persistent on disk.")
    return memory, memory_vectorstore