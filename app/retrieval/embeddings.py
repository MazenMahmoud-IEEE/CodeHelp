# ============================================================
# 🔹 Settings & Paths
# ============================================================
import os
from config.constants import CHROMA_EMBEDDINGS_DIR
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document

os.makedirs(CHROMA_EMBEDDINGS_DIR, exist_ok=True)

# ============================================================
# 🔹 Load embedding model
# ============================================================
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print("🔹 Loading embedding model...")
    embed_model = SentenceTransformer(model_name)
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    return embed_model, embedding_function

# ============================================================
# 🔹 Convert DataFrame corpus to LangChain Documents
# ============================================================
def corpus_to_documents(df_corpus):
    print("🔹 Preparing documents for Chroma...")
    documents = [
        Document(
            page_content=row["prompt"],
            metadata={
                "source": row.get("source", "unknown"),
                "canonical_solution": row["canonical_solution"],
                "task_id": str(row.get("task_id", idx))
            }
        )
        for idx, row in df_corpus.iterrows()
    ]
    return documents

# ============================================================
# 🔹 Build or Load Persistent Chroma Vector Store
# ============================================================
def get_chroma_vectorstore(documents, persist_directory=CHROMA_EMBEDDINGS_DIR, embedding_function=None):
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        print("✅ Loading existing Chroma index from disk...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    else:
        print("⚙️ Building new Chroma index (this may take a while)...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        print("💾 Embeddings saved to:", persist_directory)
    print("✅ Chroma vector store ready with", len(documents), "items.")
    return vectorstore

# ============================================================
# 🔹 Example Retrieval
# ============================================================
def demo_retrieval(vectorstore, query="Write a function that checks if two strings are anagrams.", k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)

    print("\n🔍 Query:", query)
    print("\nTop retrieved examples:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Rank {i} ---")
        print("Source:", doc.metadata.get("source", "unknown"))
        print("Task ID:", doc.metadata.get("task_id", "N/A"))
        print("Prompt (first 300 chars):", doc.page_content[:300])
        print("---- canonical_solution (first 200 chars) ----")
        print(doc.metadata["canonical_solution"][:200])

def reload_chroma_vectorstore(embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                              persist_directory: str = CHROMA_EMBEDDINGS_DIR):
    """
    Reload an existing Chroma vector store and embedding model without recalculating embeddings.
    """
    print("🔹 Reloading embedding model and Chroma store...")

    # Recreate embedding function
    embed_model = SentenceTransformer(embed_model_name)
    embedding_function = SentenceTransformerEmbeddings(model_name=embed_model_name)

    # Reload Chroma collection from disk
    chroma_collection = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    print("✅ Loaded Chroma collection from disk:", persist_directory)
    print("✅ Embedding model ready:", embed_model_name)
    return chroma_collection, embed_model, embedding_function
