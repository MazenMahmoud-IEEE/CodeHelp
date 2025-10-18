def retrieve_context_from_chroma(query: str, chroma_collection, k: int = 8) -> str:
    """
    Retrieve relevant examples from Chroma and format them into a context string.
    Returns formatted text ready to be inserted into a prompt.
    """
    try:
        # ✅ Safety check in case something unexpected is returned
        if isinstance(chroma_collection, tuple):
            chroma_collection = chroma_collection[0]

        results = chroma_collection.similarity_search_with_score(query, k=k)

    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        return ""

    context_pieces = []
    for doc, score in results:
        prompt_text = doc.page_content
        sol_text = doc.metadata.get("canonical_solution", "")
        source = doc.metadata.get("source", "unknown")
        snippet = "\n".join(sol_text.split("\n")[:6])  # limit to 6 lines
        context_pieces.append(
            f"# From {source.upper()} dataset\n"
            f"Example task:\n{prompt_text.strip()}\n\nExample solution:\n{snippet.strip()}\n"
        )

    return "\n\n".join(context_pieces).strip()
