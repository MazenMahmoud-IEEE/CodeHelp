import json
import random
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.memory.chroma_memory import init_chroma_memory

# Retrieve all memory records
memory, memory_vectorstore = init_chroma_memory()
retriever = memory.retriever
results = retriever.get_relevant_documents("factorial")  # Try with any keyword

print(f"🔍 Found {len(results)} related memory items:\n")
for i, doc in enumerate(results, 1):
    print(f"--- Memory #{i} ---")
    print("User task:", doc.page_content[:200])
    print("Metadata:", doc.metadata)
    print("----------\n")

class RAGEvaluator:
    def __init__(self, corpus_csv, ground_truth_json, chroma_dir="chroma_embeddings", embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Load corpus
        self.df_corpus = pd.read_csv(corpus_csv)
        if "id" in self.df_corpus.columns and "task_id" not in self.df_corpus.columns:
            self.df_corpus.rename(columns={"id": "task_id"}, inplace=True)

        # Load ground truth mapping
        with open(ground_truth_json, "r") as f:
            self.ground_truth = json.load(f)

        # Initialize Chroma
        embedding_function = SentenceTransformerEmbeddings(model_name=embed_model)
        self.chroma_collection = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embedding_function
        )

        # Prepare evaluation tasks
        mbpp = self.df_corpus[self.df_corpus["source"] == "mbpp"].to_dict(orient="records")
        humaneval = self.df_corpus[self.df_corpus["source"] == "humaneval"].to_dict(orient="records")
        codeparrot = self.df_corpus[self.df_corpus["source"] == "codeparrot"].to_dict(orient="records")
        random.seed(42)
        sampled_cp = random.sample(codeparrot, min(1000, len(codeparrot)))
        self.eval_tasks = mbpp + humaneval + sampled_cp

    @staticmethod
    def precision_at_k(retrieved_ids, relevant_ids, k=8):
        if not retrieved_ids or not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        return len(set(top_k) & set(relevant_ids)) / k

    def evaluate_task(self, user_task, user_task_solution, k=1):
        try:
            results = self.chroma_collection.similarity_search_with_score(user_task, k=k)
        except Exception as e:
            print("⚠️ Retrieval failed:", e)
            return 0.0, [], []

        retrieved_ids = [doc.metadata.get("task_id") for doc, _ in results]
        relevant_ids = self.ground_truth.get(user_task_solution.strip(), [])
        prec = self.precision_at_k(retrieved_ids, relevant_ids, k)
        return prec, retrieved_ids, relevant_ids

    def run_evaluation(self, k=5, report_file="rag_evaluation_report.csv"):
        report_rows = []

        for task in tqdm(self.eval_tasks, desc="Evaluating RAG tasks"):
            task_id = task.get("id") or f"{task['source']}_{hash(task['prompt']) % 10**6}"
            prec, retrieved_ids, relevant_ids = self.evaluate_task(
                task["prompt"], task["canonical_solution"], k=k
            )

            # Normalize retrieved IDs
            source_prefix = task["source"]
            normalized_ids = [
                rid if rid.startswith(source_prefix) else f"{source_prefix}_{rid}"
                for rid in retrieved_ids if rid is not None
            ]
            corrected_prec = self.precision_at_k(normalized_ids, relevant_ids, k=1)

            report_rows.append({
                "task_id": task_id,
                "source": task["source"],
                "precision_at_1": corrected_prec,
                "retrieved_ids": normalized_ids,
                "relevant_ids": relevant_ids
            })

        # Save report
        df_report = pd.DataFrame(report_rows)
        df_report.to_csv(report_file, index=False)
        avg_precision = df_report["precision_at_1"].mean()
        print(f"✅ Saved RAG evaluation report: {report_file}")
        print(f"📊 Average Precision@1 over {len(df_report)} tasks: {avg_precision:.3f}")
        print("\n🔍 Sample of first 5 evaluated tasks:")
        print(df_report.head(5).to_string(index=False))
        return df_report
