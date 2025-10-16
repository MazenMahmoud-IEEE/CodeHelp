from datasets import load_dataset
import itertools
import os
import pandas as pd
from collections import defaultdict
from config.constants import COMBINED_RAG_CORPUS, GROUND_TRUTH_JSON
import json

# --- Load MBPP (train) + HumanEval (test) ---
mbpp = load_dataset("mbpp", split="train")       # ~400 examples
humaneval = load_dataset("openai/openai_humaneval", split="test")  # 164 examples

# --- Load CodeParrot in streaming mode ---
try:
    codeparrot_stream = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    # Take only first 5000 examples
    codeparrot = list(itertools.islice(codeparrot_stream, 10000))
except Exception as e:
    print("Failed to load CodeParrot:", e)
    codeparrot = []

# --- Combine datasets ---
rows = []

def add_examples(ds, source, prompt_field, sol_field, start_idx=0, max_examples=None):
    n_added = 0
    for i, ex in enumerate(ds):
        if max_examples and n_added >= max_examples:
            break
        prompt = ex.get(prompt_field, "")
        sol = ex.get(sol_field, "")
        if not prompt.strip() or not sol.strip():
            continue
        rows.append({
            "source": source,
            "id": f"{source}_{start_idx + i}",
            "prompt": prompt.strip(),
            "canonical_solution": sol.strip(),
            "full": (prompt.strip() + "\n\n" + sol.strip())
        })
        n_added += 1
    return n_added

n_mbpp = add_examples(mbpp, "mbpp", "text", "code")
n_he = add_examples(humaneval, "humaneval", "prompt", "canonical_solution", start_idx=n_mbpp)
n_cp = 0
if codeparrot:
    n_cp = add_examples(codeparrot, "codeparrot", "content", "content", start_idx=n_mbpp+n_he, max_examples=10000)

# --- Create DataFrame ---
df_corpus = pd.DataFrame(rows)

# --- Deduplicate ---
df_corpus = df_corpus.drop_duplicates(subset=["full"]).reset_index(drop=True)
print(f"Combined corpus size (dedup): {len(df_corpus)}")
print(f"MBPP added: {n_mbpp}, HumanEval added: {n_he}, CodeParrot added: {n_cp}")

# --- Ensure types ---
df_corpus["prompt"] = df_corpus["prompt"].astype(str)
df_corpus["canonical_solution"] = df_corpus["canonical_solution"].astype(str)

# Ensure the directory exists
os.makedirs(os.path.dirname(COMBINED_RAG_CORPUS), exist_ok=True)

# Save the combined corpus
df_corpus.to_csv(COMBINED_RAG_CORPUS, index=False)
print(f"Saved combined corpus to {COMBINED_RAG_CORPUS}")



# --- Create ground truth mapping ---
ground_truth_ids_for_task = defaultdict(list)

for idx, row in df_corpus.iterrows():
    solution = row['canonical_solution'].strip()
    ground_truth_ids_for_task[solution].append(row['id'])

# Convert defaultdict to regular dict for saving
ground_truth_ids_for_task = dict(ground_truth_ids_for_task)

# Ensure the knowledge base directory exists
os.makedirs(os.path.dirname(GROUND_TRUTH_JSON), exist_ok=True)

# Save the ground truth mapping
with open(GROUND_TRUTH_JSON, "w", encoding="utf-8") as f:
    json.dump(ground_truth_ids_for_task, f, ensure_ascii=False, indent=2)

print(f"✅ Saved ground truth mapping for {len(ground_truth_ids_for_task)} unique solutions to '{GROUND_TRUTH_JSON}'")

