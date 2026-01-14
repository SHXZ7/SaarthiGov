import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "data/ration_card/faiss/ration_card.index"
META_FILE = "data/ration_card/faiss/ration_card_metadata.json"

# Load index and metadata
index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def search(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results


# Test queries
queries = [
    "What documents are needed for ration card?",
    "How long does ration card approval take?",
    "I don't have Aadhaar linked, what should I do?",
    "Akshaya centre ration card process"
]

for q in queries:
    print("\nQUERY:", q)
    results = search(q)
    for r in results:
        print("→", r["section"])
