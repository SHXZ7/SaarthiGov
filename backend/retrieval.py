import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

INDEX_PATH = os.path.join(project_root, "data/ration_card/faiss/ration_card.index")
META_PATH = os.path.join(project_root, "data/ration_card/faiss/ration_card_metadata.json")

# Load once at startup
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def retrieve_chunks(query: str, k: int = 3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        chunk = metadata[idx]
        results.append({
            "service": chunk["service"],
            "state": chunk["state"],
            "section": chunk["section"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results
