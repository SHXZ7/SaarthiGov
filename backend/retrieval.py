import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Service configurations
SERVICES = {
    "ration_card": {
        "index_path": os.path.join(project_root, "data/ration_card/faiss/ration_card.index"),
        "meta_path": os.path.join(project_root, "data/ration_card/faiss/ration_card_metadata.json")
    },
    "birth_certificate": {
        "index_path": os.path.join(project_root, "data/birth_certificate/faiss/birth_certificate.index"),
        "meta_path": os.path.join(project_root, "data/birth_certificate/faiss/birth_certificate_metadata.json")
    },
    "unemployment_allowance": {
        "index_path": os.path.join(project_root, "data/unemployment/faiss/unemployment.index"),
        "meta_path": os.path.join(project_root, "data/unemployment/faiss/unemployment_metadata.json")
    }
}

# Load all indices and metadata at startup
indices = {}
metadata_store = {}

for service_name, paths in SERVICES.items():
    if os.path.exists(paths["index_path"]) and os.path.exists(paths["meta_path"]):
        indices[service_name] = faiss.read_index(paths["index_path"])
        with open(paths["meta_path"], "r", encoding="utf-8") as f:
            metadata_store[service_name] = json.load(f)
        print(f"Loaded index for: {service_name}")
    else:
        print(f"Warning: Index not found for {service_name}")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def get_available_services():
    """Return list of services with loaded indices"""
    return list(indices.keys())

def retrieve_chunks(query: str, service: str = None, k: int = 3):
    """
    Retrieve relevant chunks for a query with STRICT service isolation.
    """

    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    results = []

    # 🚨 STRICT service enforcement
    if not service:
        raise ValueError(
            "Service must be specified for retrieval to avoid cross-service leakage."
        )

    if service not in indices:
        raise ValueError(
            f"Service '{service}' not found. Available: {get_available_services()}"
        )

    # Search ONLY the requested service
    index = indices[service]
    metadata = metadata_store[service]

    scores, idxs = index.search(query_embedding, k)

    for idx, score in zip(idxs[0], scores[0]):
        if idx < len(metadata):
            chunk = metadata[idx]

            # 🛡️ Final safety guard
            if chunk.get("service") != service:
                continue

            results.append({
                "service": chunk["service"],
                "state": chunk["state"],
                "section": chunk["section"],
                "text": chunk["text"],
                "score": float(score)
            })

    # Sort by similarity
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
