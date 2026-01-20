import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Paths
CHUNKS_FILE = os.path.join(project_root, "data/unemployment/chunks/unemployment_chunks.json")
INDEX_FILE = os.path.join(project_root, "data/unemployment/faiss/unemployment.index")
META_FILE = os.path.join(project_root, "data/unemployment/faiss/unemployment_metadata.json")

# Create faiss output directory if it doesn't exist
os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

# Load multilingual embedding model
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Generate embeddings
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Normalize embeddings (important for cosine similarity)
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine after normalization
index.add(embeddings)

# Save index
faiss.write_index(index, INDEX_FILE)

# Save metadata separately
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"FAISS index created with {index.ntotal} vectors")
