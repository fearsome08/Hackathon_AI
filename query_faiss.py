import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def query_index(query, faiss_path, metadata_path, model_name="all-MiniLM-L6-v2", top_k=5):
    model = SentenceTransformer(model_name)
    index = faiss.read_index(faiss_path)

    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# Example Usage
if __name__ == "__main__":
    query = "What are the Start-up and initialization of the SFD?"
    results = query_index(
        query,
        faiss_path="faiss_index.bin",
        metadata_path="chunk_metadata.json"
    )

    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(result['text'][:500])  # print truncated
