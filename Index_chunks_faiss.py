import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings, texts

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, path):
    faiss.write_index(index, path)

def save_metadata(chunks, path):
    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "images": chunk.get("images", [])
        }
        for chunk in chunks
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def index_chunks(json_input_path, faiss_output_path, metadata_output_path):
    chunks = load_chunks(json_input_path)
    embeddings, _ = compute_embeddings(chunks)
    index = build_faiss_index(embeddings)
    save_index(index, faiss_output_path)
    save_metadata(chunks, metadata_output_path)
    print(f"✅ Indexed {len(chunks)} chunks and saved index/metadata.")

# Usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to chunked JSON file")
    parser.add_argument("--faiss_output", default="faiss_index.bin", help="Output path for FAISS index")
    parser.add_argument("--metadata_output", default="chunk_metadata.json", help="Output path for metadata")
    args = parser.parse_args()

    index_chunks(args.input_json, args.faiss_output, args.metadata_output)
