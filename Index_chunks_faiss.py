import os
import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# ---- Convert Markdown Images to HTML <img> Tags ----
def convert_markdown_to_html_images(text):
    def replacer(match):
        path = match.group(1).replace("\\", "/")
        return f'<img src="{path}" width="400">'
    return re.sub(r'!\[Image\]\((.*?)\)', replacer, text)

# ---- Normalize Paths in Markdown ----
def normalize_markdown_image_paths(text):
    def replacer(m):
        path = m.group(1).replace("\\\\", "/").replace("\\", "/")
        return f'![Image]({path})'
    return re.sub(r'!\[Image\]\((.*?)\)', replacer, text)

# ---- Split Mixed Content and Clean Markdown Tables ----
def clean_and_split_tables(text):
    table_pattern = re.compile(r"((?:\|.*\|\n)+\|[-:\s|]*\|\n(?:\|.*\|\n?)+)", re.MULTILINE)
    parts = []
    last_end = 0
    for match in table_pattern.finditer(text):
        start, end = match.span()
        if start > last_end:
            non_table = text[last_end:start].strip()
            if non_table:
                parts.append(non_table)
        parts.append(match.group(0).strip())
        last_end = end
    if last_end < len(text):
        tail = text[last_end:].strip()
        if tail:
            parts.append(tail)
    return parts

# ---- Load Embedding Model ----
embedding_model = SentenceTransformer("intfloat/e5-base-v2")

# ---- Embed Document Chunks ----
def embed_chunks(chunks):
    texts = []
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
        page = chunk.get("page", None) if isinstance(chunk, dict) else None
        if page is not None:
            text = f"[Page {page}] " + text
        html_text = convert_markdown_to_html_images(text)
        texts.append("passage: " + html_text)
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)

# ---- FAISS Index + Metadata Storage ----
def store_embeddings(chunks, embeddings, index_path, meta_path):
    dim = embeddings.shape[1]

    # 🔄 Replaced FlatIP with HNSW index
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 = M parameter
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    def normalize_path(path):
        return path.replace("\\", "/")

    metadata = []
    for i, chunk in enumerate(chunks):
        entry = {
            "chunk_id": i,
            "text": normalize_markdown_image_paths(chunk.get("text", "") if isinstance(chunk, dict) else chunk),
        }
        if isinstance(chunk, dict) and "images" in chunk:
            entry["images"] = [normalize_path(p) for p in chunk["images"]]
        if isinstance(chunk, dict) and "page" in chunk:
            entry["page"] = chunk["page"]
        metadata.append(entry)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ FAISS HNSW index saved to `{index_path}`")
    print(f"✅ Metadata saved to `{meta_path}`")
    print(f"📐 Embedding dimension: {dim}, Total chunks: {len(chunks)}")

# ---- Main Execution ----
def main(chunks_path, index_path, meta_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("❌ Invalid or empty chunks format in JSON.")

    clean_chunks = []
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else chunk.get("text", "")
        split_parts = clean_and_split_tables(text)
        for part in split_parts:
            clean_chunks.append({"chunk_id": len(clean_chunks), "text": part})

    embeddings = embed_chunks(clean_chunks)
    store_embeddings(clean_chunks, embeddings, index_path, meta_path)

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed document chunks and store in FAISS HNSW index")
    parser.add_argument("--chunks", required=True, help="Path to input chunks.json")
    parser.add_argument("--index", default="index.faiss", help="Output FAISS index file")
    parser.add_argument("--meta", default="index_meta.json", help="Output metadata JSON file")
    args = parser.parse_args()

    main(args.chunks, args.index, args.meta)
