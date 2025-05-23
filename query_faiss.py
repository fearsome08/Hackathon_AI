import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import base64

def encode_query(query, model):
    query_text = f"query: {query}"
    embedding = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.reshape(1, -1)

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def search_index(query_embedding, index, top_k=5):
    scores, indices = index.search(query_embedding, top_k)
    return scores[0], indices[0]

def load_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def image_to_base64(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
            ext = os.path.splitext(path)[1][1:].lower()
            return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"❌ Error encoding image {path}: {e}")
        return None

# ---- For Streamlit or API use ----
def retrieve_results_structured(
    query,
    faiss_index_path="index.faiss",
    metadata_path="index_meta.json",
    model_name="intfloat/e5-base-v2",
    top_k=5
):
    model = SentenceTransformer(model_name)
    index = load_faiss_index(faiss_index_path)
    metadata = load_metadata(metadata_path)

    query_embedding = encode_query(query, model)
    scores, indices = search_index(query_embedding, index, top_k)

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        chunk = metadata[idx]
        images_data = []
        for img_path in chunk.get("images", []):
            abs_path = os.path.abspath(img_path)
            base64_img = image_to_base64(abs_path)
            images_data.append({
                "path": img_path,
                "base64": base64_img
            })

        results.append({
            "score": float(score),
            "text": chunk["text"],
            "heading": chunk.get("heading"),
            "page": chunk.get("page"),
            "chunk_id": chunk.get("chunk_id", idx),
            "images": images_data
        })

    return results

# ---- Optional: Markdown Rendering (if needed for testing or logging) ----
def render_results_markdown(results):
    md_output = "# 🔍 Top RAG Results\n"
    for i, res in enumerate(results):
        md_output += f"\n## Result {i+1} (Score: {res['score']:.4f})\n"
        md_output += f"**Chunk ID**: `{res['chunk_id']}`\n\n"
        if res.get("heading"):
            md_output += f"**Heading**: {res['heading']}\n\n"
        if res.get("page") is not None:
            md_output += f"**Page**: {res['page']}\n\n"
        md_output += f"**Text:**\n\n{res['text']}\n\n"

        if res["images"]:
            md_output += "**Images:**\n\n"
            for img in res["images"]:
                if img["base64"]:
                    md_output += f"![Image]({img['base64']})\n\n"
                else:
                    md_output += f"![Missing image]({img['path']})\n\n"
    return md_output

# ---- CLI Entrypoint for Testing ----
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--faiss_index", default="index.faiss")
    parser.add_argument("--metadata", default="index_meta.json")
    parser.add_argument("--model_name", default="intfloat/e5-base-v2")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--output", default="retrieval_results.md")
    args = parser.parse_args()

    print("📦 Retrieving results...")
    results = retrieve_results_structured(
        query=args.query,
        faiss_index_path=args.faiss_index,
        metadata_path=args.metadata,
        model_name=args.model_name,
        top_k=args.top_k
    )

    markdown = render_results_markdown(results)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"✅ Results written to `{args.output}`")

if __name__ == "__main__":
    main()
