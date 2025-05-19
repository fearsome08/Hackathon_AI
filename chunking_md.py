import os
import json
import re
import shutil
from pathlib import Path

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(tokenizer.encode(text))
except ImportError:
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    def count_tokens(text):
        return len(word_tokenize(text))

def extract_image_paths(markdown):
    return re.findall(r'!\[.*?\]\((.*?)\)', markdown)

def load_all_extracted_images(images_dir):
    if not os.path.isdir(images_dir):
        return []
    return sorted([str(p) for p in Path(images_dir).rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])

def split_markdown_into_chunks(markdown_text, max_tokens=600):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for block in markdown_text.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        tokens = count_tokens(block)
        if current_tokens + tokens > max_tokens:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [block]
            current_tokens = tokens
        else:
            current_chunk.append(block)
            current_tokens += tokens
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    return chunks

def chunk_markdown_with_images(markdown_path, output_json_path, chunk_output_dir, extracted_images_dir):
    os.makedirs(chunk_output_dir, exist_ok=True)

    with open(markdown_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    chunks = split_markdown_into_chunks(md_text)

    referenced_images = extract_image_paths(md_text)
    all_extracted_images = load_all_extracted_images(extracted_images_dir)

    # Include all extracted images, even if not referenced
    extra_images = [img for img in all_extracted_images if img not in referenced_images]

    json_chunks = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i+1}"
        chunk_dir = os.path.join(chunk_output_dir, chunk_id)
        os.makedirs(chunk_dir, exist_ok=True)

        # Find image references in this chunk
        chunk_images = extract_image_paths(chunk)

        # Copy referenced images to this chunk's folder
        copied_images = []
        for img_path in chunk_images:
            if os.path.exists(img_path):
                target_path = os.path.join(chunk_dir, os.path.basename(img_path))
                shutil.copy(img_path, target_path)
                copied_images.append(target_path)

        # Round-robin assign extra images
        if extra_images:
            extra_img = extra_images[i % len(extra_images)]
            target_path = os.path.join(chunk_dir, os.path.basename(extra_img))
            shutil.copy(extra_img, target_path)
            copied_images.append(target_path)

        json_chunks.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "images": copied_images
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(json_chunks)} chunks to: {output_json_path}")
    print(f"🖼️ Image subfolders created in: {chunk_output_dir}")

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, help="Path to the generated Markdown file")
    parser.add_argument("--images", required=True, help="Path to extracted images dir (e.g. extracted_images/pdf/)")
    parser.add_argument("--out_json", default="chunked_output.json", help="Output JSON path")
    parser.add_argument("--out_dir", default="md_chunks", help="Directory to save per-chunk image folders")
    args = parser.parse_args()

    chunk_markdown_with_images(
        markdown_path=args.md,
        output_json_path=args.out_json,
        chunk_output_dir=args.out_dir,
        extracted_images_dir=args.images
    )
