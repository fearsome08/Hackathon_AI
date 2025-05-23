import re
import json
import argparse
from collections import defaultdict
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
OCR_BLOCK_RE = re.compile(r"<!-- OCR (FIGURE|ANNOTATION) START -->(.*?)<!-- OCR \1 END -->", re.DOTALL)

MAX_TOKENS = 512
MIN_TOKENS = 100

def find_headings(text):
    return [(m.start(), m.end(), len(m.group(1)), m.group(2).strip()) for m in HEADING_RE.finditer(text)]

def find_images(text):
    return IMAGE_RE.findall(text)

def split_text_by_ocr_blocks(text):
    segments = []
    last_end = 0
    for m in OCR_BLOCK_RE.finditer(text):
        if m.start() > last_end:
            segments.append((text[last_end:m.start()], None))
        ocr_type = m.group(1)
        ocr_content = m.group(2).strip()
        segments.append((ocr_content, {"ocr_type": ocr_type}))
        last_end = m.end()
    if last_end < len(text):
        segments.append((text[last_end:], None))
    return segments

def get_heading_for_pos(headings, pos):
    current_heading = None
    for start, end, level, heading_text in headings:
        if start > pos:
            break
        current_heading = (level, heading_text)
    return current_heading

def chunk_text(text, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS):
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=False)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    chunks = []
    start_token_idx = 0
    while start_token_idx < len(input_ids):
        end_token_idx = min(start_token_idx + max_tokens, len(input_ids))
        start_char = offsets[start_token_idx][0]
        end_char = offsets[end_token_idx-1][1]
        chunk = text[start_char:end_char]

        if chunks and len(tokenizer(chunks[-1])["input_ids"]) < min_tokens:
            chunks[-1] += chunk
        else:
            chunks.append(chunk)

        start_token_idx = end_token_idx
    return chunks

def process_single_text(text, base_metadata=None, base_chunk_index=0):
    """
    Process one markdown text segment (from a JSONL chunk).
    Returns list of chunks with metadata enriched with headings, images, OCR info.
    """
    headings = find_headings(text)
    segments = split_text_by_ocr_blocks(text)

    all_chunks = []
    chunk_index = base_chunk_index

    for segment_text, ocr_meta in segments:
        if ocr_meta:
            chunk_metadata = {
                "chunk_index": chunk_index,
                "ocr_type": ocr_meta["ocr_type"],
                "heading": None,
                "images": [],
            }
            if base_metadata:
                chunk_metadata.update(base_metadata)  # preserve other metadata
            all_chunks.append({"text": segment_text, "metadata": chunk_metadata})
            chunk_index += 1
        else:
            text_chunks = chunk_text(segment_text)
            char_pos = 0
            for chunk_text_part in text_chunks:
                start_pos_in_md = text.find(chunk_text_part, char_pos)
                char_pos = start_pos_in_md + len(chunk_text_part)
                heading = get_heading_for_pos(headings, start_pos_in_md)
                chunk_images = find_images(chunk_text_part)

                chunk_metadata = {
                    "chunk_index": chunk_index,
                    "ocr_type": None,
                    "heading": heading,
                    "images": chunk_images,
                }
                if base_metadata:
                    chunk_metadata.update(base_metadata)  # preserve other metadata

                all_chunks.append({"text": chunk_text_part, "metadata": chunk_metadata})
                chunk_index += 1
    return all_chunks

def process_jsonl(input_path):
    all_processed_chunks = []
    chunk_counter = 0

    # 1) Build images by page dictionary
    images_by_page = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("type") == "image":
                page = chunk.get("page")
                path = chunk.get("content")
                if page is not None and path:
                    images_by_page[page].append(path)

    # 2) Process text chunks and inject images based on page
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            # Skip non-text chunks
            if chunk.get("type") not in ["text", "markdown"]:
                continue

            text = chunk.get("text") or chunk.get("content") or ""
            if not text.strip():
                continue

            metadata = {k: v for k, v in chunk.items() if k not in ("text", "content")}
            page = metadata.get("page")
            # Inject images that belong to the same page
            metadata["images"] = images_by_page.get(page, [])

            processed_chunks = process_single_text(text, base_metadata=metadata, base_chunk_index=chunk_counter)
            all_processed_chunks.extend(processed_chunks)
            chunk_counter += len(processed_chunks)

    return all_processed_chunks

def save_chunks(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def main(input_path, output_path):
    chunks = process_jsonl(input_path)
    save_chunks(chunks, output_path)
    print(f"Processed and saved {len(chunks)} chunks to {output_path}")
    return chunks
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL chunks of markdown, enrich headings/images/OCR")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", default="enriched_chunks.json", help="Output JSON file path")
    args = parser.parse_args()

    process_json_file(args.input, args.output)
