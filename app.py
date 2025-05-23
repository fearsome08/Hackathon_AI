import gradio as gr
import os
import shutil
import re

from universal_parser import run_parser
from chunking_json import main as chunking
from Index_chunks_faiss import main as index_chunks
from query_faiss import retrieve_results_structured

os.makedirs("temp", exist_ok=True)

def clean_text(text: str) -> str:
    # Fix common joined words by adding spaces before capital letters preceded by lowercase (e.g. SeriesResistanceCancellation -> Series Resistance Cancellation)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Replace multiple newlines with two newlines (Markdown paragraph breaks)
    text = re.sub(r'\n{2,}', '\n\n', text)
    # Remove weird joined words like "acceptstheSMBuscommunicationprotocol" => "accepts the SMBus communication protocol"
    # Here we add spaces between lowercase-uppercase, also between words ending and next word starting (simplistic)
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    # Optional: Fix bullet points merged with words (•ProcessorandFPGATemperatureMonitoring -> • Processor and FPGA Temperature Monitoring)
    text = re.sub(r'(•)([A-Za-z])', r'\1 \2', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # You can add more heuristic fixes here if needed
    return text.strip()

def upload_and_index(file):
    if file is None:
        return "❌ No file uploaded."
    
    try:
        filename = os.path.basename(file)
        filepath = os.path.join("temp", filename)
        shutil.copyfile(file, filepath)
    except Exception as e:
        return f"❌ Failed to save file: {str(e)}"
    
    try:
        parsed = run_parser(filepath)
        chunking(parsed, "temp/enriched_chunks.json")
        index_chunks("temp/enriched_chunks.json", "temp/index.faiss", "temp/index_meta.json")
    except Exception as e:
        return f"❌ Failed during parsing/indexing: {str(e)}"
    
    return f"✅ Document '{filename}' uploaded and indexed successfully!"

def search_index(query):
    if not query.strip():
        return "Please enter a valid search query."

    try:
        results = retrieve_results_structured(
            query=query,
            faiss_index_path="temp/index.faiss",
            metadata_path="temp/index_meta.json",
            top_k=5
        )
        if not results:
            return "No results found."

        print("DEBUG: Raw results:", results)  # 👈 Add this line to inspect

        output = ""
        for i, res in enumerate(results, start=1):
            score = round(res.get("score", 0), 4)
            chunk = res.get("chunk", "") or res.get("text", "")  # fallback to 'text' if 'chunk' missing
            if not chunk:
                chunk = "[Empty Chunk]"
            cleaned_chunk = clean_text(chunk)
            output += f"Result {i} (Score: {score}):\n\n{cleaned_chunk}\n\n{'-'*40}\n\n"
        return output
    except Exception as e:
        return f"❌ Search failed: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("# 📄 Universal Document Search with Gradio")

    with gr.Tab("Upload and Index"):
        file_input = gr.File(label="Upload Document", file_types=[".pdf", ".docx", ".txt", ".csv", ".xlsx", ".rtf", ".doc", ".wps", ".wpd"])
        upload_btn = gr.Button("Upload and Index")
        upload_output = gr.Markdown()

        upload_btn.click(upload_and_index, inputs=file_input, outputs=upload_output)

    with gr.Tab("Search Indexed Content"):
        query_input = gr.Textbox(label="Enter search query")
        search_btn = gr.Button("Search")
        search_output = gr.Textbox(lines=20, interactive=False, label="Search Results")

        search_btn.click(search_index, inputs=query_input, outputs=search_output)

demo.launch()
