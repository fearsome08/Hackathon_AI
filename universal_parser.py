import os
import fitz  # PyMuPDF
import textract
import pandas as pd
import pdfplumber
import chardet
import subprocess
import numpy as np
import io
import threading
import pytesseract
from scipy.stats import entropy
from PIL import Image
from docx import Document
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from typing import Dict, Union, List
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5

def ocr_image_with_tesseract(img: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def extract_visual_figures_from_pdf(filepath, dpi=300, min_length=100):
    figures = []
    try:
        doc = fitz.open(filepath)
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = ocr_image_with_tesseract(img)
            # Heuristic: consider long OCR results that are NOT captured by pdfplumber
            if ocr_text and len(ocr_text) >= min_length:
                figures.append(f"**Figure (Page {page_num+1})**:\n```\n{ocr_text}\n```")
    except Exception as e:
        print(f"OCR visual figure extraction failed: {e}")
    return figures

def clean_text(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16')

def detect_encoding(filepath):
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def parse_txt(filepath):
    encoding = detect_encoding(filepath)
    with open(filepath, 'r', encoding=encoding) as f:
        return {'text': f.read()}

def parse_csv(filepath):
    df = pd.read_csv(filepath)
    return {'text': df.to_string(index=False), 'tables': [df.to_dict(orient='records')]}

def is_image_blank(image_path, std_threshold=6, mean_threshold=245, entropy_threshold=1.0, resize_to=(64, 64)):
    try:
        img = Image.open(image_path).convert("L").resize(resize_to)
        arr = np.array(img)

        stddev = arr.std()
        mean = arr.mean()
        hist, _ = np.histogram(arr, bins=256, range=(0, 255), density=True)
        ent = entropy(hist + 1e-10)

        if stddev < std_threshold and mean > mean_threshold and ent < entropy_threshold:
            return True
        return False
    except:
        return False

def is_image_too_small(image_path, min_size=20):
    try:
        img = Image.open(image_path)
        width, height = img.size
        return width < min_size or height < min_size
    except:
        return False

def get_image_hash(image_bytes):
    return md5(image_bytes).hexdigest()

def extract_unique_images_from_pdf(filepath, output_dir="extracted_images/pdf", max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    seen_xrefs = set()
    doc = fitz.open(filepath)
    lock = threading.Lock()

    def process_xref(xref, page_num, img_index):
        try:
            with lock:
                if xref in seen_xrefs:
                    return None
                seen_xrefs.add(xref)

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            if is_image_blank_or_small(image_bytes):
                return None

            image_ext = base_image["ext"]
            image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            return image_path
        except Exception as e:
            print(f"Image processing error: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                futures.append(executor.submit(process_xref, xref, page_num, img_index))

        for future in futures:
            result = future.result()
            if result:
                images.append(result)

    return images


def process_image_from_pdf(doc, page_num, img_index, img, hashes, output_dir):
    try:
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_hash = get_image_hash(image_bytes)
        with hashes_lock:
            if image_hash in hashes:
                return None
            hashes.add(image_hash)
        if is_image_blank_or_small(image_bytes):
            return None
        image_ext = base_image["ext"]
        image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
        image_path = os.path.join(output_dir, image_filename)
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        return image_path
    except Exception as e:
        print(f"Error processing image on page {page_num+1} img {img_index+1}: {e}")
        return None


def is_image_blank_or_small(image_bytes, std_threshold=6, mean_threshold=245, entropy_threshold=1.0, resize_to=(64, 64), min_size=20):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(resize_to)
        arr = np.array(img)
        stddev = arr.std()
        mean = arr.mean()
        hist, _ = np.histogram(arr, bins=256, range=(0, 255), density=True)
        ent = entropy(hist + 1e-10)
        if stddev < std_threshold and mean > mean_threshold and ent < entropy_threshold:
            return True
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        if width < min_size or height < min_size:
            return True
        return False
    except Exception as e:
        print(f"Image blank/small check error: {e}")
        return True  # treat errors as invalid image

hashes_lock = threading.Lock()


def extract_comments_from_docx(filepath):
    comments = []
    doc = Document(filepath)
    part = doc.part
    rels = part.rels

    for rel in rels:
        if rels[rel].reltype == RT.COMMENTS:
            comments_part = rels[rel].target_part
            comments_xml = comments_part._blob.decode('utf-8')
            import xml.etree.ElementTree as ET
            tree = ET.fromstring(comments_xml)
            for comment in tree.findall('.//w:comment', namespaces={'w': qn('w')}):
                text = ''.join(comment.itertext()).strip()
                if text:
                    comments.append(text)
    return comments

def parse_docx(filepath):
    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith('Heading')]
    tables = []
    for table in doc.tables:
        data = []
        for row in table.rows:
            data.append([cell.text for cell in row.cells])
        tables.append(data)

    comments = extract_comments_from_docx(filepath)
    images = extract_images_from_docx(filepath)

    return {
        'text': "\n\n".join(paragraphs),
        'headings': headings,
        'tables': tables,
        'images': images,
        'annotations': comments
    }

def extract_annotations_from_pdf(filepath):
    annotations = []
    doc = fitz.open(filepath)
    for page in doc:
        for annot in page.annots():
            info = annot.info
            if 'content' in info:
                annotations.append(info['content'])
    return annotations

def parse_pdf(filepath):
    content = []
    tables = []
    annotations = extract_annotations_from_pdf(filepath)
    images = extract_unique_images_from_pdf(filepath)

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                content.append(text)
            try:
                tables.extend(page.extract_tables())
            except:
                pass

    visual_figures = extract_visual_figures_from_pdf(filepath)

    return {
        'text': '\n\n'.join(content),
        'tables': tables,
        'images': images,
        'annotations': annotations,
        'visual_figures': visual_figures
    }


def parse_with_textract(filepath):
    try:
        text = textract.process(filepath).decode("utf-8")
        return {'text': text}
    except Exception as e:
        return {'error': f"Textract failed: {e}"}

def convert_with_libreoffice(filepath) -> Union[str, None]:
    try:
        output_dir = "/tmp"
        subprocess.run(["libreoffice", "--headless", "--convert-to", "docx", filepath, "--outdir", output_dir],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        filename = os.path.basename(filepath).rsplit('.', 1)[0] + '.docx'
        return os.path.join(output_dir, filename)
    except Exception as e:
        print(f"LibreOffice conversion failed: {e}")
        return None

def parse_xlsx(filepath: str) -> Dict[str, Union[str, List]]:
    xlsx_data = {
        "tables": [],
        "headings": [],
        "text": ""
    }

    try:
        xls = pd.read_excel(filepath, sheet_name=None, engine="openpyxl")
        xlsx_data["headings"] = list(xls.keys())

        for sheet_name, df in xls.items():
            if df.empty:
                continue
            df = df.fillna("").astype(str)
            table = [df.columns.tolist()] + df.values.tolist()
            xlsx_data["tables"].append(table)

        if not xlsx_data["tables"]:
            xlsx_data["text"] = "[No data found in any Excel sheets]"

    except Exception as e:
        xlsx_data["text"] = f"[Error parsing XLSX file: {e}]"

    return xlsx_data

def parse_file(filepath: str) -> Dict[str, Union[str, List]]:
    ext = filepath.lower().split('.')[-1]

    if ext == 'pdf':
        return parse_pdf(filepath)
    elif ext == 'docx':
        return parse_docx(filepath)
    elif ext == 'txt':
        return parse_txt(filepath)
    elif ext == 'csv':
        return parse_csv(filepath)
    elif ext == "xlsx":
        return parse_xlsx(filepath)
    elif ext in ['rtf', 'doc', 'wpd', 'wps']:
        converted = convert_with_libreoffice(filepath)
        if converted and os.path.exists(converted):
            return parse_docx(converted)
        else:
            return parse_with_textract(filepath)
    else:
        return {'error': f"Unsupported file type: {ext}"}

def format_markdown(result: Dict[str, Union[str, List]]) -> str:
    md = []
    headings = result.get("headings", [])
    if headings:
        md.append("## \ud83d\udccc Headings")
        for idx, h in enumerate(headings, 1):
            md.append(f"{idx}. {h}")

    text = result.get("text", "")
    if text:
        md.append("\n## \ud83d\udcc4 Text\n")
        md.append(text.strip())

    tables = result.get("tables", [])
    if tables:
        md.append("\n## \ud83d\udcca Tables\n")
        for i, table in enumerate(tables):
            if isinstance(table, list) and all(isinstance(row, list) for row in table):
                md.append(f"**Table {i+1}:**\n")
                headers = [str(h) if h is not None else "" for h in table[0]]
                md.append("| " + " | ".join(headers) + " |")
                md.append("|" + " --- |" * len(headers))
                for row in table[1:]:
                    row_clean = [str(cell) if cell is not None else "" for cell in row]
                    md.append("| " + " | ".join(row_clean) + " |")
                md.append("")
            elif isinstance(table, list) and all(isinstance(row, dict) for row in table):
                headers = list(table[0].keys())
                md.append(f"**Table {i+1}:**\n")
                md.append("| " + " | ".join(headers) + " |")
                md.append("|" + " --- |" * len(headers))
                for row in table:
                    md.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
                md.append("")

    annotations = result.get("annotations", [])
    if annotations:
        md.append("\n## \ud83d\udcc2 Annotations\n")
        for idx, note in enumerate(annotations, 1):
            md.append(f"- {note}")

    images = result.get("images", [])
    if images:
        md.append("\n## 🖼️ Images\n")
        for i, img_path in enumerate(images, 1):
            md.append(f"![Image {i}]({img_path})")
    
    visual_figures = result.get("visual_figures", [])
    if visual_figures:
        md.append("\n## 🧾 OCR-based Visual Figures\n")
        md.extend(visual_figures)

    return "\n".join(md)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python universal_parser.py <file_path>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    result = parse_file(filepath)
    markdown_output = format_markdown(result)

    output_path = os.path.splitext(filepath)[0] + "_output.md"

    cleaned_markdown = clean_text(markdown_output)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_markdown)


    print(f"✅ Markdown output saved to: {output_path}")