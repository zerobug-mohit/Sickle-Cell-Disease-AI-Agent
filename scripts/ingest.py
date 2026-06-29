#!/usr/bin/env python3
"""
Run ONCE to build the FAISS vector store from SCD training materials.
Re-run only when source documents change.

Usage:
    python scripts/ingest.py
"""
import json
import os
import sys

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    TRAINING_MATERIALS_DIR,
    FAISS_INDEX_PATH,
    OPENAI_API_KEY,
    OCR_ENABLED,
    OCR_LANG,
    OCR_DPI,
    TESSERACT_CMD,
)
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
EMBED_BATCH_SIZE = 100

# OCR only contributes a line if the text layer is missing it. A page whose text
# layer already has at least this many characters still gets OCR'd, but only the
# net-new lines (e.g. text inside diagrams) are merged in.
_OCR_MIN_LINE_LEN = 4        # ignore tiny OCR fragments / noise
_OCR_DEDUP_LINE_LEN = 8      # lines >= this are checked against the text layer to avoid re-adding prose

client = OpenAI(api_key=OPENAI_API_KEY)

_ocr_ready = False


def _init_ocr():
    """Return the pytesseract module if OCR is usable, else None."""
    global _ocr_ready
    if not OCR_ENABLED:
        return None
    try:
        import pytesseract
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        # Probe the binary so we fail loudly here, not mid-document.
        pytesseract.get_tesseract_version()
        import fitz  # noqa: F401  (PyMuPDF, for rendering)
        from PIL import Image  # noqa: F401
        if not _ocr_ready:
            print(f"  OCR enabled (lang={OCR_LANG}, dpi={OCR_DPI})")
            _ocr_ready = True
        return pytesseract
    except Exception as e:
        print(f"  WARNING: OCR requested but unavailable ({e}). Falling back to text layer only.")
        return None


def _normalize(s: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in s).split())


def _merge_ocr(text_layer: str, ocr_text: str) -> str:
    """Keep the clean text layer; append only OCR lines it does not already contain."""
    base = (text_layer or "").strip()
    base_norm = _normalize(base)
    extra: list[str] = []
    for raw in (ocr_text or "").splitlines():
        line = raw.strip()
        if len(line) < _OCR_MIN_LINE_LEN:
            continue
        norm = _normalize(line)
        if not norm:
            continue
        # Skip lines already present in the text layer (avoids duplicating prose).
        if len(line) >= _OCR_DEDUP_LINE_LEN and norm in base_norm:
            continue
        extra.append(line)
    if not extra:
        return base
    return (base + "\n" + "\n".join(extra)).strip() if base else "\n".join(extra)


def _ocr_page(pytesseract, page) -> str:
    import io
    from PIL import Image
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=OCR_LANG)


def extract_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    text_pages = [page.extract_text() or "" for page in reader.pages]

    pyt = _init_ocr()
    if pyt is None:
        return "\n".join(text_pages)

    import fitz
    doc = fitz.open(path)
    merged_pages: list[str] = []
    added = 0
    for i, text_layer in enumerate(text_pages):
        ocr_text = ""
        if i < doc.page_count:
            try:
                ocr_text = _ocr_page(pyt, doc[i])
            except Exception as e:
                print(f"    OCR failed on page {i + 1}: {e}")
        before = len(text_layer)
        merged = _merge_ocr(text_layer, ocr_text)
        if len(merged) > before:
            added += 1
        merged_pages.append(merged)
    doc.close()
    print(f"    OCR added content on {added}/{len(text_pages)} page(s)")
    return "\n".join(merged_pages)


def extract_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_documents(directory: str) -> list[tuple[str, str]]:
    docs = []
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.lower().endswith(".pdf"):
            print(f"  Reading PDF: {fname}")
            docs.append((fname, extract_pdf(fpath)))
        elif fname.lower().endswith(".docx"):
            print(f"  Reading DOCX: {fname}")
            docs.append((fname, extract_docx(fpath)))
    return docs


def chunk_documents(docs: list[tuple[str, str]]) -> tuple[list[str], list[dict]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_texts: list[str] = []
    all_meta: list[dict] = []

    for source, text in docs:
        if not text.strip():
            print(f"  WARNING: {source} yielded no extractable text.")
            print(f"  This PDF may be image-based (scanned). See CLAUDE.md for the pymupdf fallback.")
            continue
        chunks = splitter.split_text(text)
        print(f"  {source}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_meta.append({
                "chunk_id": f"{source}_{i}",
                "text": chunk,
                "source": source,
            })

    return all_texts, all_meta


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    total_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
        batch_num = i // EMBED_BATCH_SIZE + 1
        print(f"  Embedded batch {batch_num}/{total_batches} ({len(all_embeddings)}/{len(chunks)} chunks)")

    return all_embeddings


def build_and_save_index(embeddings: list[list[float]], metadata: list[dict]) -> None:
    emb_array = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(emb_array)

    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
    with open(os.path.join(FAISS_INDEX_PATH, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nFAISS index saved -> {FAISS_INDEX_PATH}/")
    print(f"  Vectors: {index.ntotal}  Dimensions: {dim}")


def main() -> None:
    print(f"Loading documents from: {TRAINING_MATERIALS_DIR}")
    docs = load_documents(TRAINING_MATERIALS_DIR)
    if not docs:
        print("No PDF or DOCX files found. Copy the PDFs to data/training_materials/ and re-run.")
        sys.exit(1)
    print(f"Loaded {len(docs)} document(s)\n")

    print("Chunking documents...")
    texts, metadata = chunk_documents(docs)
    if not texts:
        print("ERROR: All documents yielded zero text chunks.")
        print("The PDFs are likely scanned/image-based. Install pymupdf and use the fallback in CLAUDE.md.")
        sys.exit(1)
    print(f"Total chunks: {len(texts)}\n")

    print("Embedding chunks (may take several minutes for large documents)...")
    embeddings = embed_chunks(texts)
    print()

    print("Building FAISS index...")
    build_and_save_index(embeddings, metadata)
    print(f"\nDone. {len(docs)} documents -> {len(texts)} chunks indexed.")


if __name__ == "__main__":
    main()
