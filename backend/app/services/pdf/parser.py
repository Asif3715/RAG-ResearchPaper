from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz
import pymupdf4llm

from backend.app.core.config import EXTRACTED_DIR


NOISE_PATTERNS = (
    r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*",
    r"----- Start of picture text -----.*?----- End of picture text -----",
)


def sha256_file(file_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def save_parsed_output(doc_id: str, parsed: dict[str, Any], cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or EXTRACTED_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{doc_id}.json"
    cache_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path


def load_parsed_output(doc_id: str, cache_dir: Path | None = None) -> dict[str, Any] | None:
    cache_dir = cache_dir or EXTRACTED_DIR
    cache_path = cache_dir / f"{doc_id}.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def delete_parsed_output(doc_id: str, cache_dir: Path | None = None) -> None:
    cache_dir = cache_dir or EXTRACTED_DIR
    cache_path = cache_dir / f"{doc_id}.json"
    if cache_path.exists():
        cache_path.unlink()


def _clean_text(text: str) -> str:
    cleaned = text
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.S)
    lines = []
    for line in cleaned.splitlines():
        normalized = normalize_text(line)
        if normalized == "##":
            continue
        if normalized:
            lines.append(normalized)
    return "\n".join(lines).strip()


def _extract_pages(file_path: Path) -> list[dict[str, Any]]:
    doc = fitz.open(file_path)
    try:
        pages: list[dict[str, Any]] = []
        for page_index in range(doc.page_count):
            md = pymupdf4llm.to_markdown(doc, pages=[page_index], page_chunks=True)
            if isinstance(md, list):
                raw_text = "\n\n".join(str(item.get("text", "")) for item in md if str(item.get("text", "")).strip())
            else:
                raw_text = str(md)
            page_text = _clean_text(raw_text)
            if page_text:
                pages.append({"page_number": page_index + 1, "text": page_text})
        return pages
    finally:
        doc.close()


def _recursive_chunks(text: str, max_chars: int = 1100, overlap_chars: int = 120) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    for sep in separators:
        if sep in text:
            parts = [part.strip() for part in text.split(sep) if part.strip()]
            if len(parts) > 1:
                chunks: list[str] = []
                current = ""
                for part in parts:
                    candidate = f"{current}{sep}{part}".strip() if current else part
                    if len(candidate) <= max_chars:
                        current = candidate
                        continue
                    if current:
                        chunks.extend(_recursive_chunks(current, max_chars=max_chars, overlap_chars=overlap_chars))
                    current = part
                if current:
                    chunks.extend(_recursive_chunks(current, max_chars=max_chars, overlap_chars=overlap_chars))
                return chunks

    step = max(1, max_chars - overlap_chars)
    return [text[i : i + max_chars].strip() for i in range(0, len(text), step) if text[i : i + max_chars].strip()]


def _build_chunks(pages: list[dict[str, Any]], doc_id: str, title: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for page in pages:
        for chunk_text in _recursive_chunks(page["text"]):
            chunk_index += 1
            chunks.append(
                {
                    "id": f"{doc_id}_chunk_{chunk_index}",
                    "type": "text",
                    "content": chunk_text,
                    "metadata": {
                        "page_number": page["page_number"],
                        "page_range": [page["page_number"], page["page_number"]],
                        "source": "pymupdf4llm",
                        "chunk_index": chunk_index,
                    },
                    "doc_id": doc_id,
                    "doc_title": title,
                }
            )
    return chunks


def parse_pdf(file_path: str | Path, cache_dir: Path | None = None) -> dict[str, Any]:
    file_path = Path(file_path)
    doc_id = sha256_file(file_path)
    cached = load_parsed_output(doc_id, cache_dir=cache_dir)
    if cached is not None:
        return cached

    pages = _extract_pages(file_path)
    title = file_path.stem
    markdown = "\n\n".join(page["text"] for page in pages)
    chunks = _build_chunks(pages, doc_id=doc_id, title=title)

    parsed = {
        "doc_id": doc_id,
        "title": title,
        "markdown": markdown,
        "pages": pages,
        "raw_blocks": pages,
        "chunks": chunks,
    }
    save_parsed_output(doc_id, parsed, cache_dir=cache_dir)
    return parsed


def parse_pdf_bytes(pdf_bytes: bytes, original_name: str = "uploaded.pdf", cache_dir: Path | None = None) -> dict[str, Any]:
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()
    display_title = Path(original_name).stem or "uploaded"
    cached = load_parsed_output(doc_id, cache_dir=cache_dir)
    if cached is not None:
        if not cached.get("title") or cached.get("title") == doc_id:
            cached["title"] = display_title
            save_parsed_output(doc_id, cached, cache_dir=cache_dir)
        return cached

    temp_path = (cache_dir or EXTRACTED_DIR).parent / "uploads" / f"{doc_id}.pdf"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(pdf_bytes)
    parsed = parse_pdf(temp_path, cache_dir=cache_dir)
    parsed["title"] = display_title if not parsed.get("title") or parsed.get("title") == doc_id else parsed.get("title")
    save_parsed_output(doc_id, parsed, cache_dir=cache_dir)
    return parsed
