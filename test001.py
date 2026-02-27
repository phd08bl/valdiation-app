import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Text helpers
# -----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)      # collapse spaces
    s = re.sub(r"\n{3,}", "\n\n", s)   # collapse blank lines
    return s.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.
    Heuristic:
      - Prefer blank line separation
      - Fallback to grouping lines
    """
    txt = normalize_text(text)
    if not txt:
        return []

    # Prefer blank line split
    parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return parts

    # Fallback: join non-empty lines
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    if not lines:
        return []
    return [" ".join(lines).strip()]


# -----------------------------
# Docling parser (v2.75.0)
# -----------------------------
def _try_export_dict(doc: Any) -> Optional[Dict[str, Any]]:
    """
    Try to export docling document to dict (structure may vary by version).
    """
    for fn_name in ("export_to_dict", "to_dict", "model_dump"):
        if hasattr(doc, fn_name):
            try:
                fn = getattr(doc, fn_name)
                data = fn() if callable(fn) else fn
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return None


def _try_export_markdown(doc: Any) -> Optional[str]:
    """
    Try to export docling document to markdown string.
    """
    for fn_name in ("export_to_markdown", "to_markdown"):
        if hasattr(doc, fn_name):
            try:
                fn = getattr(doc, fn_name)
                md = fn() if callable(fn) else fn
                if isinstance(md, str) and md.strip():
                    return md
            except Exception:
                pass
    return None


def _extract_from_doc_dict(doc_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Attempt to extract per-page blocks/paragraphs from exported dict.
    This tries multiple possible keys because docling output schemas can vary.
    """
    paragraphs: List[Dict[str, Any]] = []

    # Common patterns: doc_dict["pages"] is a list
    pages = doc_dict.get("pages") or doc_dict.get("document", {}).get("pages") or []
    if not isinstance(pages, list) or not pages:
        return paragraphs

    for idx, page in enumerate(pages, start=1):
        # Try common page number keys
        page_num = (
            page.get("page_num")
            or page.get("page_number")
            or page.get("page")
            or idx
        )

        # Possible containers of text blocks
        blocks = (
            page.get("paragraphs")
            or page.get("blocks")
            or page.get("content")
            or page.get("items")
            or []
        )

        # Sometimes page itself has plain text
        page_text = page.get("text") or page.get("content_text") or ""
        if isinstance(page_text, str) and page_text.strip() and not blocks:
            for p in split_into_paragraphs(page_text):
                paragraphs.append({"content": p, "page_num": int(page_num), "bbox": None})
            continue

        if not isinstance(blocks, list):
            # If blocks isn’t list, fallback to stringify
            if page_text:
                for p in split_into_paragraphs(str(page_text)):
                    paragraphs.append({"content": p, "page_num": int(page_num), "bbox": None})
            continue

        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            text = blk.get("text") or blk.get("content") or blk.get("markdown") or ""
            if not isinstance(text, str) or not text.strip():
                continue

            bbox = blk.get("bbox") or blk.get("bounding_box") or blk.get("box")
            # Keep one paragraph per block (or split if block is long)
            for p in split_into_paragraphs(text):
                paragraphs.append({"content": p, "page_num": int(page_num), "bbox": bbox})

    return paragraphs


def parse_pdf_with_docling(
    pdf_path: str,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Parse PDF using docling (local, OSS) and return paragraphs compatible with your MinerU output:
      [{"content": ..., "page_num": ..., "bbox": ...}, ...]

    Strategy (robust):
      1) Use doc.export_to_dict() (best chance for per-page mapping)
      2) If fails, use doc.export_to_markdown() and split paragraphs (page_num defaults to 1)
    """
    pdf_path = str(Path(pdf_path))
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ---- docling import (v2.75.0) ----
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = getattr(result, "document", None)
    if doc is None:
        raise RuntimeError("Docling conversion failed: result.document is None")

    # 1) Try dict export for structured per-page extraction
    doc_dict = _try_export_dict(doc)
    if isinstance(doc_dict, dict):
        paragraphs = _extract_from_doc_dict(doc_dict)

        # Apply max_pages if requested
        if max_pages is not None:
            paragraphs = [p for p in paragraphs if int(p.get("page_num", 1)) <= max_pages]

        if paragraphs:
            return paragraphs

    # 2) Fallback: markdown export (no page mapping guarantee)
    md = _try_export_markdown(doc)
    if md:
        paras = split_into_paragraphs(md)
        if max_pages is not None:
            # markdown fallback doesn't know pages; just return all or first N paragraphs
            paras = paras[: max_pages * 50]  # heuristic
        return [{"content": p, "page_num": 1, "bbox": None} for p in paras]

    # 3) Last resort: stringify document
    txt = str(doc)
    paras = split_into_paragraphs(txt)
    return [{"content": p, "page_num": 1, "bbox": None} for p in paras]



# Change to your PDF path
pdf_path = "./data/ss123.pdf"

paragraphs = parse_pdf_with_docling(pdf_path)

print("\n====== 解析结果 (docling 2.75.0) ======")
print(f"Total paragraphs: {len(paragraphs)}\n")
for i, p in enumerate(paragraphs[:60]):
    preview = p["content"][:80].replace("\n", " ")
    print(f"[{i}] 第{p['page_num']}页: {preview}")