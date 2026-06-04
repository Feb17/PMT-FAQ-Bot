"""Helpers for connecting processed image metadata to exported documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROCESSED_IMAGE_STATUSES = {"processed", "cached"}


def find_export_root(source_dir: Path) -> Path | None:
    """Find the Confluence export root for a documents directory or export root."""
    source_dir = Path(source_dir)
    candidates = [source_dir, source_dir.parent]
    for candidate in candidates:
        if (candidate / "manifest.jsonl").is_file() or (
            candidate / "processed_image_manifest.jsonl"
        ).is_file():
            return candidate
    return None


def load_processed_images_by_doc_id(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Return processed image records keyed by document id.

    ``source_dir`` can be either the export root or its ``documents`` directory.
    Failed/unavailable and decorative images are intentionally omitted.
    """
    export_root = find_export_root(source_dir)
    if export_root is None:
        return {}

    manifest_path = export_root / "manifest.jsonl"
    processed_path = export_root / "processed_image_manifest.jsonl"
    if not manifest_path.is_file() or not processed_path.is_file():
        return {}

    page_to_doc = _load_page_to_doc_id(manifest_path)
    page_to_images: dict[str, list[dict[str, Any]]] = {}
    for record in _load_jsonl(processed_path):
        if record.get("imageProcessingStatus") not in PROCESSED_IMAGE_STATUSES:
            continue
        if bool(record.get("isDecorative", False)):
            continue
        page_id = str(record.get("pageId") or "")
        if not page_id:
            continue
        page_to_images.setdefault(page_id, []).append(normalize_image_record(record))

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for page_id, doc_id in page_to_doc.items():
        images = page_to_images.get(page_id, [])
        if images:
            by_doc[doc_id] = images
    return by_doc


def normalize_image_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": str(record.get("imageId") or record.get("image_id") or ""),
        "page_id": str(record.get("pageId") or record.get("page_id") or ""),
        "filename": str(record.get("filename") or ""),
        "relative_path": str(record.get("relativePath") or record.get("relative_path") or ""),
        "markdown_path": str(record.get("markdownPath") or record.get("markdown_path") or ""),
        "source_url": str(record.get("sourceUrl") or record.get("source_url") or ""),
        "sha256": str(record.get("sha256") or ""),
        "caption": str(record.get("caption") or ""),
        "summary": str(record.get("summary") or ""),
        "ocr_text": str(record.get("ocrText") or record.get("ocr_text") or ""),
        "visible_text": str(record.get("visibleText") or record.get("visible_text") or ""),
        "image_type": str(record.get("imageType") or record.get("image_type") or "other"),
        "embedding_text": str(record.get("embeddingText") or record.get("embedding_text") or ""),
        "is_decorative": bool(record.get("isDecorative", False)),
    }


def hash_image_records(records: list[dict[str, Any]]) -> str:
    """Stable JSON representation used as part of document content hashes."""
    relevant = [
        {
            "image_id": record.get("image_id"),
            "sha256": record.get("sha256"),
            "caption": record.get("caption"),
            "summary": record.get("summary"),
            "ocr_text": record.get("ocr_text"),
            "visible_text": record.get("visible_text"),
            "embedding_text": record.get("embedding_text"),
        }
        for record in records
    ]
    return json.dumps(relevant, ensure_ascii=False, sort_keys=True)


def _load_page_to_doc_id(manifest_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for record in _load_jsonl(manifest_path):
        page_id = str(record.get("pageId") or record.get("page_id") or "")
        relative_path = str(record.get("relativePath") or "")
        if not page_id or not relative_path:
            continue
        mapping[page_id] = Path(relative_path).stem
    return mapping


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"invalid JSONL record at {path}:{line_no}")
        rows.append(row)
    return rows
