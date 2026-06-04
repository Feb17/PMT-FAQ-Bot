"""Hierarchical chunking: split parsed documents into parent, child, and image chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .parser import (
    ParsedDocument,
    Section,
    _is_confluence_macro_noise,
    is_low_value_section,
    is_placeholder_content,
)


@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str  # "parent" | "child" | "image"
    parent_chunk_id: Optional[str]
    doc_id: str
    section_title: str
    content: str  # raw content without context prefix
    content_for_embedding: str  # content with context prefix
    is_low_value: bool = False  # True if from References/Others/etc.
    images: list[dict] = field(default_factory=list)
    image: Optional[dict] = None


def chunk_document(
    doc: ParsedDocument,
    cfg: Config,
    images: list[dict] | None = None,
) -> list[Chunk]:
    """Generate parent/child text chunks plus independent image chunks."""
    if not doc.body_sections:
        return []

    all_chunks: list[Chunk] = []
    child_only_mode = _is_short_document(doc, cfg)
    image_records = images or []
    emitted_image_ids: set[str] = set()

    for sec_idx, section in enumerate(doc.body_sections):
        parent_id = f"{doc.doc_id}__{_slug(section.title or 'body')}_{sec_idx}"
        section_images = _images_for_section(section, image_records)

        if child_only_mode:
            content = _join_section_content(section)
            if is_placeholder_content(content):
                image_chunks = _make_image_chunks(
                    doc, section.title, section_images, emitted_image_ids
                )
                all_chunks.extend(image_chunks)
                continue
            all_chunks.append(
                Chunk(
                    chunk_id=f"{parent_id}__c0",
                    chunk_type="child",
                    parent_chunk_id=None,
                    doc_id=doc.doc_id,
                    section_title=section.title,
                    content=content,
                    content_for_embedding=_make_embed_text(
                        doc, section.title, content
                    ),
                    is_low_value=is_low_value_section(section.title),
                    images=section_images,
                )
            )
            all_chunks.extend(
                _make_image_chunks(
                    doc, section.title, section_images, emitted_image_ids
                )
            )
            continue

        children = _split_section(section, doc, parent_id, cfg)
        children = _attach_section_images(children, section_images)
        if not children:
            all_chunks.extend(
                _make_image_chunks(
                    doc, section.title, section_images, emitted_image_ids
                )
            )
            continue

        parent_content = _join_section_content(section)
        all_chunks.append(
            Chunk(
                chunk_id=parent_id,
                chunk_type="parent",
                parent_chunk_id=None,
                doc_id=doc.doc_id,
                section_title=section.title,
                content=parent_content,
                content_for_embedding="",
                is_low_value=is_low_value_section(section.title),
                images=section_images,
            )
        )
        all_chunks.extend(children)
        all_chunks.extend(
            _make_image_chunks(doc, section.title, section_images, emitted_image_ids)
        )

    return all_chunks


def _is_short_document(doc: ParsedDocument, cfg: Config) -> bool:
    total = sum(len(s.content) for s in doc.body_sections)
    return total < cfg.chars_for_tokens(cfg.child_chunk_max_tokens)


def _join_section_content(section: Section) -> str:
    header = f"## {section.title}\n\n" if section.title else ""
    return (header + section.content).strip()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", text.lower()).strip("_")
    return s[:60] if s else "untitled"


def _make_embed_text(doc: ParsedDocument, section_title: str, content: str) -> str:
    parts = [f"文档: {doc.title}"]
    if doc.path:
        parts.append(f"路径: {doc.path}")
    if section_title:
        parts.append(f"章节: {section_title}")
    parts.append("---")
    parts.append(content)
    return "\n".join(parts)


def _make_image_chunks(
    doc: ParsedDocument,
    section_title: str,
    images: list[dict],
    emitted_image_ids: set[str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, image in enumerate(images):
        image_id = str(image.get("image_id") or image.get("filename") or idx)
        if image_id in emitted_image_ids:
            continue
        emitted_image_ids.add(image_id)
        content = str(image.get("embedding_text") or "").strip()
        if not content:
            content = _image_fallback_content(image)
        if is_placeholder_content(content):
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}__image_{_slug(image_id)}",
                chunk_type="image",
                parent_chunk_id=None,
                doc_id=doc.doc_id,
                section_title=section_title,
                content=content,
                content_for_embedding=_make_embed_text(doc, section_title, content),
                is_low_value=is_low_value_section(section_title),
                images=[image],
                image=image,
            )
        )
    return chunks


def _image_fallback_content(image: dict) -> str:
    parts = [
        f"图片文件: {image.get('filename', '')}",
        f"图片说明: {image.get('caption', '')}",
    ]
    summary = str(image.get("summary") or "").strip()
    visible_text = str(image.get("visible_text") or "").strip()
    ocr_text = str(image.get("ocr_text") or "").strip()
    if summary:
        parts.append(f"图片摘要: {summary}")
    if visible_text:
        parts.append(f"图片可见文字: {visible_text}")
    if ocr_text:
        parts.append(f"图片OCR文本: {ocr_text}")
    return "\n".join(parts)


def _images_for_section(section: Section, images: list[dict]) -> list[dict]:
    return _images_for_content(section.content, images)


def _attach_section_images(children: list[Chunk], images: list[dict]) -> list[Chunk]:
    if not children or not images:
        return children

    matched_any = False
    for child in children:
        child_images = _images_for_content(child.content, images)
        if child_images:
            child.images = child_images
            matched_any = True

    if not matched_any:
        children[0].images = images
    return children


def _images_for_content(content: str, images: list[dict]) -> list[dict]:
    matched: list[dict] = []
    for image in images:
        tokens = [
            str(image.get("markdown_path") or ""),
            str(image.get("relative_path") or ""),
            str(image.get("filename") or ""),
        ]
        if any(token and token in content for token in tokens):
            matched.append(image)
    return matched


def _split_section(
    section: Section,
    doc: ParsedDocument,
    parent_id: str,
    cfg: Config,
) -> list[Chunk]:
    """Split a section's content into child chunks."""
    blocks = _extract_blocks(section.content)
    children: list[Chunk] = []
    child_idx = 0

    for block in blocks:
        if block.kind == "table":
            for sub in _split_table(block.text, cfg):
                c = _make_child(doc, section.title, parent_id, child_idx, sub, cfg)
                if c is not None:
                    children.append(c)
                    child_idx += 1
        elif block.kind == "code":
            c = _make_child(doc, section.title, parent_id, child_idx, block.text, cfg)
            if c is not None:
                children.append(c)
                child_idx += 1
        else:
            for sub in _split_text(block.text, cfg):
                c = _make_child(doc, section.title, parent_id, child_idx, sub, cfg)
                if c is not None:
                    children.append(c)
                    child_idx += 1

    return children


def _make_child(
    doc: ParsedDocument,
    section_title: str,
    parent_id: str,
    idx: int,
    content: str,
    cfg: Config,
) -> Optional[Chunk]:
    if is_placeholder_content(content):
        return None
    return Chunk(
        chunk_id=f"{parent_id}__c{idx}",
        chunk_type="child",
        parent_chunk_id=parent_id,
        doc_id=doc.doc_id,
        section_title=section_title,
        content=content,
        content_for_embedding=_make_embed_text(doc, section_title, content),
        is_low_value=is_low_value_section(section_title),
    )


@dataclass
class _Block:
    kind: str  # "text" | "table" | "code"
    text: str


def _extract_blocks(content: str) -> list[_Block]:
    """Split markdown content into typed blocks (text, table, code)."""
    blocks: list[_Block] = []
    lines = _strip_noise_lines(content.split("\n"))
    buf: list[str] = []
    buf_kind = "text"
    in_code = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                if buf:
                    blocks.append(_Block(buf_kind, "\n".join(buf).strip()))
                    buf = []
                buf_kind = "code"
                in_code = True
                buf.append(line)
            else:
                buf.append(line)
                blocks.append(_Block("code", "\n".join(buf).strip()))
                buf = []
                buf_kind = "text"
                in_code = False
            continue

        if in_code:
            buf.append(line)
            continue

        if stripped.startswith("|"):
            if buf_kind != "table":
                if buf:
                    blocks.append(_Block(buf_kind, "\n".join(buf).strip()))
                    buf = []
                buf_kind = "table"
            buf.append(line)
        else:
            if buf_kind == "table":
                blocks.append(_Block("table", "\n".join(buf).strip()))
                buf = []
                buf_kind = "text"
            buf.append(line)

    if buf:
        text = "\n".join(buf).strip()
        if text:
            blocks.append(_Block(buf_kind, text))

    return [b for b in blocks if b.text]


def _split_table(table_text: str, cfg: Config) -> list[str]:
    """Split a large table into sub-tables, each retaining the header."""
    lines = table_text.strip().splitlines()
    if len(lines) <= cfg.large_table_split_rows + 2:
        return [table_text]

    header_lines: list[str] = []
    data_lines: list[str] = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        is_sep = all(re.fullmatch(r"-{3,}", c.strip()) for c in cells if c.strip())
        if i <= 1 or (i == 1 and is_sep):
            header_lines.append(line)
        elif is_sep and not data_lines:
            header_lines.append(line)
        else:
            data_lines.append(line)

    if not data_lines:
        return [table_text]

    chunks: list[str] = []
    step = cfg.large_table_split_rows
    for start in range(0, len(data_lines), step):
        batch = data_lines[start : start + step]
        chunks.append("\n".join(header_lines + batch))

    return chunks


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？\n])\s+")


def _split_text(text: str, cfg: Config) -> list[str]:
    """Split prose text into chunks of roughly target size with overlap."""
    text = text.strip()
    if not text:
        return []

    max_chars = cfg.chars_for_tokens(cfg.child_chunk_max_tokens)
    if len(text) <= max_chars:
        return [text]

    target_chars = cfg.chars_for_tokens(cfg.child_chunk_target_tokens)
    overlap_chars = cfg.chars_for_tokens(cfg.child_chunk_overlap_tokens)

    sentences = _SENTENCE_BOUNDARY.split(text)
    if len(sentences) <= 1:
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            return _merge_small_pieces(paragraphs, target_chars, overlap_chars)
        return _hard_split(text, max_chars, overlap_chars)

    return _merge_small_pieces(sentences, target_chars, overlap_chars)


def _merge_small_pieces(
    pieces: list[str], target_chars: int, overlap_chars: int
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        piece_len = len(piece)

        if current and current_len + piece_len > target_chars:
            chunks.append(
                "\n".join(current) if "\n" in "".join(current) else " ".join(current)
            )
            overlap_buf: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) > overlap_chars:
                    break
                overlap_buf.insert(0, p)
                overlap_len += len(p)
            current = overlap_buf
            current_len = overlap_len

        current.append(piece)
        current_len += piece_len

    if current:
        chunks.append(
            "\n".join(current) if "\n" in "".join(current) else " ".join(current)
        )

    return chunks if chunks else [" ".join(pieces)]


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end - overlap_chars if end < len(text) else end
    return chunks


def _strip_noise_lines(lines: list[str]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if stripped.startswith("|") or stripped.startswith("```"):
            kept.append(line)
            continue
        if _is_confluence_macro_noise(stripped):
            continue
        kept.append(line)
    return kept
