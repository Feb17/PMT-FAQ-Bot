"""RAG prompt construction and citation formatting."""

from __future__ import annotations

import os
from urllib.parse import quote

from ..retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "You are an ISA-CN IT operations assistant for Bosch dedicated infrastructure. "
    "Answer the user's question based ONLY on the context provided below. "
    "If the context does not contain enough information to answer, say so honestly. "
    "Cite your sources using numbered references like [1], [2] etc. "
    "When the context includes images (marked with 📷), embed them inline in your "
    "answer at the appropriate location by copying the provided markdown image "
    "syntax exactly. "
    "Respond in the same language the user uses."
)

NO_RESULT_REPLY = (
    "抱歉，我在知识库中没有找到与您的问题直接相关的信息。"
    "请尝试用不同的关键词重新提问，或联系 BD/ISA-CN 运维团队获取帮助。"
)


def _build_chunk_image_refs(
    chunk: RetrievedChunk, asset_base_url: str | None
) -> str:
    """Build inline image reference text for one chunk's associated images."""
    candidates: list[dict] = []
    if chunk.image:
        candidates.append(chunk.image)
    candidates.extend(chunk.images or [])

    if not candidates:
        return ""

    lines: list[str] = []
    for img in candidates:
        if not isinstance(img, dict):
            continue
        if img.get("is_decorative") or img.get("isDecorative"):
            continue
        url = _image_url(img, asset_base_url)
        caption = _image_caption(img)
        if url:
            lines.append(f"> 📷 {caption}\n> ![{caption}]({url})")
    return "\n".join(lines)


def build_messages(
    user_query: str,
    chunks: list[RetrievedChunk],
    history: list[dict],
    max_turns: int = 4,
    asset_base_url: str | None = None,
) -> list[dict]:
    """Assemble the final message list for the LLM with RAG context."""
    parts = []
    for i, c in enumerate(chunks):
        block = (
            f"[{i + 1}] 文档: {c.title} / 章节: {c.section_title}\n"
            f"Source: {c.source_url}\n"
            f"内容:\n{c.content}"
        )
        img_refs = _build_chunk_image_refs(c, asset_base_url)
        if img_refs:
            block += "\n" + img_refs
        parts.append(block)
    context_block = "\n\n".join(parts)
    system = f"{SYSTEM_PROMPT}\n\n## Context\n\n{context_block}"

    trimmed = _trim_history(history, max_turns)

    return [
        {"role": "system", "content": system},
        *trimmed,
        {"role": "user", "content": user_query},
    ]


def format_sources(
    chunks: list[RetrievedChunk],
) -> str:
    """Format a Sources block to append after the LLM answer."""
    if not chunks:
        return ""
    lines = ["\n\n---\n**Sources:**"]
    for i, c in enumerate(chunks):
        title = c.title or c.doc_id
        section = f" / {c.section_title}" if c.section_title else ""
        url = c.source_url or ""
        if url:
            lines.append(f"[{i + 1}] [{title}{section}]({url})")
        else:
            lines.append(f"[{i + 1}] {title}{section}")
    return "\n".join(lines)


def format_related_images(
    chunks: list[RetrievedChunk],
    asset_base_url: str | None = None,
    max_images: int = 3,
) -> str:
    """Format de-duplicated related images as Markdown image links."""
    images: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        candidates: list[dict] = []
        if chunk.image:
            candidates.append(chunk.image)
        candidates.extend(chunk.images or [])
        for image in candidates:
            if not isinstance(image, dict):
                continue
            if image.get("is_decorative") or image.get("isDecorative"):
                continue
            image_url = _image_url(image, asset_base_url)
            if not image_url:
                continue
            key = str(image.get("image_id") or image.get("imageId") or image_url)
            if key in seen:
                continue
            seen.add(key)
            images.append({**image, "resolved_url": image_url})
            if len(images) >= max_images:
                break
        if len(images) >= max_images:
            break

    if not images:
        return ""

    lines = ["", "**相关图片:**", ""]
    for idx, image in enumerate(images, start=1):
        caption = _image_caption(image)
        lines.append(f"![图{idx}: {caption}]({image['resolved_url']})")
        source_url = str(image.get("source_url") or image.get("sourceUrl") or "")
        if source_url:
            lines.append(f"来源: [原始页面]({source_url})")
        lines.append("")
    return "\n".join(lines).rstrip()


def _image_url(image: dict, asset_base_url: str | None = None) -> str:
    for key in ("url", "public_url", "publicUrl"):
        value = str(image.get(key) or "").strip()
        if value:
            return value

    base_url = (asset_base_url or os.getenv("RAG_ASSET_BASE_URL", "")).strip()
    relative_path = str(
        image.get("relative_path")
        or image.get("relativePath")
        or image.get("markdown_path")
        or image.get("markdownPath")
        or ""
    ).strip()
    if not relative_path:
        return ""
    relative_path = relative_path.replace("\\", "/").lstrip("/")
    if relative_path.startswith("../"):
        relative_path = relative_path[3:]
    if base_url:
        return f"{base_url.rstrip('/')}/{quote(relative_path, safe='/')}"
    return relative_path


def _image_caption(image: dict) -> str:
    caption = str(image.get("caption") or image.get("filename") or "相关图片")
    return caption.replace("[", "(").replace("]", ")").replace("\n", " ").strip()


def _trim_history(messages: list[dict], max_turns: int) -> list[dict]:
    """Keep the most recent N user-assistant turn pairs from history.

    Filters out system messages and keeps at most max_turns * 2 messages.
    """
    relevant = [
        m for m in messages
        if m.get("role") in ("user", "assistant")
    ]
    # Each "turn" is a user + assistant pair
    keep = max_turns * 2
    if len(relevant) > keep:
        relevant = relevant[-keep:]
    return relevant
