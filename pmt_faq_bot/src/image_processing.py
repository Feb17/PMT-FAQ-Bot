"""Offline processing for exported Confluence images.

This module turns exported image assets into text metadata that later RAG ingest
can index as image chunks. It intentionally treats OCR and VLM captioning as
pluggable dependencies so tests can run without heavy model packages.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import request


DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_CAPTION_PROMPT = """请分析这张 Confluence 文档图片，并只输出 JSON。
字段:
- caption: 一句话中文描述
- image_type: screenshot|diagram|table|photo|other
- summary: 2-4 句中文说明关键信息
- visible_text: 图中主要文字，尽量原样保留，不翻译错误码、命令、IP、路径、按钮文字
- entities: 关键实体数组
- is_decorative: true/false，logo、图标、装饰图为 true
"""


class OcrEngine(Protocol):
    def extract_text(self, image_path: Path) -> str:
        ...


class Captioner(Protocol):
    def describe(self, image_path: Path) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ImageProcessingConfig:
    cache_dir: Path
    output_path: Path | None = None
    ocr_engine: str = "paddleocr"
    ocr_version: str = "unknown"
    caption_model: str = field(
        default_factory=lambda: os.getenv(
            "IMAGE_CAPTION_MODEL", "Qwen/Qwen3-VL-4B-Instruct"
        )
    )
    caption_prompt_version: str = DEFAULT_PROMPT_VERSION
    caption_prompt: str = DEFAULT_CAPTION_PROMPT
    vlm_url: str = field(default_factory=lambda: os.getenv("IMAGE_VLM_URL", ""))
    vlm_timeout: float = field(
        default_factory=lambda: float(os.getenv("IMAGE_VLM_TIMEOUT", "120"))
    )


@dataclass
class ImageProcessingReport:
    total_images: int = 0
    processed: int = 0
    cached: int = 0
    skipped_unavailable: int = 0
    missing_assets: int = 0
    errors: list[str] = field(default_factory=list)


class NullOcrEngine:
    """Fallback OCR engine used when PaddleOCR is not configured."""

    def extract_text(self, image_path: Path) -> str:
        return ""


class PaddleOcrEngine:
    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install offline wheels or pass a custom OCR engine."
            ) from exc
        self._engine = PaddleOCR(use_angle_cls=True, lang="ch")

    def extract_text(self, image_path: Path) -> str:
        result = self._engine.ocr(str(image_path), cls=True)
        texts = list(_walk_ocr_text(result))
        return "\n".join(text.strip() for text in texts if text and text.strip())


class OpenAiVisionCaptioner:
    """OpenAI-compatible VLM caption client for Qwen3-VL-Instruct style models."""

    def __init__(self, cfg: ImageProcessingConfig) -> None:
        if not cfg.vlm_url:
            raise RuntimeError("IMAGE_VLM_URL is not configured")
        self._url = cfg.vlm_url.rstrip("/") + "/v1/chat/completions"
        self._model = cfg.caption_model
        self._prompt = cfg.caption_prompt
        self._timeout = cfg.vlm_timeout

    def describe(self, image_path: Path) -> dict[str, Any]:
        image_bytes = image_path.read_bytes()
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        data_url = (
            f"data:{mime_type};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
        req = request.Request(
            self._url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
        )
        with request.urlopen(req, timeout=self._timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return _parse_caption_json(content)


def process_image_manifest(
    export_dir: Path,
    cfg: ImageProcessingConfig,
    ocr_engine: OcrEngine | None = None,
    captioner: Captioner | None = None,
) -> ImageProcessingReport:
    """Process ``image_manifest.jsonl`` and write processed image metadata."""
    export_dir = Path(export_dir)
    manifest_path = export_dir / "image_manifest.jsonl"
    output_path = cfg.output_path or export_dir / "processed_image_manifest.jsonl"
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(manifest_path)
    report = ImageProcessingReport(total_images=len(records))
    ocr = ocr_engine if ocr_engine is not None else _default_ocr_engine(cfg)
    vlm = captioner if captioner is not None else _default_captioner(cfg)
    output_records: list[dict[str, Any]] = []

    for record in records:
        if record.get("status") != "downloaded":
            output_records.append(
                {
                    **record,
                    "imageProcessingStatus": "skipped_unavailable",
                    "embeddingText": "",
                }
            )
            report.skipped_unavailable += 1
            continue

        image_path = export_dir / str(record.get("relativePath", ""))
        if not image_path.is_file():
            output_records.append(
                {
                    **record,
                    "imageProcessingStatus": "missing_asset",
                    "embeddingText": "",
                }
            )
            report.missing_assets += 1
            continue

        image_bytes = image_path.read_bytes()
        sha256 = "sha256:" + hashlib.sha256(image_bytes).hexdigest()
        cache_path = _cache_path(cfg.cache_dir, sha256)
        cached = _read_valid_cache(cache_path, cfg)
        if cached is not None:
            output_records.append(_merge_processed_record(record, cached, "cached"))
            report.cached += 1
            continue

        processed = _process_one_image(image_path, record, sha256, cfg, ocr, vlm)
        cache_path.write_text(
            json.dumps(processed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_records.append(_merge_processed_record(record, processed, "processed"))
        report.processed += 1

    _write_jsonl(output_path, output_records)
    return report


def _process_one_image(
    image_path: Path,
    record: dict[str, Any],
    sha256: str,
    cfg: ImageProcessingConfig,
    ocr: OcrEngine,
    captioner: Captioner,
) -> dict[str, Any]:
    ocr_text = ""
    ocr_error = ""
    try:
        ocr_text = ocr.extract_text(image_path)
    except Exception as exc:
        ocr_error = str(exc)[:500]

    caption_data: dict[str, Any] = {}
    caption_error = ""
    try:
        caption_data = _normalize_caption(captioner.describe(image_path))
    except Exception as exc:
        caption_error = str(exc)[:500]
        caption_data = _fallback_caption(record)

    dimensions = _read_image_dimensions(image_path)
    processed = {
        "sha256": sha256,
        "ocrEngine": cfg.ocr_engine,
        "ocrVersion": cfg.ocr_version,
        "captionModel": cfg.caption_model,
        "captionPromptVersion": cfg.caption_prompt_version,
        "width": dimensions.get("width"),
        "height": dimensions.get("height"),
        "ocrText": ocr_text,
        "caption": caption_data["caption"],
        "imageType": caption_data["image_type"],
        "summary": caption_data["summary"],
        "visibleText": caption_data["visible_text"],
        "entities": caption_data["entities"],
        "isDecorative": caption_data["is_decorative"],
        "embeddingText": _build_embedding_text(
            record=record,
            ocr_text=ocr_text,
            caption_data=caption_data,
        ),
    }
    if ocr_error:
        processed["ocrError"] = ocr_error
    if caption_error:
        processed["captionError"] = caption_error
    return processed


def _default_ocr_engine(cfg: ImageProcessingConfig) -> OcrEngine:
    if cfg.ocr_engine.lower() in {"", "none", "null"}:
        return NullOcrEngine()
    return PaddleOcrEngine()


def _default_captioner(cfg: ImageProcessingConfig) -> Captioner:
    if cfg.vlm_url:
        return OpenAiVisionCaptioner(cfg)
    return _FallbackCaptioner()


class _FallbackCaptioner:
    def describe(self, image_path: Path) -> dict[str, Any]:
        return {
            "caption": f"Confluence page image: {image_path.name}",
            "image_type": "other",
            "summary": "",
            "visible_text": "",
            "entities": [],
            "is_decorative": False,
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"image manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid image manifest record at {path}:{line_no}")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _cache_path(cache_dir: Path, sha256: str) -> Path:
    return cache_dir / f"{sha256.replace(':', '-')}.json"


def _read_valid_cache(
    cache_path: Path, cfg: ImageProcessingConfig
) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached.get("ocrEngine") != cfg.ocr_engine:
        return None
    if cached.get("captionModel") != cfg.caption_model:
        return None
    if cached.get("captionPromptVersion") != cfg.caption_prompt_version:
        return None
    return cached if isinstance(cached, dict) else None


def _merge_processed_record(
    record: dict[str, Any], processed: dict[str, Any], status: str
) -> dict[str, Any]:
    return {
        **record,
        **processed,
        "imageProcessingStatus": status,
    }


def _normalize_caption(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    entities = data.get("entities")
    if not isinstance(entities, list):
        entities = []
    return {
        "caption": str(data.get("caption") or "").strip(),
        "image_type": _normalize_image_type(str(data.get("image_type") or "other")),
        "summary": str(data.get("summary") or "").strip(),
        "visible_text": str(data.get("visible_text") or "").strip(),
        "entities": [str(item) for item in entities if str(item).strip()],
        "is_decorative": bool(data.get("is_decorative", False)),
    }


def _fallback_caption(record: dict[str, Any]) -> dict[str, Any]:
    filename = str(record.get("filename") or record.get("relativePath") or "image")
    return {
        "caption": f"Confluence page image: {filename}",
        "image_type": "other",
        "summary": "",
        "visible_text": "",
        "entities": [],
        "is_decorative": False,
    }


def _normalize_image_type(value: str) -> str:
    allowed = {"screenshot", "diagram", "table", "photo", "other"}
    return value if value in allowed else "other"


def _build_embedding_text(
    record: dict[str, Any],
    ocr_text: str,
    caption_data: dict[str, Any],
) -> str:
    parts = [
        f"图片文件: {record.get('filename', '')}",
        f"图片说明: {caption_data.get('caption', '')}",
        f"图片类型: {caption_data.get('image_type', '')}",
    ]
    summary = str(caption_data.get("summary") or "").strip()
    visible = str(caption_data.get("visible_text") or "").strip()
    if summary:
        parts.append(f"图片摘要: {summary}")
    if visible:
        parts.append(f"图片可见文字: {visible}")
    if ocr_text.strip():
        parts.append(f"图片OCR文本: {ocr_text.strip()}")
    entities = caption_data.get("entities") or []
    if entities:
        parts.append("图片实体: " + ", ".join(str(item) for item in entities))
    return "\n".join(parts)


def _parse_caption_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("caption response is not a JSON object")
    return data


def _walk_ocr_text(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return [value[0]]
    if isinstance(value, list):
        for item in value:
            texts.extend(_walk_ocr_text(item))
    return texts


def _read_image_dimensions(path: Path) -> dict[str, int | None]:
    data = path.read_bytes()[:256]
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return {"width": width, "height": height}
    if data.startswith(b"\xff\xd8"):
        return _read_jpeg_dimensions(path)
    return {"width": None, "height": None}


def _read_jpeg_dimensions(path: Path) -> dict[str, int | None]:
    data = path.read_bytes()
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker in {0xD8, 0xD9}:
            continue
        if i + 2 > len(data):
            break
        length = struct.unpack(">H", data[i : i + 2])[0]
        if marker in range(0xC0, 0xC4) and i + 7 < len(data):
            height, width = struct.unpack(">HH", data[i + 3 : i + 7])
            return {"width": width, "height": height}
        i += length
    return {"width": None, "height": None}
