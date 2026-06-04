"""CLI entry point for the RAG ingest pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("pmt_faq_bot")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


def cmd_ingest(args: argparse.Namespace) -> None:
    from .config import Config
    from .incremental import run_ingest

    cfg = Config()
    source = Path(args.source)
    if not source.is_dir():
        log.error("Source directory does not exist: %s", source)
        sys.exit(1)

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    stats = run_ingest(source, cfg, mode=args.mode)
    elapsed = time.monotonic() - start

    report = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "source": str(source),
        "total_files": stats.total_files,
        "added": stats.added,
        "modified": stats.modified,
        "deleted": stats.deleted,
        "skipped": stats.skipped,
        "total_chunks": stats.total_chunks,
        "errors": stats.errors,
        "duration_seconds": round(elapsed, 2),
    }

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    print(report_json)

    if log_dir:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = log_dir / f"ingest-{ts}.json"
        log_file.write_text(report_json, encoding="utf-8")
        log.info("Report written to %s", log_file)

    if stats.errors:
        log.warning("%d errors occurred during ingest", len(stats.errors))
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    from .config import Config
    from .qdrant_store import QdrantStore

    cfg = Config()
    store = QdrantStore(cfg)
    try:
        info = store.collection_info()
        print(json.dumps(info, indent=2, ensure_ascii=False))
    finally:
        store.close()


def cmd_process_images(args: argparse.Namespace) -> None:
    from .image_processing import ImageProcessingConfig, process_image_manifest

    export_dir = _resolve_export_dir(Path(args.export_dir))
    if not export_dir.is_dir():
        log.error("Export directory does not exist: %s", export_dir)
        sys.exit(1)
    if not (export_dir / "image_manifest.jsonl").is_file():
        _fail_missing_image_manifest(export_dir)

    cfg = ImageProcessingConfig(
        cache_dir=Path(args.cache_dir)
        if args.cache_dir
        else export_dir / "image_cache",
        output_path=Path(args.output) if args.output else None,
        ocr_engine=args.ocr_engine,
        ocr_version=args.ocr_version,
        caption_model=args.caption_model,
        caption_prompt_version=args.caption_prompt_version,
        vlm_url=args.vlm_url,
        vlm_timeout=args.vlm_timeout,
    )
    report = process_image_manifest(export_dir, cfg)
    print(json.dumps(report.__dict__, indent=2, ensure_ascii=False))
    if report.errors:
        sys.exit(1)


def _resolve_export_dir(export_dir: Path) -> Path:
    """Resolve a Confluence export root from a direct or one-level parent path."""
    export_dir = Path(export_dir)
    if (export_dir / "image_manifest.jsonl").is_file():
        return export_dir
    if not export_dir.is_dir():
        return export_dir

    child_exports = [
        child
        for child in export_dir.iterdir()
        if child.is_dir() and (child / "image_manifest.jsonl").is_file()
    ]
    if len(child_exports) == 1:
        resolved = child_exports[0]
        log.info("Using nested Confluence export directory: %s", resolved)
        return resolved
    return export_dir


def _fail_missing_image_manifest(export_dir: Path) -> None:
    entries = []
    try:
        entries = sorted(path.name for path in export_dir.iterdir())[:20]
    except OSError:
        pass
    log.error(
        "image_manifest.jsonl does not exist in %s. "
        "Set EXPORT_DIR to the complete Confluence export root, not only documents/. "
        "Expected files: documents/, assets/, manifest.jsonl, image_manifest.jsonl. "
        "Directory entries: %s",
        export_dir,
        ", ".join(entries) if entries else "<unavailable>",
    )
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pmt_faq_bot",
        description="Import PMT FAQ Markdown documents into Qdrant and serve RAG answers",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Run document ingestion")
    p_ingest.add_argument(
        "--source", required=True, help="Directory containing .md files"
    )
    p_ingest.add_argument(
        "--mode",
        choices=["auto", "full"],
        default="auto",
        help="auto=incremental, full=rebuild (default: auto)",
    )
    p_ingest.add_argument(
        "--log-dir", default=None, help="Directory to write JSON run logs"
    )

    # status
    sub.add_parser("status", help="Show Qdrant collection status")

    # process-images
    p_images = sub.add_parser(
        "process-images",
        help="Run OCR/VLM processing for an exported Confluence image_manifest.jsonl",
    )
    p_images.add_argument(
        "--export-dir",
        required=True,
        help="Confluence export directory containing image_manifest.jsonl",
    )
    p_images.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory (default: <export-dir>/image_cache)",
    )
    p_images.add_argument(
        "--output",
        default=None,
        help="Optional output JSONL path (default: <export-dir>/processed_image_manifest.jsonl)",
    )
    p_images.add_argument(
        "--ocr-engine",
        default="paddleocr",
        help="OCR engine name for cache versioning; use 'none' to disable OCR",
    )
    p_images.add_argument(
        "--ocr-version",
        default="unknown",
        help="OCR version string for cache versioning",
    )
    p_images.add_argument(
        "--caption-model",
        default=os.getenv("IMAGE_CAPTION_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
        help="OpenAI-compatible vision model name",
    )
    p_images.add_argument(
        "--caption-prompt-version",
        default="v1",
        help="Caption prompt version for cache versioning",
    )
    p_images.add_argument(
        "--vlm-url",
        default=os.getenv("IMAGE_VLM_URL", ""),
        help="OpenAI-compatible VLM base URL. If empty, caption falls back to filename.",
    )
    p_images.add_argument(
        "--vlm-timeout",
        type=float,
        default=float(os.getenv("IMAGE_VLM_TIMEOUT", "120")),
        help="VLM request timeout in seconds",
    )

    # serve
    p_serve = sub.add_parser("serve", help="Run the RAG API server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host")
    p_serve.add_argument("--port", type=int, default=8088, help="Bind port")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "process-images":
        cmd_process_images(args)
    elif args.command == "serve":
        cmd_serve(args)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    log.info("Starting RAG API server on %s:%d", args.host, args.port)
    uvicorn.run(
        "src.server.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
