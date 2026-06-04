"""Incremental update logic: detect changes, ingest new/modified, remove deleted."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .bm25 import BM25Encoder
from .chunker import Chunk, chunk_document
from .config import Config
from .embedder import EmbeddingResult, Embedder
from .image_manifest import (
    hash_image_records,
    load_processed_images_by_doc_id,
)
from .parser import ParsedDocument, parse_file

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    total_files: int = 0
    added: int = 0
    modified: int = 0
    deleted: int = 0
    skipped: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)


def compute_file_hash(
    filepath: Path, image_records: list[dict] | None = None
) -> str:
    h = hashlib.sha256(filepath.read_bytes()).hexdigest()
    if image_records:
        combined = hashlib.sha256()
        combined.update(h.encode("utf-8"))
        combined.update(b"\n")
        combined.update(hash_image_records(image_records).encode("utf-8"))
        h = combined.hexdigest()
    return f"sha256:{h}"


def scan_source_dir(source_dir: Path) -> dict[str, Path]:
    """Return {doc_id: filepath} for all .md files under source_dir."""
    files: dict[str, Path] = {}
    for p in sorted(source_dir.rglob("*.md")):
        doc_id = p.stem
        files[doc_id] = p
    return files


def compute_diff(
    local_files: dict[str, Path],
    existing_hashes: dict[str, str],
    image_records_by_doc_id: dict[str, list[dict]] | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Compare local files against Qdrant state.

    Returns (to_add, to_modify, to_delete, to_skip) — each a list of doc_ids.
    """
    to_add: list[str] = []
    to_modify: list[str] = []
    to_skip: list[str] = []
    image_records_by_doc_id = image_records_by_doc_id or {}

    for doc_id, fpath in local_files.items():
        file_hash = compute_file_hash(
            fpath, image_records_by_doc_id.get(doc_id, [])
        )
        if doc_id not in existing_hashes:
            to_add.append(doc_id)
        elif existing_hashes[doc_id] != file_hash:
            to_modify.append(doc_id)
        else:
            to_skip.append(doc_id)

    to_delete = [did for did in existing_hashes if did not in local_files]

    return to_add, to_modify, to_delete, to_skip


def run_ingest(
    source_dir: Path,
    cfg: Config,
    mode: str = "auto",
) -> IngestStats:
    """Execute the full ingest pipeline.

    Args:
        source_dir: directory containing .md files
        cfg: configuration
        mode: "auto" (incremental) or "full" (rebuild)
    """
    stats = IngestStats()

    from .qdrant_store import QdrantStore

    store = QdrantStore(cfg)
    embedder = Embedder(cfg)

    try:
        vector_dim = embedder.detect_dim()
        store.ensure_collection(vector_dim)

        local_files = scan_source_dir(source_dir)
        image_records_by_doc_id = load_processed_images_by_doc_id(source_dir)
        stats.total_files = len(local_files)
        log.info("Found %d markdown files in %s", stats.total_files, source_dir)
        log.info(
            "Found processed images for %d documents",
            len(image_records_by_doc_id),
        )

        if mode == "full":
            existing_hashes: dict[str, str] = {}
            # In full mode, treat every file as new
            to_add = list(local_files.keys())
            to_modify: list[str] = []
            to_delete: list[str] = []
            to_skip: list[str] = []
            # Wipe collection data by deleting all known docs
            old_hashes = store.get_existing_doc_hashes()
            for doc_id in old_hashes:
                store.delete_by_doc_id(doc_id)
                log.debug("Deleted old data for %s (full rebuild)", doc_id)
        else:
            existing_hashes = store.get_existing_doc_hashes()
            log.info("Existing documents in Qdrant: %d", len(existing_hashes))
            to_add, to_modify, to_delete, to_skip = compute_diff(
                local_files, existing_hashes, image_records_by_doc_id
            )

        log.info(
            "Diff: add=%d, modify=%d, delete=%d, skip=%d",
            len(to_add), len(to_modify), len(to_delete), len(to_skip),
        )

        stats.skipped = len(to_skip)

        # --- Delete removed documents ---
        for doc_id in to_delete:
            store.delete_by_doc_id(doc_id)
            stats.deleted += 1
            log.info("Deleted: %s", doc_id)

        # --- Fit or restore BM25 encoder ---
        bm25 = _prepare_bm25(
            cfg, store, local_files, to_add, to_modify, to_skip,
            image_records_by_doc_id,
        )

        # --- Add new documents ---
        for i, doc_id in enumerate(to_add, 1):
            try:
                n = _ingest_one(
                    doc_id, local_files[doc_id], source_dir, cfg, store,
                    embedder, bm25, image_records_by_doc_id.get(doc_id, []),
                )
                stats.added += 1
                stats.total_chunks += n
                log.info("[%d/%d] Added: %s (%d chunks)", i, len(to_add), doc_id, n)
            except Exception as exc:
                msg = f"Error adding {doc_id}: {exc}"
                log.error(msg)
                stats.errors.append(msg)

        # --- Modify changed documents (atomic: write new, then delete old) ---
        for i, doc_id in enumerate(to_modify, 1):
            try:
                old_hash = existing_hashes.get(doc_id, "")
                n = _ingest_one(
                    doc_id, local_files[doc_id], source_dir, cfg, store,
                    embedder, bm25, image_records_by_doc_id.get(doc_id, []),
                )
                # Remove old version
                if old_hash:
                    store.delete_by_doc_id_and_hash(doc_id, old_hash)
                stats.modified += 1
                stats.total_chunks += n
                log.info("[%d/%d] Modified: %s (%d chunks)", i, len(to_modify), doc_id, n)
            except Exception as exc:
                msg = f"Error modifying {doc_id}: {exc}"
                log.error(msg)
                stats.errors.append(msg)

        # --- Persist BM25 state for query-side reuse ---
        if bm25 is not None and (to_add or to_modify):
            store.save_bm25_state(bm25.to_json())

    finally:
        embedder.close()
        store.close()

    return stats


def _prepare_bm25(
    cfg: Config,
    store: QdrantStore,
    local_files: dict[str, Path],
    to_add: list[str],
    to_modify: list[str],
    to_skip: list[str],
    image_records_by_doc_id: dict[str, list[dict]] | None = None,
) -> Optional[BM25Encoder]:
    """Either fit a new BM25 encoder on the corpus or restore a saved one."""
    if not cfg.enable_bm25:
        return None

    # If changes are small relative to corpus, try to reuse existing state
    changed = len(to_add) + len(to_modify)
    total = changed + len(to_skip)
    reuse_existing = changed < max(10, total * 0.1)  # <10% churn → reuse

    if reuse_existing:
        saved = store.load_bm25_state()
        if saved:
            enc = BM25Encoder.from_json(saved)
            log.info(
                "BM25 state restored from Qdrant (N=%d, avgdl=%.1f)",
                enc.N, enc.avgdl,
            )
            return enc
        log.info("No saved BM25 state; will fit from full corpus")

    # Fit from entire corpus (all local files)
    log.info("Fitting BM25 on %d documents...", len(local_files))
    enc = BM25Encoder()

    from .parser import parse_file
    from .chunker import chunk_document

    texts: list[str] = []
    image_records_by_doc_id = image_records_by_doc_id or {}
    for doc_id, fpath in local_files.items():
        try:
            doc = parse_file(fpath, fpath.parent)
            chunks = chunk_document(
                doc, cfg, images=image_records_by_doc_id.get(doc_id, [])
            )
            for c in chunks:
                if c.chunk_type in ("child", "image"):
                    texts.append(c.content)
        except Exception as exc:
            log.warning("BM25 fit: skipping %s: %s", doc_id, exc)

    enc.fit(texts)
    return enc


def _ingest_one(
    doc_id: str,
    filepath: Path,
    base_dir: Path,
    cfg: Config,
    store: QdrantStore,
    embedder: Embedder,
    bm25: Optional[BM25Encoder],
    image_records: list[dict] | None = None,
) -> int:
    """Parse, chunk, embed, and upsert a single document. Returns chunk count."""
    doc = parse_file(filepath, base_dir)

    if not doc.title:
        log.warning("Document %s has no title, using filename", doc_id)
        doc.title = doc_id

    image_records = image_records or []
    chunks = chunk_document(doc, cfg, images=image_records)
    if not chunks:
        log.warning("Document %s produced 0 chunks, skipping", doc_id)
        return 0

    # Separate embeddable chunks from parents (no embedding)
    children = [c for c in chunks if c.chunk_type in ("child", "image")]
    parents = [c for c in chunks if c.chunk_type == "parent"]

    # Embed child chunks in batches
    child_embeddings: list[Optional[EmbeddingResult]] = []
    batch_size = cfg.batch_size
    for i in range(0, len(children), batch_size):
        batch = children[i : i + batch_size]
        texts = [c.content_for_embedding for c in batch]
        results = embedder.embed_batch(texts)
        # Attach BM25 sparse if enabled
        if bm25 is not None:
            for j, chunk in enumerate(batch):
                sparse = bm25.encode(chunk.content)
                if sparse["indices"]:
                    results[j].sparse_indices = sparse["indices"]
                    results[j].sparse_values = sparse["values"]
        child_embeddings.extend(results)

    # Build metadata dict shared across all points of this document
    doc_meta = {
        "title": doc.title,
        "page_id": doc.page_id,
        "path": doc.path,
        "source_url": doc.source_url,
        "last_updated": doc.last_updated,
        "scope": doc.scope,
        "file_path": doc.file_path,
        "ingested_at": _now_iso(),
    }

    file_hash = compute_file_hash(filepath, image_records)

    # Upsert parents (without vectors)
    parent_embeddings: list[Optional[EmbeddingResult]] = [None] * len(parents)
    if parents:
        store.upsert_chunks(parents, parent_embeddings, file_hash, doc_meta)

    # Upsert children (with vectors)
    if children:
        store.upsert_chunks(children, child_embeddings, file_hash, doc_meta)

    return len(children) + len(parents)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
