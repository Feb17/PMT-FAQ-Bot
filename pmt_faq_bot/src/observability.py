"""Observability: metrics collection, timing, and JSON serialization.

Provides structured dataclasses for retrieval latency, ingest run metrics,
and Qdrant collection health snapshots. Uses a context-manager Timer for
per-stage latency measurement and JSONL file I/O for persistent reports.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage-level latency
# ---------------------------------------------------------------------------


@dataclass
class StageLatency:
    """Per-stage latency in milliseconds for a single retrieval operation."""
    stage: str = ""          # e.g. "dense_embed", "sparse_embed", "qdrant_recall", "rerank"
    ms: float = 0.0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Retrieval metrics (per-query)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation."""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    query: str = ""
    query_length: int = 0
    candidates_count: int = 0
    reranked_count: int = 0
    top_rerank_score: float = 0.0
    stages: list[StageLatency] = field(default_factory=list)
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Ingest run metrics
# ---------------------------------------------------------------------------


@dataclass
class IngestMetrics:
    """Metrics for a full ingest run."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    mode: str = "auto"
    total_files: int = 0
    added: int = 0
    modified: int = 0
    deleted: int = 0
    skipped: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    # Per-phase timings (seconds)
    diff_seconds: float = 0.0
    deletion_seconds: float = 0.0
    bm25_fit_seconds: float = 0.0
    addition_seconds: float = 0.0
    modification_seconds: float = 0.0
    embedding_seconds: float = 0.0

    # Collection health post-ingest
    points_count: int = 0
    vectors_count: Optional[int] = None
    collection_status: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Qdrant collection health snapshot
# ---------------------------------------------------------------------------


@dataclass
class CollectionHealth:
    """Snapshot of Qdrant collection health."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    collection_name: str = ""
    points_count: int = 0
    vectors_count: Optional[int] = None
    segments_count: int = 0
    status: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """Context manager for measuring stage latency.

    Usage::

        with Timer("qdrant_recall") as t:
            results = hybrid_recall(...)
        metrics.stages.append(t.to_latency())
    """

    __slots__ = ("stage_name", "start", "elapsed_ms")

    def __init__(self, stage_name: str) -> None:
        self.stage_name = stage_name
        self.start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        return False  # do not suppress exceptions

    def to_latency(self) -> StageLatency:
        return StageLatency(stage=self.stage_name, ms=self.elapsed_ms)

    def to_latency_error(self, error: str) -> StageLatency:
        return StageLatency(
            stage=self.stage_name, ms=self.elapsed_ms, success=False, error=error
        )


# ---------------------------------------------------------------------------
# JSONL persistence helpers
# ---------------------------------------------------------------------------


def save_metrics_jsonl(metrics: Any, filepath: Path | str) -> None:
    """Append one metrics record as a JSON line to *filepath*."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    d = metrics.to_dict() if hasattr(metrics, "to_dict") else asdict(metrics)  # type: ignore[arg-type]
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.debug("Metrics appended to %s", filepath)


def load_metrics_jsonl(filepath: Path | str) -> list[dict]:
    """Load JSONL file as a list of dicts. Returns [] if file missing."""
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    out: list[dict] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_json(filepath: Path | str, data: dict | list) -> None:
    """Write *data* as a single pretty-printed JSON object (not JSONL)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Metrics snapshot written to %s", filepath)


def load_json(filepath: Path | str) -> dict:
    """Load a single JSON object file. Returns {} if missing."""
    filepath = Path(filepath)
    if not filepath.exists():
        return {}
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)
