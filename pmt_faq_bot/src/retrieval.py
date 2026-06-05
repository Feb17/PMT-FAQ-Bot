"""Retrieval pipeline: Hybrid (Dense + BM25 Sparse) recall + Rerank.

Reusable by both the RAG API server and the test harness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .bm25 import BM25Encoder
from .config import Config
from .observability import RetrievalMetrics, StageLatency, Timer

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    chunk_type: str
    doc_id: str
    title: str
    section_title: str
    content: str
    source_url: str
    score: float
    images: list[dict] = field(default_factory=list)
    image: Optional[dict] = None


class RetrievalPipeline:
    """Encapsulates embed → hybrid recall → rerank."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._embed_url = cfg.embedding_url.rstrip("/")
        self._rerank_url = cfg.rerank_url.rstrip("/")
        self._qdrant_url = cfg.qdrant_url.rstrip("/")
        self._collection = cfg.collection_name
        self._api_key = cfg.qdrant_api_key
        self._client = httpx.Client(timeout=cfg.embed_timeout)
        self._bm25: Optional[BM25Encoder] = None
        self._bm25_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        recall_k: int = 20,
        exclude_low_value: bool = True,
        collect_metrics: bool = False,
    ) -> list[RetrievedChunk] | tuple[list[RetrievedChunk], RetrievalMetrics]:
        """Full pipeline: embed → hybrid recall → rerank → top_k chunks."""
        metrics: RetrievalMetrics | None = (
            RetrievalMetrics(query=query, query_length=len(query)) if collect_metrics else None
        )
        dense_vec: list[float] = []
        sparse_vec: Optional[dict] = None
        candidates: list[dict] = []
        reranked: list[dict] = []

        with Timer("dense_embed") as t_dense:
            try:
                dense_vec = self._embed_dense(query)
            except Exception as exc:
                if metrics is not None:
                    metrics.stages.append(t_dense.to_latency_error(str(exc)))
                raise
        if metrics is not None and not metrics.stages:
            metrics.stages.append(t_dense.to_latency())

        sparse_stage_name = "sparse_embed_skip" if not self._cfg.enable_bm25 else "sparse_embed"
        with Timer(sparse_stage_name) as t_sparse:
            try:
                sparse_vec = self._embed_sparse(query)
            except Exception as exc:
                if metrics is not None:
                    metrics.stages.append(t_sparse.to_latency_error(str(exc)))
                raise
        if metrics is not None and len(metrics.stages) < 2:
            metrics.stages.append(StageLatency(stage=sparse_stage_name, ms=t_sparse.elapsed_ms))

        with Timer("qdrant_recall") as t_recall:
            try:
                candidates = self._hybrid_recall(
                    dense_vec, sparse_vec, recall_k, exclude_low_value
                )
            except Exception as exc:
                if metrics is not None:
                    metrics.stages.append(t_recall.to_latency_error(str(exc)))
                raise
        if metrics is not None and len(metrics.stages) < 3:
            metrics.stages.append(t_recall.to_latency())

        with Timer("rerank") as t_rerank:
            try:
                reranked = self._rerank(query, candidates, top_k)
            except Exception as exc:
                if metrics is not None:
                    metrics.stages.append(t_rerank.to_latency_error(str(exc)))
                raise
        if metrics is not None and len(metrics.stages) < 4:
            metrics.stages.append(t_rerank.to_latency())

        results = [
            RetrievedChunk(
                chunk_id=hit["payload"].get("chunk_id", ""),
                chunk_type=hit["payload"].get("chunk_type", ""),
                doc_id=hit["payload"].get("doc_id", ""),
                title=hit["payload"].get("title", ""),
                section_title=hit["payload"].get("section_title", ""),
                content=hit["payload"].get("content", ""),
                source_url=hit["payload"].get("source_url", ""),
                score=hit.get("rerank_score", 0.0),
                images=hit["payload"].get("images") or [],
                image=hit["payload"].get("image"),
            )
            for hit in reranked
        ]

        if not collect_metrics:
            return results

        assert metrics is not None
        metrics.candidates_count = len(candidates)
        metrics.reranked_count = len(reranked)
        metrics.top_rerank_score = max((hit.get("rerank_score", 0.0) for hit in reranked), default=0.0)
        metrics.total_ms = sum(stage.ms for stage in metrics.stages)
        logger.debug(
            "Retrieval latencies: dense=%.1fms sparse=%.1fms qdrant=%.1fms rerank=%.1fms total=%.1fms",
            next((s.ms for s in metrics.stages if s.stage == "dense_embed"), 0.0),
            next((s.ms for s in metrics.stages if s.stage in ("sparse_embed", "sparse_embed_skip")), 0.0),
            next((s.ms for s in metrics.stages if s.stage == "qdrant_recall"), 0.0),
            next((s.ms for s in metrics.stages if s.stage == "rerank"), 0.0),
            metrics.total_ms,
        )
        return results, metrics

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Dense embedding
    # ------------------------------------------------------------------

    def _embed_dense(self, text: str) -> list[float]:
        r = self._client.post(
            f"{self._embed_url}/embed",
            json={"inputs": text, "truncate": True},
        )
        r.raise_for_status()
        vec = r.json()[0]
        dim = self._cfg.embed_dim
        if dim > 0 and len(vec) > dim:
            from .embedder import _truncate_and_normalize
            vec = _truncate_and_normalize(vec, dim)
        return vec

    # ------------------------------------------------------------------
    # BM25 sparse embedding
    # ------------------------------------------------------------------

    def _embed_sparse(self, text: str) -> Optional[dict]:
        if not self._cfg.enable_bm25:
            return None
        enc = self._get_bm25()
        if enc is None:
            return None
        sparse = enc.encode(text)
        return sparse if sparse.get("indices") else None

    def _get_bm25(self) -> Optional[BM25Encoder]:
        if self._bm25_loaded:
            return self._bm25
        self._bm25_loaded = True
        try:
            headers = self._qdrant_headers()
            r = self._client.post(
                f"{self._qdrant_url}/collections/{self._collection}/points/scroll",
                headers=headers,
                json={
                    "filter": {"must": [{"key": "chunk_type", "match": {"value": "__bm25_state__"}}]},
                    "limit": 1,
                    "with_payload": True,
                },
            )
            r.raise_for_status()
            points = r.json()["result"]["points"]
            if points:
                state_json = points[0]["payload"].get("bm25_state")
                if state_json:
                    self._bm25 = BM25Encoder.from_json(state_json)
                    logger.info("BM25 encoder loaded (N=%d)", self._bm25.N)
        except Exception as exc:
            logger.warning("Failed to load BM25 state: %s", exc)
        return self._bm25

    # ------------------------------------------------------------------
    # Qdrant hybrid recall (server-side RRF)
    # ------------------------------------------------------------------

    def _hybrid_recall(
        self,
        dense_vec: list[float],
        sparse_vec: Optional[dict],
        k: int,
        exclude_low_value: bool,
    ) -> list[dict]:
        filt = self._base_filter(exclude_low_value)
        prefetch = [
            {"query": dense_vec, "using": "dense", "filter": filt, "limit": k * 2}
        ]
        if sparse_vec is not None:
            prefetch.append(
                {"query": sparse_vec, "using": "sparse", "filter": filt, "limit": k * 2}
            )

        payload = {
            "prefetch": prefetch,
            "query": {"fusion": "rrf"},
            "limit": k,
            "with_payload": True,
        }
        r = self._client.post(
            f"{self._qdrant_url}/collections/{self._collection}/points/query",
            headers=self._qdrant_headers(),
            json=payload,
        )
        r.raise_for_status()
        return r.json()["result"]["points"]

    # ------------------------------------------------------------------
    # Rerank via TEI
    # ------------------------------------------------------------------

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        texts = [c["payload"].get("content", "") for c in candidates]
        r = self._client.post(
            f"{self._rerank_url}/rerank",
            json={"query": query, "texts": texts, "truncate": True},
        )
        r.raise_for_status()
        ranked = r.json()
        ranked.sort(key=lambda x: x["score"], reverse=True)
        out = []
        for entry in ranked[:top_k]:
            hit = dict(candidates[entry["index"]])
            hit["rerank_score"] = entry["score"]
            out.append(hit)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _qdrant_headers(self) -> dict:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        return headers

    @staticmethod
    def _base_filter(exclude_low_value: bool) -> dict:
        must = [{"key": "chunk_type", "match": {"any": ["child", "image"]}}]
        must_not: list[dict] = []
        if exclude_low_value:
            must_not.append({"key": "is_low_value", "match": {"value": True}})
        return {"must": must, "must_not": must_not}
