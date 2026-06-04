"""TEI Embedding API client — generates dense and sparse vectors via BGE-M3."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse_indices: Optional[list[int]] = None
    sparse_values: Optional[list[float]] = None


class Embedder:
    """Thin client for the HuggingFace TEI ``/embed`` and ``/embed_sparse`` endpoints."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._base = cfg.embedding_url.rstrip("/")
        self._client = httpx.Client(timeout=cfg.embed_timeout)
        self._sparse_supported: Optional[bool] = None
        self._vector_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_dim(self) -> int:
        """Return the dense vector dimension that will be used.

        If ``cfg.embed_dim`` is set (MRL truncation), return that directly.
        Otherwise probe TEI with a dummy request.
        """
        if self._vector_dim is not None:
            return self._vector_dim
        if self._cfg.embed_dim > 0:
            self._vector_dim = self._cfg.embed_dim
            log.info("Using configured MRL dimension: %d", self._vector_dim)
        else:
            vecs = self._call_dense(["dimension probe"])
            self._vector_dim = len(vecs[0])
            log.info("Detected native embedding dimension: %d", self._vector_dim)
        return self._vector_dim

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts, returning dense (and optionally sparse) vectors."""
        if not texts:
            return []

        dense_vecs = self._call_dense(texts)

        sparse_results: Optional[list[dict]] = None
        if self._supports_sparse():
            sparse_results = self._call_sparse(texts)

        results: list[EmbeddingResult] = []
        for i, dvec in enumerate(dense_vecs):
            sr = EmbeddingResult(dense=dvec)
            if sparse_results and i < len(sparse_results):
                sr.sparse_indices = sparse_results[i].get("indices")
                sr.sparse_values = sparse_results[i].get("values")
            results.append(sr)

        return results

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Dense embedding
    # ------------------------------------------------------------------

    def _call_dense(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"inputs": texts, "truncate": True}
        data = self._post("/embed", payload)
        if not (isinstance(data, list) and data and isinstance(data[0], list)):
            raise ValueError(f"Unexpected dense response shape: {type(data)}")

        # Client-side MRL truncation: take first N dims, then L2-normalize.
        # (TEI does not currently support `truncate_dim` parameter.)
        target_dim = self._cfg.embed_dim
        if target_dim > 0 and len(data[0]) > target_dim:
            return [_truncate_and_normalize(v, target_dim) for v in data]
        return data

    # ------------------------------------------------------------------
    # Sparse embedding (optional)
    # ------------------------------------------------------------------

    def _supports_sparse(self) -> bool:
        if self._sparse_supported is not None:
            return self._sparse_supported

        try:
            resp = self._client.get(f"{self._base}/info")
            if resp.status_code == 200:
                info = resp.json()
                # TEI exposes model type; BGE-M3 reports sparse capability
                model_type = info.get("model_type", {})
                if isinstance(model_type, dict) and "Embedding" in model_type:
                    sub = model_type["Embedding"]
                    if isinstance(sub, dict) and sub.get("sparse"):
                        self._sparse_supported = True
                        log.info("TEI sparse embedding supported")
                        return True
        except Exception:
            pass

        # Try a probe request
        try:
            probe = self._client.post(
                f"{self._base}/embed_sparse",
                json={"inputs": ["test"], "truncate": True},
            )
            if probe.status_code == 200:
                self._sparse_supported = True
                log.info("TEI sparse embedding available (probe succeeded)")
                return True
        except Exception:
            pass

        self._sparse_supported = False
        log.info("TEI sparse embedding NOT available — will use dense only")
        return False

    def _call_sparse(self, texts: list[str]) -> list[dict]:
        payload = {"inputs": texts, "truncate": True}
        data = self._post("/embed_sparse", payload)
        # TEI returns list[list[{index, value}]]
        results: list[dict] = []
        for entry in data:
            if isinstance(entry, list):
                indices = [item["index"] for item in entry]
                values = [item["value"] for item in entry]
                results.append({"indices": indices, "values": values})
            else:
                results.append({})
        return results

    # ------------------------------------------------------------------
    # HTTP helper with retry
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> list:
        url = f"{self._base}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(1, self._cfg.embed_max_retries + 1):
            try:
                resp = self._client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_err = exc
                wait = 2 ** attempt
                log.warning(
                    "Embedding request failed (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    self._cfg.embed_max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Embedding failed after {self._cfg.embed_max_retries} attempts: {last_err}"
        )


def _truncate_and_normalize(vec: list[float], target_dim: int) -> list[float]:
    """MRL truncation: take first N dimensions and L2-normalize.

    Qwen3-Embedding / BGE-M3 are MRL-trained: prefix of the vector is itself
    a valid embedding after renormalization.
    """
    truncated = vec[:target_dim]
    norm = math.sqrt(sum(x * x for x in truncated))
    if norm == 0:
        return truncated
    return [x / norm for x in truncated]
