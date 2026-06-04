"""Qdrant collection management and read/write operations."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from .chunker import Chunk
from .config import Config
from .embedder import EmbeddingResult

log = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._client = QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key or None,
            timeout=60,
        )
        self._collection = cfg.collection_name
        self._vector_dim: int = 0

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def ensure_collection(self, vector_dim: int) -> None:
        """Create collection if it does not exist, with named dense + sparse vectors."""
        self._vector_dim = vector_dim
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection in collections:
            log.info("Collection '%s' already exists", self._collection)
            self._ensure_indexes()
            return

        log.info(
            "Creating collection '%s' (dense dim=%d)", self._collection, vector_dim
        )
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": VectorParams(size=vector_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        for field_name, schema_type in [
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("doc_content_hash", PayloadSchemaType.KEYWORD),
            ("is_low_value", PayloadSchemaType.BOOL),
        ]:
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass  # index already exists

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[Optional[EmbeddingResult]],
        doc_content_hash: str,
        doc_metadata: dict,
    ) -> int:
        """Upsert a list of chunks with their embeddings. Returns points written."""
        points: list[PointStruct] = []

        for chunk, emb in zip(chunks, embeddings):
            vectors: dict = {}
            if emb is not None:
                vectors["dense"] = emb.dense
                if emb.sparse_indices is not None and emb.sparse_values is not None:
                    vectors["sparse"] = SparseVector(
                        indices=emb.sparse_indices,
                        values=emb.sparse_values,
                    )
            else:
                vectors["dense"] = [0.0] * self._vector_dim

            payload = {
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type,
                "parent_chunk_id": chunk.parent_chunk_id,
                "doc_id": chunk.doc_id,
                "section_title": chunk.section_title,
                "content": chunk.content,
                "is_low_value": chunk.is_low_value,
                "images": chunk.images,
                "image": chunk.image,
                "doc_content_hash": doc_content_hash,
                **doc_metadata,
            }

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            points.append(
                PointStruct(id=point_id, vector=vectors, payload=payload)
            )

        # Batch upsert
        batch_size = self._cfg.batch_size
        written = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=self._collection, points=batch)
            written += len(batch)

        return written

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points belonging to a document."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    def delete_by_doc_id_and_hash(self, doc_id: str, doc_content_hash: str) -> None:
        """Delete points with a specific doc_id AND content hash (old version)."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    FieldCondition(
                        key="doc_content_hash",
                        match=MatchValue(value=doc_content_hash),
                    ),
                ]
            ),
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_existing_doc_hashes(self) -> dict[str, str]:
        """Scroll through collection and return {doc_id: doc_content_hash}.

        Uses scroll with a filter on chunk_type=child to avoid counting parents
        multiple times. We only need one point per doc to get the hash.
        """
        doc_hashes: dict[str, str] = {}
        offset = None

        while True:
            results, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_type", match=MatchValue(value="child")
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=["doc_id", "doc_content_hash"],
                with_vectors=False,
            )

            for point in results:
                did = point.payload.get("doc_id", "")
                dhash = point.payload.get("doc_content_hash", "")
                if did and did not in doc_hashes:
                    doc_hashes[did] = dhash

            if next_offset is None:
                break
            offset = next_offset

        return doc_hashes

    def collection_info(self) -> dict:
        """Return basic collection statistics."""
        try:
            info = self._client.get_collection(self._collection)
            return {
                "collection": self._collection,
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", None),
                "status": str(info.status),
            }
        except Exception as exc:
            return {"collection": self._collection, "error": str(exc)}

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # BM25 state persistence (stored as a single special point)
    # ------------------------------------------------------------------

    _BM25_STATE_POINT_ID = "00000000-0000-0000-0000-000000000bm2"  # reserved UUID

    def save_bm25_state(self, state_json: str) -> None:
        """Store BM25 corpus statistics so query side can restore identical encoder."""
        import uuid

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "__bm25_state__"))
        from qdrant_client.models import PointStruct

        payload = {
            "chunk_type": "__bm25_state__",
            "bm25_state": state_json,
        }
        vectors: dict = {"dense": [0.0] * self._vector_dim}
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=vectors, payload=payload)],
        )
        log.info("BM25 state persisted to Qdrant (%d bytes)", len(state_json))

    def load_bm25_state(self) -> Optional[str]:
        """Retrieve previously saved BM25 state JSON, if any."""
        import uuid
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            results, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_type",
                            match=MatchValue(value="__bm25_state__"),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if results:
                return results[0].payload.get("bm25_state")
        except Exception as exc:
            log.warning("Failed to load BM25 state: %s", exc)
        return None
