from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://qdrant:6333")
    )
    qdrant_api_key: str = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY", "")
    )
    embedding_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_URL", "http://tei-embedding:80")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "PMT-FAQ")
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "64"))
    )
    embed_timeout: float = field(
        default_factory=lambda: float(os.getenv("EMBED_TIMEOUT", "120"))
    )
    embed_max_retries: int = field(
        default_factory=lambda: int(os.getenv("EMBED_MAX_RETRIES", "3"))
    )
    embed_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBED_DIM", "0"))
    )  # 0 = use model native dim; set to e.g. 1024 for MRL truncation

    enable_bm25: bool = field(
        default_factory=lambda: os.getenv("ENABLE_BM25", "true").lower()
        in ("1", "true", "yes")
    )

    # Rerank / retrieval
    rerank_url: str = field(
        default_factory=lambda: os.getenv("RERANK_URL", "http://tei-rerank:80")
    )
    llm_url: str = field(
        default_factory=lambda: os.getenv("LLM_URL", "http://vllm-chat:8000")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL", "Qwen3-235B-A22B-Instruct-2507"
        )
    )
    rag_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
    )
    rag_model_name: str = field(
        default_factory=lambda: os.getenv("RAG_MODEL_NAME", "pmt_faq_bot")
    )
    history_max_turns: int = field(
        default_factory=lambda: int(os.getenv("HISTORY_MAX_TURNS", "4"))
    )
    rag_asset_base_url: str = field(
        default_factory=lambda: os.getenv("RAG_ASSET_BASE_URL", "")
    )
    rag_asset_dir: str = field(
        default_factory=lambda: os.getenv("RAG_ASSET_DIR", "")
    )
    max_images_per_answer: int = field(
        default_factory=lambda: int(os.getenv("MAX_IMAGES_PER_ANSWER", "3"))
    )

    # Chunking parameters
    child_chunk_target_tokens: int = 384
    child_chunk_max_tokens: int = 512
    child_chunk_overlap_tokens: int = 50
    parent_chunk_max_tokens: int = 2048
    large_table_split_rows: int = 18

    CHARS_PER_TOKEN: float = 3.5  # rough estimate for mixed CJK + English

    def chars_for_tokens(self, tokens: int) -> int:
        return int(tokens * self.CHARS_PER_TOKEN)
