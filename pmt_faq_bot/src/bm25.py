"""Client-side BM25 sparse encoder for Chinese + English mixed text.

Design:
 - Tokenize with jieba (CJK) and regex (ASCII alphanumerics).
 - Map each token deterministically to a bucket index via MD5 hashing.
 - Build corpus-wide IDF during ingest, reuse at query time.
 - Persist state (N, avgdl, idf) to Qdrant as a special payload point
   so that query-side encoders can restore the same statistics.

Output sparse vector: {"indices": [int], "values": [float]} — matches the
Qdrant SparseVector protocol exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from math import log as math_log
from typing import Optional

import jieba

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
_ASCII_RE = re.compile(r"[a-z0-9_]+")
# Combined scanner: match a CJK run OR an ascii token
_TOKEN_SCANNER = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf]+|[a-zA-Z0-9_]+"
)

# Minimal English stopword list for noise reduction in short queries.
_EN_STOPWORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "of", "in", "on", "at",
    "for", "with", "by", "from", "as", "this", "that", "these", "those",
    "it", "its", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "and", "or", "not", "no", "but", "if", "then", "so",
])

# Simple Chinese function-word stopwords
_CN_STOPWORDS = frozenset([
    "的", "了", "是", "在", "和", "与", "及", "或", "也", "都", "就",
    "还", "又", "但", "而", "这", "那", "我", "你", "他", "她", "它",
    "们", "个", "要", "能", "会", "可", "可以", "如何", "什么", "怎么",
    "怎样", "哪里", "哪些", "谁", "多少", "为什么", "介绍", "下",
])


class BM25Encoder:
    """Stateful BM25 encoder: fit corpus once, then encode queries/docs."""

    def __init__(
        self,
        vocab_size: int = 1 << 20,  # 1 048 576 buckets; collision rate negligible for <~100k vocab
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.vocab_size = vocab_size
        self.k1 = k1
        self.b = b
        self.avgdl: float = 0.0
        self.N: int = 0
        self.idf: dict[int, float] = {}
        # pre-initialize jieba (idempotent; loads dict once)
        jieba.initialize()

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Return a list of normalized tokens (lowercase, stopwords removed)."""
        tokens: list[str] = []
        for match in _TOKEN_SCANNER.finditer(text):
            segment = match.group()
            if _CJK_RE.match(segment):
                # CJK: use jieba's search-mode segmentation (good for retrieval)
                for tok in jieba.lcut_for_search(segment):
                    tok = tok.strip()
                    if not tok or tok in _CN_STOPWORDS:
                        continue
                    tokens.append(tok)
            else:
                tok = segment.lower()
                if tok in _EN_STOPWORDS or len(tok) <= 1:
                    continue
                tokens.append(tok)
        return tokens

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.vocab_size

    # ------------------------------------------------------------------
    # Fit (compute IDF + avgdl from corpus)
    # ------------------------------------------------------------------

    def fit(self, documents: list[str]) -> None:
        """Scan the corpus to build IDF statistics."""
        doc_freq: dict[int, int] = {}
        total_len = 0
        self.N = 0

        for doc in documents:
            toks = self.tokenize(doc)
            if not toks:
                continue
            self.N += 1
            total_len += len(toks)
            for term_id in {self._hash(t) for t in toks}:
                doc_freq[term_id] = doc_freq.get(term_id, 0) + 1

        self.avgdl = total_len / self.N if self.N else 1.0
        self.idf = {
            term_id: math_log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for term_id, df in doc_freq.items()
        }
        logger.info(
            "BM25 fitted: N=%d docs, avgdl=%.1f tokens, vocab=%d unique terms",
            self.N, self.avgdl, len(self.idf),
        )

    # ------------------------------------------------------------------
    # Encode (produce a sparse vector)
    # ------------------------------------------------------------------

    def encode(self, text: str) -> dict:
        """Encode a single text. Returns {'indices': [...], 'values': [...]}."""
        toks = self.tokenize(text)
        if not toks:
            return {"indices": [], "values": []}

        tf = Counter(self._hash(t) for t in toks)
        dl = len(toks)
        denom_norm = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1.0))

        indices: list[int] = []
        values: list[float] = []
        for term_id, freq in tf.items():
            idf = self.idf.get(term_id, 0.0)
            if idf <= 0:
                continue
            value = idf * (freq * (self.k1 + 1)) / (freq + denom_norm)
            indices.append(term_id)
            values.append(float(value))
        return {"indices": indices, "values": values}

    # ------------------------------------------------------------------
    # Persistence (to restore at query time)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "vocab_size": self.vocab_size,
            "k1": self.k1,
            "b": self.b,
            "N": self.N,
            "avgdl": self.avgdl,
            "idf": {str(k): v for k, v in self.idf.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Encoder":
        enc = cls(
            vocab_size=data.get("vocab_size", 1 << 20),
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )
        enc.N = int(data.get("N", 0))
        enc.avgdl = float(data.get("avgdl", 1.0))
        enc.idf = {int(k): float(v) for k, v in data.get("idf", {}).items()}
        return enc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "BM25Encoder":
        return cls.from_dict(json.loads(text))
