"""End-to-end retrieval test: Dense recall + Jina Rerank.

Usage (on the server, from a container on llm-net):
    docker run --rm --network llm-net \
      -e QDRANT_URL=http://qdrant:6333 \
      -e EMBED_URL=http://tei-embedding:80 \
      -e RERANK_URL=http://tei-rerank:80 \
      -e QDRANT_API_KEY=<key> \
      -v /opt/PMT-FAQ-Bot/pmt_faq_bot/tests/test_retrieval.py:/app/test_retrieval.py \
      -v /opt/PMT-FAQ-Bot/pmt_faq_bot:/out \
      --entrypoint python pmt_faq_bot:latest test_retrieval.py --output /out/retrieval_report.txt

The --output flag writes the report directly to a UTF-8 file inside the
container, which avoids any host-side (PowerShell / cmd) re-encoding of the
redirected stdout stream.
"""

import argparse
import io
import os
import sys
from typing import Optional

import httpx

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8001")
RERANK_URL = os.getenv("RERANK_URL", "http://localhost:8002")
LLM_URL = os.getenv("LLM_URL", "http://vllm-chat:8000")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3-235B-A22B-Instruct-2507")
API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("COLLECTION_NAME", "PMT-FAQ")

RECALL_K = 20   # Dense recall pool size
FINAL_K = 5     # After rerank
INCLUDE_LOW_VALUE = False  # Set True to include References/Others chunks
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() in ("1", "true", "yes")


# Each test case: (category, query, list of doc_id substrings expected to appear in top-K)
# Use "NO_MATCH" to assert the system should NOT confidently return any doc
# (accepts any result but flags if rerank score > 0.3)
TEST_CASES: list[tuple[str, str, list[str]]] = [
    # --- A. Conceptual (semantic understanding) ---
    ("concept-en", "What is WSUS and what does it do?",
     ["disc-wsus-service"]),

    ("concept-cn", "什么是 server hardening？",
     ["os-hardening-implementation", "hardening"]),

    ("concept-cn", "Nutanix 的快照机制是怎么工作的？",
     ["nutanix-ahv-data-protection"]),

    ("concept-en", "What data types does Redis support?",
     ["redis"]),

    ("concept-mix", "介绍下 DISC foundational services",
     ["disc-foundational-services"]),

    # --- B. Fact / parameter lookup (keyword-heavy) ---
    ("fact-en", "Which server is the Veeam VBR Server?",
     ["backup-recovery"]),

    ("fact-cn", "Oracle 数据库的归档日志多久备份一次？",
     ["backup-recovery"]),

    ("fact-cn", "CCTV 监控视频保留多少天？",
     ["cctv"]),

    ("fact-en", "When is the Nutanix snapshot scheduled?",
     ["nutanix-ahv-data-protection"]),

    ("fact-cn", "RHEL 安装的时区应该设置成什么？",
     ["rhel-os-implementation"]),

    ("fact-en", "What is innodb_buffer_pool_size default value?",
     ["mysql"]),

    ("fact-cn", "漏洞扫描计划的频率是多少？",
     ["vulnerability-management"]),

    # --- C. How-to / procedural ---
    ("howto-cn", "如何在 Windows Server 上手动部署硬化策略？",
     ["os-hardening-implementation"]),

    ("howto-en", "How to report a security incident in Bosch?",
     ["security-incident-management"]),

    ("howto-cn", "Linux 服务器如何加入 AD 域？",
     ["active-directory-integration"]),

    ("howto-en", "How to restore a VM using Veeam?",
     ["data-restore-for-vm"]),

    ("howto-cn", "如何申请 CoDC 机房的物理访问权限？",
     ["physical-access-management-for-codc"]),

    ("howto-en", "How to upgrade Cisco Nexus switch IOS?",
     ["network-operations-runbook"]),

    ("howto-cn", "ME 环境下如何集成 RB-PAM？",
     ["rb-pam-integration"]),

    ("howto-en", "How to handle Veeam backup alerts?",
     ["veeam-alert-handling"]),

    # --- D. Exact keyword / entity lookup ---
    ("entity-en", "What is SZHRMEBCKVM001?",
     ["backup-recovery"]),

    ("entity-en", "Where is DISC-WIN-Base-Hardening used?",
     ["os-hardening-implementation", "patch-management"]),

    ("entity-cn", "MongoDB 分片集群部署文档在哪里？",
     ["mongo-db"]),

    # --- E. Broad / cross-document ---
    ("broad-cn", "Windows Server 的月度打补丁流程",
     ["windows-server-patch-management", "patch-management"]),

    ("broad-en", "How do we back up Oracle RAC databases?",
     ["backup-recovery", "create-backup-job-for-oracle"]),

    ("broad-cn", "网络设备的配置备份怎么做？",
     ["network-operations-runbook"]),

    # --- F. No-match (should return low confidence / low rerank scores) ---
    ("nomatch", "How to configure Kubernetes Ingress controller?",
     ["NO_MATCH"]),

    ("nomatch", "What is AWS S3 pricing model?",
     ["NO_MATCH"]),

    ("nomatch", "如何搭建一个 Qdrant 向量数据库？",
     ["NO_MATCH"]),

    # --- G. Edge cases ---
    ("edge-cn", "ISA-CN 团队负责什么服务？",
     ["dedicated-infrastructure-services", "disc"]),

    ("edge-cn", "RHEL 服务器的代理如何设置？",
     ["rhel-os-implementation"]),
]


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def embed_query_dense(query: str) -> list[float]:
    r = httpx.post(
        f"{EMBED_URL}/embed",
        json={"inputs": query, "truncate": True},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()[0]


_BM25_ENCODER = None


def _get_bm25_encoder():
    """Load the BM25 encoder state from Qdrant (stored during ingest)."""
    global _BM25_ENCODER
    if _BM25_ENCODER is not None:
        return _BM25_ENCODER

    try:
        from src.bm25 import BM25Encoder
    except ImportError:
        return None

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["api-key"] = API_KEY
    try:
        r = httpx.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            headers=headers,
            json={
                "filter": {"must": [{"key": "chunk_type", "match": {"value": "__bm25_state__"}}]},
                "limit": 1,
                "with_payload": True,
            },
            timeout=30,
        )
        r.raise_for_status()
        points = r.json()["result"]["points"]
        if not points:
            return None
        state_json = points[0]["payload"].get("bm25_state")
        if not state_json:
            return None
        _BM25_ENCODER = BM25Encoder.from_json(state_json)
        return _BM25_ENCODER
    except Exception as exc:
        print(f"[warn] Failed to load BM25 encoder: {exc}", file=sys.stderr)
        return None


def embed_query_sparse(query: str) -> dict | None:
    """Compute BM25 sparse vector on the client side."""
    enc = _get_bm25_encoder()
    if enc is None:
        return None
    sparse = enc.encode(query)
    if not sparse["indices"]:
        return None
    return sparse


def _qdrant_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["api-key"] = API_KEY
    return headers


def _base_filter(exclude_low_value: bool) -> dict:
    must = [{"key": "chunk_type", "match": {"any": ["child", "image"]}}]
    must_not = []
    if exclude_low_value:
        must_not.append({"key": "is_low_value", "match": {"value": True}})
    # Implicit: chunk_type=child already excludes __bm25_state__
    return {"must": must, "must_not": must_not}


def dense_recall(qvec: list[float], k: int, exclude_low_value: bool) -> list[dict]:
    payload = {
        "query": qvec,
        "using": "dense",
        "filter": _base_filter(exclude_low_value),
        "limit": k,
        "with_payload": True,
    }
    r = httpx.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        headers=_qdrant_headers(),
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["result"]["points"]


def hybrid_recall(
    qvec_dense: list[float],
    qvec_sparse: Optional[dict],
    k: int,
    exclude_low_value: bool,
) -> list[dict]:
    """Dense + Sparse prefetch, fused with RRF by Qdrant server-side."""
    prefetch = [
        {
            "query": qvec_dense,
            "using": "dense",
            "filter": _base_filter(exclude_low_value),
            "limit": k * 2,
        }
    ]
    if qvec_sparse is not None:
        prefetch.append({
            "query": qvec_sparse,
            "using": "sparse",
            "filter": _base_filter(exclude_low_value),
            "limit": k * 2,
        })

    payload = {
        "prefetch": prefetch,
        "query": {"fusion": "rrf"},
        "limit": k,
        "with_payload": True,
    }
    r = httpx.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        headers=_qdrant_headers(),
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["result"]["points"]


def multi_query_recall(
    queries: list[str],
    k: int,
    exclude_low_value: bool,
    rrf_k: int = 60,
) -> list[dict]:
    """Run hybrid recall for multiple query rewrites and merge via client-side RRF.

    For each query in `queries`, we do a hybrid (dense+sparse) recall, then
    fuse the ranked lists using Reciprocal Rank Fusion to produce a unified
    candidate pool. Queries first in the list get full weight; use query[0]
    as the anchor (original user query).
    """
    scores: dict[str, float] = {}
    point_by_id: dict[str, dict] = {}
    queries_hit: dict[str, set] = {}  # point_id -> set of query indices that surfaced it

    for q_idx, q in enumerate(queries):
        try:
            dense_vec = embed_query_dense(q)
            sparse_vec = embed_query_sparse(q)
            results = hybrid_recall(dense_vec, sparse_vec, k, exclude_low_value)
        except Exception as exc:
            print(f"[warn] recall failed for rewrite #{q_idx}: {exc}", file=sys.stderr)
            continue

        # Original query (q_idx=0) gets a small weight boost as anchor
        weight = 1.2 if q_idx == 0 else 1.0
        for rank, point in enumerate(results):
            pid = str(point["id"])
            scores[pid] = scores.get(pid, 0.0) + weight / (rrf_k + rank + 1)
            point_by_id[pid] = point
            queries_hit.setdefault(pid, set()).add(q_idx)

    # Sort by fused RRF score
    sorted_ids = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
    out: list[dict] = []
    for pid in sorted_ids[:k]:
        pt = dict(point_by_id[pid])
        pt["score"] = scores[pid]
        pt["_queries_matched"] = len(queries_hit[pid])
        out.append(pt)
    return out


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Standard rerank: Jina reranker reorders candidates by query-doc relevance."""
    if not candidates:
        return []
    texts = [c["payload"].get("content", "") for c in candidates]
    r = httpx.post(
        f"{RERANK_URL}/rerank",
        json={"query": query, "texts": texts, "truncate": True},
        timeout=30,
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def match_doc_id(doc_id: str, expected_substrings: list[str]) -> bool:
    if not doc_id:
        return False
    did = doc_id.lower()
    return any(sub.lower() in did for sub in expected_substrings)


_REWRITER = None


def _get_rewriter():
    global _REWRITER
    if _REWRITER is not None:
        return _REWRITER
    if not ENABLE_QUERY_REWRITE:
        return None
    try:
        from src.query_rewriter import QueryRewriter
        _REWRITER = QueryRewriter(llm_url=LLM_URL, model=LLM_MODEL)
        return _REWRITER
    except Exception as exc:
        print(f"[warn] QueryRewriter init failed: {exc}", file=sys.stderr)
        return None


def evaluate_case(query: str, expected: list[str]) -> dict:
    qvec_dense = embed_query_dense(query)
    qvec_sparse = embed_query_sparse(query)

    # Dense-only recall (baseline)
    dense_candidates = dense_recall(
        qvec_dense, RECALL_K, exclude_low_value=not INCLUDE_LOW_VALUE
    )

    # Hybrid recall on the original query alone (baseline for comparison)
    if qvec_sparse is not None:
        hybrid_only_candidates = hybrid_recall(
            qvec_dense, qvec_sparse, RECALL_K,
            exclude_low_value=not INCLUDE_LOW_VALUE,
        )
    else:
        hybrid_only_candidates = dense_candidates

    # Full production pipeline: LLM query rewriting → multi-query hybrid recall
    rewriter = _get_rewriter()
    rewrites: list[str] = [query]
    if rewriter is not None:
        rewrites = rewriter.rewrite(query)

    if len(rewrites) > 1:
        candidates = multi_query_recall(
            rewrites, RECALL_K,
            exclude_low_value=not INCLUDE_LOW_VALUE,
        )
    else:
        candidates = hybrid_only_candidates

    reranked = rerank(query, candidates, FINAL_K)

    is_nomatch = expected == ["NO_MATCH"]
    sparse_enabled = qvec_sparse is not None
    used_rewrites = len(rewrites) if len(rewrites) > 1 else 1

    def _hits(pool: list[dict], k: int) -> bool:
        return any(
            match_doc_id(p["payload"].get("doc_id", ""), expected)
            for p in pool[:k]
        )

    dense_hit_at_1 = False
    dense_hit_at_5 = False
    hybrid_hit_at_1 = False
    hybrid_hit_at_5 = False
    multi_hit_at_1 = False
    multi_hit_at_5 = False
    rerank_hit_at_1 = False
    rerank_hit_at_5 = False

    if not is_nomatch:
        if dense_candidates:
            dense_hit_at_1 = _hits(dense_candidates, 1)
            dense_hit_at_5 = _hits(dense_candidates, 5)
        if hybrid_only_candidates:
            hybrid_hit_at_1 = _hits(hybrid_only_candidates, 1)
            hybrid_hit_at_5 = _hits(hybrid_only_candidates, 5)
        if candidates:
            multi_hit_at_1 = _hits(candidates, 1)
            multi_hit_at_5 = _hits(candidates, 5)
        if reranked:
            rerank_hit_at_1 = _hits(reranked, 1)
            rerank_hit_at_5 = _hits(reranked, 5)

    top_rerank_score = reranked[0]["rerank_score"] if reranked else 0.0
    nomatch_ok = is_nomatch and top_rerank_score < 0.3

    return {
        "is_nomatch": is_nomatch,
        "sparse_enabled": sparse_enabled,
        "used_rewrites": used_rewrites,
        "rewrites": rewrites,
        "dense_hit_at_1": dense_hit_at_1,
        "dense_hit_at_5": dense_hit_at_5,
        "hybrid_hit_at_1": hybrid_hit_at_1,
        "hybrid_hit_at_5": hybrid_hit_at_5,
        "multi_hit_at_1": multi_hit_at_1,
        "multi_hit_at_5": multi_hit_at_5,
        "rerank_hit_at_1": rerank_hit_at_1,
        "rerank_hit_at_5": rerank_hit_at_5,
        "top_rerank_score": top_rerank_score,
        "nomatch_ok": nomatch_ok,
        "dense_candidates": dense_candidates,
        "hybrid_only_candidates": hybrid_only_candidates,
        "candidates": candidates,
        "reranked": reranked,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def print_hit(rank: int, score: float, p: dict, key: str, extra: dict = None) -> None:
    doc_id = p.get("doc_id", "")[:45]
    section = p.get("section_title", "")[:25]
    content = (p.get("content", "") or "")[:140].replace("\n", " ")
    low = " [LOW]" if p.get("is_low_value") else ""
    extras = ""
    if extra and extra.get("_queries_matched"):
        extras += f" [qs:{extra['_queries_matched']}]"
    print(f"  #{rank} {key}={score:.4f}{low}{extras}  {doc_id!r}  sec={section!r}")
    print(f"        {content}...")


def run_case(category: str, query: str, expected: list[str]) -> dict:
    header = f"[{category}] {query}"
    print(f"\n{'='*90}\n{header}")
    print(f"Expected: {expected}\n{'='*90}")

    result = evaluate_case(query, expected)

    if result["used_rewrites"] > 1:
        print(f"\n[Query Rewrites ({result['used_rewrites']})]")
        for i, rw in enumerate(result["rewrites"]):
            tag = " (original)" if i == 0 else ""
            print(f"  {i}. {rw}{tag}")

    print(f"\n[Multi-Query Recall top 3]")
    for i, hit in enumerate(result["candidates"][:3], 1):
        print_hit(i, hit["score"], hit["payload"], key="score", extra=hit)

    print(f"\n[After Rerank top {FINAL_K}]")
    for i, hit in enumerate(result["reranked"], 1):
        print_hit(i, hit["rerank_score"], hit["payload"], key="rerank", extra=hit)

    verdict = _verdict(result)
    print(f"\nVERDICT: {verdict}")
    return {"category": category, "query": query, "result": result, "verdict": verdict}


def _verdict(r: dict) -> str:
    if r["is_nomatch"]:
        return "OK (no-match, low rerank)" if r["nomatch_ok"] else "FAIL (no-match but rerank score too high)"
    if r["rerank_hit_at_1"]:
        return "PASS@1 (rerank)"
    if r["rerank_hit_at_5"]:
        return "PASS@5 (rerank)"
    if r["dense_hit_at_5"]:
        return "WEAK (dense@5 hit, rerank missed)"
    return "FAIL"


def _force_utf8_stdio() -> None:
    """Ensure stdout/stderr can emit non-ASCII text (CJK, etc.).

    Inside minimal Docker images (e.g. python:3.11-slim) the default stdio
    encoding may be ASCII when there is no TTY and no UTF-8 locale is set,
    which silently replaces every CJK character with ``?``. Reconfiguring
    here guarantees UTF-8 regardless of the runtime environment.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
                continue
            except (ValueError, OSError):
                pass
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            setattr(
                sys,
                stream_name,
                io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True),
            )


class _Tee:
    """Write to multiple text streams at once (e.g. stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            n = s.write(data)
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval smoke tests.")
    parser.add_argument(
        "--output",
        "-o",
        default=os.getenv("RETRIEVAL_REPORT_PATH"),
        help="Write the report to this UTF-8 file in addition to stdout. "
             "Also reads RETRIEVAL_REPORT_PATH env var.",
    )
    return parser.parse_args()


def main() -> None:
    _force_utf8_stdio()
    args = _parse_args()

    log_fh = None
    if args.output:
        log_fh = open(args.output, "w", encoding="utf-8", errors="replace", newline="\n")
        sys.stdout = _Tee(sys.stdout, log_fh)

    print(f"Qdrant:     {QDRANT_URL}")
    print(f"Embedding:  {EMBED_URL}")
    print(f"Rerank:     {RERANK_URL}")
    print(f"Collection: {COLLECTION}")
    print(f"Exclude low-value sections: {not INCLUDE_LOW_VALUE}")
    print(f"Test cases: {len(TEST_CASES)}")

    results = []
    for category, query, expected in TEST_CASES:
        try:
            results.append(run_case(category, query, expected))
        except Exception as exc:
            print(f"ERROR for {query!r}: {exc}", file=sys.stderr)

    # --- Summary ---
    print(f"\n\n{'#'*90}\nSUMMARY\n{'#'*90}\n")

    total = len(results)
    answerable_results = [r for r in results if not r["result"]["is_nomatch"]]
    answerable = len(answerable_results)
    nomatch_total = total - answerable

    sparse_enabled = any(r["result"]["sparse_enabled"] for r in results)
    any_rewrites = any(r["result"]["used_rewrites"] > 1 for r in results)
    nomatch_ok = sum(1 for r in results if r["result"]["is_nomatch"] and r["result"]["nomatch_ok"])

    def _pct(n: int) -> str:
        return f"{n}/{answerable}  ({100*n/answerable if answerable else 0:.1f}%)"

    dense_at_1 = sum(1 for r in answerable_results if r["result"]["dense_hit_at_1"])
    dense_at_5 = sum(1 for r in answerable_results if r["result"]["dense_hit_at_5"])
    hybrid_at_1 = sum(1 for r in answerable_results if r["result"]["hybrid_hit_at_1"])
    hybrid_at_5 = sum(1 for r in answerable_results if r["result"]["hybrid_hit_at_5"])
    multi_at_1 = sum(1 for r in answerable_results if r["result"]["multi_hit_at_1"])
    multi_at_5 = sum(1 for r in answerable_results if r["result"]["multi_hit_at_5"])
    rerank_at_1 = sum(1 for r in answerable_results if r["result"]["rerank_hit_at_1"])
    rerank_at_5 = sum(1 for r in answerable_results if r["result"]["rerank_hit_at_5"])

    print(f"Total cases: {total}  (answerable: {answerable}, no-match: {nomatch_total})")
    print(f"Sparse (BGE-M3) enabled: {sparse_enabled}")
    print(f"Query rewriting enabled: {any_rewrites}")
    print()
    print(f"Retrieval progression (answerable only):")
    print(f"  Dense only    @1: {_pct(dense_at_1)}   @5: {_pct(dense_at_5)}")
    print(f"  + BM25 Hybrid @1: {_pct(hybrid_at_1)}   @5: {_pct(hybrid_at_5)}")
    print(f"  + LLM Rewrite @1: {_pct(multi_at_1)}   @5: {_pct(multi_at_5)}")
    print(f"  + Rerank      @1: {_pct(rerank_at_1)}   @5: {_pct(rerank_at_5)}")
    print(f"No-match correct: {nomatch_ok}/{nomatch_total}")

    # Failures
    failures = [r for r in results if "FAIL" in r["verdict"] or "WEAK" in r["verdict"]]
    if failures:
        print(f"\nFailures / Weak ({len(failures)}):")
        for f in failures:
            print(f"  [{f['category']}] {f['verdict']}: {f['query']}")

    if log_fh is not None:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        log_fh.close()


if __name__ == "__main__":
    main()
