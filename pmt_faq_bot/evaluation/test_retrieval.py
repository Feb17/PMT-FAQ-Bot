"""End-to-end retrieval regression test: Dense recall + Jina Rerank.

Dataset: Bosch ADC toolchain FAQ (CAAS / CVD / JFrog / Bitbucket / Jenkins / Arena / COS)
14 Chinese documents covering Tencent private cloud toolchain account/permission/usage guides.

Usage (on the server, from a container on llm-net):
    docker run --rm --network llm-net \\
      -e QDRANT_URL=http://qdrant:6333 \\
      -e EMBED_URL=http://tei-embedding:80 \\
      -e RERANK_URL=http://tei-rerank:80 \\
      -e QDRANT_API_KEY=<key> \\
      -v /opt/PMT-FAQ-Bot/pmt_faq_bot/evaluation/test_retrieval.py:/app/test_retrieval.py \\
      -v /opt/PMT-FAQ-Bot/pmt_faq_bot:/out \\
      --entrypoint python pmt_faq_bot:latest test_retrieval.py --output /out/retrieval_report.txt

Run directly on the server:
    python3 pmt_faq_bot/evaluation/test_retrieval.py --save-baseline baseline.json --report-json report.json

The --output flag writes the report directly to a UTF-8 file inside the
container, which avoids any host-side (PowerShell / cmd) re-encoding of the
redirected stdout stream.
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional

# Ensure pmt_faq_bot/ is on sys.path so that `from src.bm25 import ...` works
# when running directly on the host (not via Docker where WORKDIR=/app).
_syspath_root = Path(__file__).resolve().parents[1]
if str(_syspath_root) not in sys.path:
    sys.path.insert(0, str(_syspath_root))

import httpx

QDRANT_URL = os.getenv("QDRANT_URL", "http://10.203.97.4:6333")
EMBED_URL = os.getenv("EMBED_URL", "http://10.203.97.4:8001")
RERANK_URL = os.getenv("RERANK_URL", "http://10.203.97.4:8002")
LLM_URL = os.getenv("LLM_URL", "http://10.203.97.4:8000")
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
#
# Dataset: Bosch ADC toolchain FAQ (CAAS / CVD / JFrog / Bitbucket / Jenkins / Arena / COS)
# 14 documents (14 ingested markdown pages), all Chinese with English technical terms.
TEST_CASES: list[tuple[str, str, list[str]]] = [
    # --- A. Conceptual (semantic understanding) ---
    ("concept-cn", "什么是 CAAS 账号？",
     ["caas"]),

    ("concept-cn", "什么是 CVD 远程桌面？",
     ["cvd"]),

    ("concept-cn", "什么是 Arena 平台的 DMP 应龙？",
     ["arena", "dmp"]),

    ("concept-en", "What is CVD remote desktop used for?",
     ["cvd"]),

    ("concept-cn", "COS 桶是什么？",
     ["cos"]),

    ("concept-cn", "oneIDM 和工具链权限有什么关系？",
     ["faq-721033694", "caas"]),

    ("concept-cn", "什么是 Artifactory 的 Access Token？",
     ["jfrog", "721033694"]),

    ("concept-cn", "专区 ADC 是什么意思？",
     ["caas", "cvd"]),

    # --- B. Fact / parameter lookup (keyword-heavy) ---
    ("fact-cn", "CAAS 账号的用户名格式是什么？",
     ["caas"]),

    ("fact-cn", "Bitbucket 的访问 URL 是什么？",
     ["bitbucket"]),

    ("fact-en", "What is the Bitbucket access URL?",
     ["bitbucket"]),

    ("fact-cn", "申请 CVD 需要在 IDM 搜索什么角色名称？",
     ["cvd"]),

    ("fact-cn", "Jenkins 的 admin IDM role 是什么？",
     ["jenkins"]),

    ("fact-cn", "哪些项目可以写在 CAAS 申请理由里？",
     ["caas"]),

    ("fact-cn", "hosts 文件中 bitbucket.prod.boscharena.ai 对应的 IP 是多少？",
     ["bitbucketfaq", "faq-866374506"]),

    ("fact-cn", "CVD 不活跃多久会被系统回收？",
     ["cvd"]),

    ("fact-cn", "乘黄 FMP 是做什么的？",
     ["arena", "fmp"]),

    ("fact-cn", "CAAS 初始密码为什么不能用？",
     ["caas", "866359274"]),

    # --- C. How-to / procedural ---
    ("howto-cn", "如何申请 CAAS 账号？",
     ["caas"]),

    ("howto-en", "How to change CAAS password?",
     ["caas"]),

    ("howto-cn", "如何在办公电脑上直接访问 Bitbucket 而不通过 CVD？",
     ["bitbucketfaq"]),

    ("howto-cn", "如何申请 JFrog Artifactory 的制品仓权限？",
     ["jfrog"]),

    ("howto-cn", "如何在 Citrix Workspace 中修改 CAAS 密码？",
     ["cvd"]),

    ("howto-en", "How to install Citrix Workspace on Linux to access ADC zone?",
     ["cvd"]),

    ("howto-cn", "CVD 和办公电脑之间如何互传文件？",
     ["faq-866374506"]),

    ("howto-cn", "如何获取办公电脑的临时管理员权限？",
     ["faq-866374506", "721033694"]),

    ("howto-cn", "如何申请 COS 桶的读写权限？",
     ["cos"]),

    ("howto-cn", "如何在 Jenkins 里创建 Access Token？",
     ["faq-721033694"]),

    # --- D. Exact keyword / entity / troubleshooting lookup ---
    ("entity-cn", "Bitbucket 克隆报错 fatal the remote end hung up unexpectedly 怎么解决？",
     ["721033694"]),

    ("entity-cn", "CAAS 账号被冻结了怎么办？",
     ["721033694"]),

    ("entity-cn", "JFrog API key 不再使用，如何迁移到 Access Token？",
     ["jfrog", "721033694"]),

    ("entity-cn", "浏览器开了代理，无法访问 bitbucket 和 jfrog 怎么办？",
     ["faq-866374506", "721033694"]),

    ("entity-cn", "项目 PJW3 的 Bitbucket 代码仓管理员是谁？",
     ["bitbucket"]),

    ("entity-cn", "应龙 DMP 和玄武 Unisim 分别是什么？",
     ["arena"]),

    # --- E. Broad / cross-document ---
    ("broad-cn", "我要申请 Bitbucket 代码仓权限和 JFrog 制品仓权限，分别需要什么前提条件和步骤？",
     ["bitbucket", "jfrog"]),

    ("broad-cn", "在 ADC 专区工作，需要哪些账号和工具？",
     ["caas", "cvd", "faq-721033694"]),

    ("broad-cn", "hosts 文件里需要配置哪些域名才能直接访问 ADC 工具链？",
     ["faq-866374506", "bitbucketfaq"]),

    ("broad-cn", "Arena 平台有哪些子系统，各自的用途是什么？",
     ["arena"]),

    # --- F. No-match (should return low confidence / low rerank scores) ---
    ("nomatch", "How to deploy a Kubernetes cluster on AWS EKS?",
     ["NO_MATCH"]),

    ("nomatch", "Python 中如何使用 requests 库发送 POST 请求？",
     ["NO_MATCH"]),

    ("nomatch", "如何配置 TortoiseGit 连接 GitHub 仓库？",
     ["NO_MATCH"]),

    ("nomatch", "What is the price of Bosch XC3 ultrasonic sensor?",
     ["NO_MATCH"]),

    ("nomatch", "How do I train a PyTorch transformer model for image classification?",
     ["NO_MATCH"]),

    # --- G. Edge cases ---
    ("edge-cn", "CAAS",
     ["caas"]),

    ("edge-cn", "专区",
     ["caas", "cvd"]),

    ("edge-mix", "How to 申请 JFrog Artifactory 的 permission?",
     ["jfrog"]),

    ("edge-cn", "我想从零开始接入 ADC 专区做开发，需要申请 CAAS 账号、CVD 远程桌面、Bitbucket 代码仓、JFrog 制品仓和 Jenkins 流水线权限，整个流程分别需要哪些 IDM 角色、hosts 配置、以及各个平台的访问 URL？请详细说明步骤。",
     ["caas", "cvd", "bitbucket", "jfrog", "jenkins"]),

    ("edge-cn", "ADC",
     ["caas", "cvd"]),
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

def print_hit(rank: int, score: float, p: dict, key: str, extra: Optional[dict] = None) -> None:
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
    return {"category": category, "query": query, "expected": expected, "result": result, "verdict": verdict}


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


def compute_mrr(results: list[dict], k: int, stage: str) -> float:
    total = 0.0
    count = 0
    for r in results:
        if r["result"]["is_nomatch"]:
            continue
        count += 1
        pool = r["result"].get(stage, [])[:k]
        rr = 0.0
        for idx, hit in enumerate(pool, 1):
            if match_doc_id(hit.get("payload", {}).get("doc_id", ""), r["expected"]):
                rr = 1.0 / idx
                break
        total += rr
    return total / count if count else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _dump_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


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
    parser.add_argument("--save-baseline", help="Save full run results as JSON baseline.")
    parser.add_argument("--compare", help="Compare current run against a JSON baseline.")
    parser.add_argument("--report-json", help="Write structured JSON report.")
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

    expected_by_case = {(c, q): e for c, q, e in TEST_CASES}

    if args.compare:
        baseline = _load_json(args.compare)
        baseline_cases = {(c["category"], c["query"]): c for c in baseline.get("cases", [])}
        print(f"\n\n{'#'*90}\nBASELINE COMPARISON\n{'#'*90}")
        regressions = improvements = unchanged = missing = 0
        for r in results:
            prev = baseline_cases.get((r["category"], r["query"]))
            if prev is None:
                missing += 1
                continue
            curr_hit = bool(r["result"]["rerank_hit_at_1"])
            prev_hit = bool(prev.get("rerank_hit_at_1"))
            curr_score = float(r["result"]["top_rerank_score"])
            prev_score = float(prev.get("top_rerank_score", 0.0))
            curr_verdict = r["verdict"]
            prev_verdict = str(prev.get("verdict", ""))
            if curr_verdict == prev_verdict and curr_hit == prev_hit and abs(curr_score - prev_score) < 1e-9:
                unchanged += 1
            elif curr_hit and not prev_hit:
                improvements += 1
                print(f"+ {r['category']}: {r['query']} ({prev_verdict} -> {curr_verdict})")
            elif prev_hit and not curr_hit:
                regressions += 1
                print(f"- {r['category']}: {r['query']} ({prev_verdict} -> {curr_verdict})")
            else:
                unchanged += 1
        print(f"Regressions: {regressions}  Improvements: {improvements}  Unchanged: {unchanged}  Missing: {missing}")

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

    print(f"  Dense MRR@1: {compute_mrr(results, 1, 'dense_candidates'):.3f}   MRR@5: {compute_mrr(results, 5, 'dense_candidates'):.3f}")
    print(f"  Hybrid MRR@1: {compute_mrr(results, 1, 'hybrid_only_candidates'):.3f}   MRR@5: {compute_mrr(results, 5, 'hybrid_only_candidates'):.3f}")
    print(f"  Multi MRR@1: {compute_mrr(results, 1, 'candidates'):.3f}   MRR@5: {compute_mrr(results, 5, 'candidates'):.3f}")
    print(f"  Rerank MRR@1: {compute_mrr(results, 1, 'reranked'):.3f}   MRR@5: {compute_mrr(results, 5, 'reranked'):.3f}")

    print("\nPer-Category Breakdown")
    print("======================")
    print(f"{'Category':<14}{'N':>4}{'Pass@1(Rerank)':>18}{'MRR@5':>10}{'AvgRerankScore':>18}")
    for cat in sorted({r['category'] for r in results}):
        cr = [r for r in results if r["category"] == cat]
        n = len(cr)
        pass1 = (sum(1 for r in cr if r["result"]["rerank_hit_at_1"]) / n * 100) if n else 0.0
        mrr5 = compute_mrr(cr, 5, "reranked")
        avg_score = sum(r["result"]["top_rerank_score"] for r in cr) / n if n else 0.0
        print(f"{cat:<14}{n:>4}{pass1:>17.1f}%{mrr5:>10.3f}{avg_score:>18.3f}")

    scores = [r["result"]["top_rerank_score"] for r in results]
    print("\nRerank Score Distribution")
    print("=========================")
    print(f"Min: {min(scores) if scores else 0:.3f}  P25: {_percentile(scores, 25):.3f}  Median: {median(scores) if scores else 0:.3f}  P75: {_percentile(scores, 75):.3f}  Max: {max(scores) if scores else 0:.3f}")

    warning_pct = float(os.getenv("ALERT_PASS_RATE_WARNING_PCT", "0") or "0")
    if warning_pct:
        pass_rate = rerank_at_1 / answerable * 100 if answerable else 0.0
        if pass_rate < warning_pct:
            print(f"[warn] Rerank pass@1 {pass_rate:.1f}% below warning threshold {warning_pct:.1f}%")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "answerable": answerable,
            "nomatch_total": nomatch_total,
            "dense_at_1": dense_at_1,
            "dense_at_5": dense_at_5,
            "hybrid_at_1": hybrid_at_1,
            "hybrid_at_5": hybrid_at_5,
            "multi_at_1": multi_at_1,
            "multi_at_5": multi_at_5,
            "rerank_at_1": rerank_at_1,
            "rerank_at_5": rerank_at_5,
        },
        "cases": [
            {
                "category": r["category"],
                "query": r["query"],
                "expected": expected_by_case.get((r["category"], r["query"]), []),
                "verdict": r["verdict"],
                "dense_hit_at_1": r["result"]["dense_hit_at_1"],
                "dense_hit_at_5": r["result"]["dense_hit_at_5"],
                "hybrid_hit_at_1": r["result"]["hybrid_hit_at_1"],
                "hybrid_hit_at_5": r["result"]["hybrid_hit_at_5"],
                "multi_hit_at_1": r["result"]["multi_hit_at_1"],
                "multi_hit_at_5": r["result"]["multi_hit_at_5"],
                "rerank_hit_at_1": r["result"]["rerank_hit_at_1"],
                "rerank_hit_at_5": r["result"]["rerank_hit_at_5"],
                "top_rerank_score": r["result"]["top_rerank_score"],
            }
            for r in results
        ],
    }

    if args.save_baseline:
        _dump_json(args.save_baseline, report)
    if args.report_json:
        _dump_json(args.report_json, report)

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
