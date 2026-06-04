# Handoff: Confluence RAG → Qdrant + Open WebUI

**Generated:** 2026-05-19  
**Workspace (local):** `H20-GPU/models/`  
**Server:** `xe9680:/models/`  
**Prior transcript:** `agent-transcripts/a6a14622-f332-4a57-b37b-58a080ab78c4.jsonl`

---

## Goal

End-to-end RAG for ~1005 Confluence-exported Markdown docs: reliable ingest into Qdrant, high-recall hybrid retrieval, OpenAI-compatible RAG API for Open WebUI, all on air-gapped XE9680 with existing vLLM / TEI / Qdrant stack.

---

## Current State

### Done

| Area | Status |
|------|--------|
| Ingest pipeline | Parser → hierarchical chunker → dense (TEI) + client BM25 → Qdrant; incremental SHA256 |
| Full ingest | 1005 files → 10853 chunks in `isacn_knowledge` (see server ingest log in transcript) |
| Retrieval module | `models/rag-ingest/src/retrieval.py` — hybrid RRF + Jina rerank |
| RAG API | FastAPI OpenAI-compat: `/v1/models`, `/v1/chat/completions` (SSE), `/healthz` |
| Multi-turn | `rewrite_standalone()` in `query_rewriter.py`; used by `server/chat.py` |
| Docker | Single image `rag-ingest:latest`; profiles: default=`rag-api`, `ingest` profile for one-shot job |
| Orchestration | **`models/Makefile`** wraps three independent compose projects (root `compose.yml` **abandoned** due to `llm-net` conflicts) |
| Network | All stacks use **`llm-net` external: true** (`name: llm-net`) |

### Retrieval metrics (31-case harness)

Source: `models/rag-ingest/retrieval_report.txt` (answerable n=28).

| Stage | Hit@1 | Hit@5 |
|-------|-------|-------|
| Dense only | 57.1% | 75.0% |
| + BM25 hybrid | 42.9% | **85.7%** |
| + LLM multi-query rewrite | 53.6% | 82.1% |
| + Rerank | 67.9% | 78.6% |

**Decision:** LLM query expansion hurts recall on this corpus (semantic drift); keep code but **do not enable in test harness by default**. Production RAG API uses **standalone** rewrite for multi-turn only, not multi-query expansion.

### Known failures (7 cases)

Listed at end of `retrieval_report.txt` — e.g. DISC foundational services, Veeam VBR server, K8s Ingress no-match gate, ISA-CN team scope.

---

## Paths & Layout

```
H20-GPU/models/
├── Makefile                 # unified up/down/status/ingest
├── llm-services/
│   └── docker-compose.yml   # vllm-chat, tei-embedding, tei-rerank, open-webui
├── qdrant/
│   └── compose.yml
└── rag-ingest/              # moved here from repo root; sync to xe9680:/models/rag-ingest/
    ├── compose.yml
    ├── .env
    ├── Dockerfile
    ├── src/                 # ingest + api
    └── tests/test_retrieval.py
```

Server documents: `/models/confluence-dify-clean-export/documents`

---

## Runtime Topology

| Service | Host port | Container / notes |
|---------|-----------|-------------------|
| vllm-chat | 8000 | Qwen3-235B |
| tei-embedding | 8001 | BGE-M3 (user switched from Qwen3-Embedding-4B) |
| tei-rerank | 8002 | Jina Reranker v3 |
| qdrant | 6333 | collection `isacn_knowledge` |
| rag-api | **8090→8088** | host 8088 occupied; `.env` has `RAG_API_PORT=8090` |
| open-webui | 3000 | sees rag-api at `http://rag-api:8088/v1` (Docker DNS, not host port) |

**Prerequisite:** `docker network create llm-net` (one-time) if missing.

**Makefile quick refs:** `make up`, `make status`, `make ingest-auto`, `make rebuild-api`.

---

## Open Items / Likely Next Work

1. **Sync & verify rag-api on server**  
   Push latest `models/rag-ingest/compose.yml` + `.env` (`RAG_API_PORT=8090`, Python healthcheck).  
   ```bash
   cd /models/rag-ingest && docker compose up -d --force-recreate rag-api
   curl -s http://127.0.0.1:8090/healthz
   cd /models && make status
   ```

2. **Full stack bring-up after network cleanup**  
   User had `llm-services_llm-net` creation errors when subnet overlapped existing `llm-net`. Fix is **external `llm-net`** in all compose files (already in workspace). Do **not** delete `llm-net` if other colleagues' containers attach; use external reference only.

3. **Open WebUI E2E smoke** (not confirmed complete on server)  
   - Select model `isacn-rag`  
   - Single-turn Q&A with citations  
   - Multi-turn follow-up (standalone rewrite)  
   - No-match query → `NO_RESULT_REPLY` path  

4. **Retrieval quality** (optional)  
   Triage 7 failures in `retrieval_report.txt`; avoid test-script gaming per user requirement.  
   Consider: disable multi-query rewrite in API; tune `RAG_SCORE_THRESHOLD`; chunk/parser fixes for weak cases.

5. **Secrets**  
   `models/rag-ingest/.env` contains `QDRANT_API_KEY` — treat as sensitive; do not commit to shared repos.

6. **Root compose**  
   Server had experimental `/models/compose.yml` with `networks.llm-net conflicts with imported resource`. **Use Makefile only** unless redesigning network ownership.

---

## Key Implementation Notes

- **Sparse vectors:** TEI BGE-M3 does not expose sparse; **client-side BM25** (`src/bm25.py`, jieba + EN regex). State persisted in Qdrant via `save_bm25_state` / `load_bm25_state`.
- **MRL:** `EMBED_DIM=0` = native dim; non-zero triggers client truncate + L2 norm in `embedder.py`.
- **Parent chunks:** zero dense vector; children carry embeddings + optional sparse.
- **Low-value chunks:** `is_low_value` payload + index; filtered at retrieval when `INCLUDE_LOW_VALUE=False`.
- **SSE:** `StreamingResponse` in `server/app.py` (dropped `sse-starlette` v3 `sep=""` issue).
- **Healthcheck:** Python `urllib` inside container (slim image has no `curl`).
- **Offline wheels:** Linux `manylinux` wheels required for Docker build on air-gapped host.

---

## Important Files (do not duplicate here)

| Artifact | Purpose |
|----------|---------|
| `models/rag-ingest/src/retrieval.py` | Production retrieval pipeline |
| `models/rag-ingest/src/server/chat.py` | RAG chat + streaming + citations |
| `models/rag-ingest/tests/test_retrieval.py` | Evaluation harness (31 cases) |
| `models/rag-ingest/retrieval_report.txt` | Latest eval output |
| `models/Makefile` | Stack orchestration |
| Transcript `a6a14622-...jsonl` | Full design decisions, error fixes, user quotes |

---

## Suggested Skills for Next Session

| Skill | When |
|-------|------|
| `diagnose` / `systematic-debugging` | rag-api unhealthy, compose network errors, retrieval regressions |
| `verification-before-completion` | Before claiming stack or smoke tests pass |
| `babysit` | If opening a PR or fixing CI after merge |
| `bosch-procedure-documentation` | If extending Confluence doc handling |
| `create-jira-subtasks` / `jira-description-updater` | If tracking remaining work in Jira |
| `handoff` | If chaining to another agent again |

---

## User Constraints (verbatim intent)

- Tests must reflect **real end-user scenarios**; do not tweak eval labels/scripts to inflate scores.
- vLLM on server is available for LLM steps when needed.
- `rag-ingest` lives under **`H20-GPU/models/rag-ingest`** (sync to `/models/rag-ingest` on xe9680).
- Ignore colleague's `llm-nets` network (Ollama); use shared `llm-net` only.

---

## Commands Cheat Sheet (server)

```bash
# Network (once)
docker network create llm-net 2>/dev/null || true

# Full resident stack
cd /models && make up && make status

# Rebuild after code change
cd /models/rag-ingest && docker build -t rag-ingest:latest .
cd /models && make rebuild-api

# Incremental ingest
cd /models && make ingest-auto

# Retrieval eval (inside container or with env pointing to services)
# See tests/test_retrieval.py in repo
```
