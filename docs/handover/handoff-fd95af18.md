# Handoff: H20-GPU Vector DB / BDPIX-1739

**Date:** 2026-05-07  
**Workspace:** `c:\Users\agu2szh\OneDrive - Bosch Group\Documents\H20-GPU`  
**Primary Jira:** [BDPIX-1739](https://rb-tracker.bosch.com/tracker19/browse/BDPIX-1739) — *Task3.39_Build up a Vector Database of ISA-CN Operation Knowledge Base*

---

## Executive summary

This session covered: (1) creating and refining Jira sub-tasks under BDPIX-1739 from actual `rag-ingest` work; (2) building a reusable Cursor skill `create-jira-subtasks`; (3) syncing Jira status/comments after a demoable end-to-end RAG stack; (4) attaching a formal retrieval regression report and closing BDPIX-1795 + parent BDPIX-1739 as **Done**; (5) an Ask-mode analysis concluding **PoC/pilot-ready, not full production** without further eval (answer-side metrics, latency SLO, larger test set, no-match hardening).

---

## What was accomplished

### Jira (via `user-mcp-atlassian` MCP)

| Action | Details |
|--------|---------|
| Created 9 subtasks | BDPIX-1792 … BDPIX-1800 under BDPIX-1739 (5 current-work + 4 follow-ups) |
| Aligned attributes | `duedate=2026-05-29`, `components=[BRP_2026]`, `labels=[Q2]` only (no extra labels) |
| Progress comments | Apr 28 progress update; May 7 demo milestone on parent |
| BDPIX-1795 | Description updates + `retrieval_report.txt` attached ([attachment 1109225](https://rb-tracker.bosch.com/tracker19/secure/attachment/1109225/retrieval_report.txt)) |
| Closed | **BDPIX-1795** → Done; **BDPIX-1739** → Done (PoC scope complete; follow-ups remain open) |

### Cursor skill (local)

- **`~/.cursor/skills/create-jira-subtasks/`** — SKILL.md + README.md; Phase 1 refactor done (two-pass create→align, preview-first, reference run BDPIX-1739→1792..1800).
- Compared against user's [jira-description-updater](https://github.com/Feb17/jira-description-updater) for optimization ideas (README/SKILL split, REST fallback deferred to Phase 3).

### Workspace / stack (unchanged since Apr 28 in repo; demo on server)

Code lives under **`models/`** (not repo-root `rag-ingest/`):

```
models/
├── Makefile              # up / status / ingest-auto / ingest-full / rebuild-api
├── llm-services/         # vllm-chat, tei-embedding, tei-rerank, open-webui
├── qdrant/               # Qdrant + API key, snapshots volume
└── rag-ingest/           # rag-ingest job + rag-api (FastAPI /v1/chat/completions)
```

- Architecture doc: `docs/architecture/rag-query-flow.{drawio,png,svg,arch.json,spec.yaml}`
- Retrieval harness: `models/rag-ingest/tests/test_retrieval.py`
- Report artifact: `models/rag-ingest/retrieval_report.txt` (~80 KB, 31 cases, run 2026-05-07)

---

## Current Jira state (as of handoff)

| Key | Status | Notes |
|-----|--------|--------|
| **BDPIX-1739** | **Done** | Parent closed after PoC + documented follow-ups |
| BDPIX-1792–1794, 1796 | Done | AC1–AC4 PoC build/deploy/scope |
| **BDPIX-1795** | **Done** | Retrieval report delivered |
| BDPIX-1797 | To Do | Scheduled ingest + alerting |
| BDPIX-1798 | To Do | More knowledge sources |
| BDPIX-1799 | In Progress | Qdrant backup/DR/capacity (snapshots dir exists; playbook/drill pending) |
| BDPIX-1800 | In Progress | Observability + retrieval regression suite |

---

## Key technical conclusions (from retrieval report — do not re-read full file here)

Source: `models/rag-ingest/retrieval_report.txt` (summary only).

- **31 cases** (28 answerable, 3 no-match); BGE-M3 sparse + query rewrite enabled.
- **Rerank @1: 67.9%** (main production-facing retrieval metric).
- **Hybrid @5: 85.7%** > **Rerank @5: 78.6%** → rerank can demote correct docs from top-5; tune thresholds/top_k.
- **No-match: 2/3** correct (K8s Ingress case failed — high rerank score → hallucination risk).
- **7 FAIL/WEAK** cases documented on BDPIX-1795 description; tuning tracked toward BDPIX-1800.

**Production readiness (agreed in session, not a ticket):** Pilot/internal demo OK; full production needs answer-side eval (RAGAS/DeepEval), latency SLO, n=100+ eval set, no-match hardening, DR drill (1799), scheduled ingest (1797).

---

## Commands the next agent may need

**Retrieval regression (from `models/rag-ingest/`, uses `.env` for Qdrant key):**

```bash
cd models/rag-ingest && mkdir -p logs
docker run --rm --network llm-net --env-file .env \
  -e EMBED_URL=http://tei-embedding:80 \
  -e RERANK_URL=http://tei-rerank:80 \
  -e LLM_URL=http://vllm-chat:8000 \
  -v "$(pwd)/tests/test_retrieval.py:/app/test_retrieval.py:ro" \
  -v "$(pwd)/logs:/out" \
  --entrypoint python rag-ingest:latest \
  test_retrieval.py --output /out/retrieval_report.txt
```

**Stack health:** `cd models && make status`

**Suggested Makefile addition (not implemented):** `make test-retrieval` wrapping the above.

---

## MCP / environment notes

- **`user-mcp-atlassian`** required for Jira; was **unavailable** in one mid-session turn (only cursor-app-control, browser, context7). User re-enabled it later — verify MCP list before Jira work.
- Jira comments: prefer Jira wiki (`h3.` / `h4.` / `* bullets`) over raw `**markdown**` to avoid escaped asterisks in Bosch Jira.
- Subtask transitions: `transition_id=2` (Done), `4` (Start Progress) — confirm per project if scripts fail.

---

## User preferences captured this session

- Sub-tasks must **stay on parent subject** (“build vector database”) — **no** Portal API / Dify integration subtasks under BDPIX-1739.
- Sub-task granularity: **milestones**, not per-file chores (`parser.py`, `chunker.py`, …).
- **No extra Jira labels** beyond parent (`Q2` only).
- Jira updates in **中文** acceptable for descriptions/comments when user asks.
- User closed parent **BDPIX-1739** while follow-ups 1797–1800 remain — intentional phased rollout.

---

## Recommended next session focus (pick one track)

1. **Production hardening (highest risk):** No-match / rerank threshold tuning in `rag-api` config; add RAGAS or DeepEval eval subtask + run.
2. **BDPIX-1800:** Expand `test_retrieval.py` to nDCG/MRR, CI baseline, 100+ cases; optional `make test-retrieval`.
3. **BDPIX-1797:** Cron/systemd or compose scheduled `make ingest-auto` + alert on ingest JSON errors.
4. **BDPIX-1799:** Qdrant snapshot schedule + restore drill runbook.
5. **Skill Phase 3:** `scripts/jira_subtask_batch.py` REST fallback (mirror jira-description-updater pattern) if MCP unreliable.

---

## Suggested skills for next agent

| Skill | When |
|-------|------|
| **`create-jira-subtasks`** | New subtasks under 1797–1800 or new parent; follow two-pass + preview rules |
| **`jira-description-updater`** | If updating ticket descriptions to DoD style ([repo](https://github.com/Feb17/jira-description-updater)) |
| **`diagnose`** | Rerank @5 regression, no-match false positives |
| **`improve-codebase-architecture`** | If refactoring eval harness / Makefile |
| **`create-skill` / handoff skill** | Further skill iterations |
| **Ask mode** | Production readiness / benchmark methodology questions only |

---

## Artifacts to open first in a fresh session

1. `models/rag-ingest/retrieval_report.txt` — if continuing retrieval quality work  
2. `models/rag-ingest/tests/test_retrieval.py` — eval logic and TEST_CASES  
3. `~/.cursor/skills/create-jira-subtasks/SKILL.md` — if creating more Jira structure  
4. Jira: BDPIX-1797, 1799, 1800 (open children of closed parent)

---

## Out of scope for this handoff / already closed

- Do not re-close BDPIX-1739 or BDPIX-1795 without user request.  
- Parent DoD Confluence links remain in BDPIX-1739 description (PIX pages — not fetched this session).  
- `rag-api` + Open WebUI integration is **demo scope**; not tracked as BDPIX-1739 subtasks by user choice.
