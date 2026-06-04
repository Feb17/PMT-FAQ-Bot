"""FastAPI app exposing OpenAI-compatible RAG endpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..config import Config
from .assets import resolve_asset_path
from .chat import ChatHandler

logger = logging.getLogger(__name__)

cfg = Config()
chat_handler = ChatHandler(cfg)

app = FastAPI(title="ISA-CN RAG API", version="1.0.0")


# ------------------------------------------------------------------
# GET /v1/models  — so Open WebUI discovers this as a "model"
# ------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": cfg.rag_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "isa-cn",
            }
        ],
    }


# ------------------------------------------------------------------
# POST /v1/chat/completions  — the main RAG endpoint
# ------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    try:
        result = await chat_handler.handle(body)
    except Exception as exc:
        logger.exception("Chat handler error")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "server_error"}},
        )

    if stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable proxy buffering
            },
        )
    return result


# ------------------------------------------------------------------
# GET /healthz  — readiness probe
# ------------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    checks: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in [
            ("qdrant", f"{cfg.qdrant_url}/readyz"),
            ("embedding", f"{cfg.embedding_url.rstrip('/')}/health"),
            ("rerank", f"{cfg.rerank_url.rstrip('/')}/health"),
            ("llm", f"{cfg.llm_url.rstrip('/')}/health"),
        ]:
            try:
                r = await client.get(url)
                checks[name] = "ok" if r.status_code < 400 else f"status={r.status_code}"
            except Exception as exc:
                checks[name] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(content={"status": "ok" if all_ok else "degraded", "checks": checks}, status_code=status_code)


@app.get("/assets/confluence/{asset_path:path}")
async def confluence_asset(asset_path: str):
    if not cfg.rag_asset_dir:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "RAG_ASSET_DIR is not configured"}},
        )
    resolved = resolve_asset_path(Path(cfg.rag_asset_dir), asset_path)
    if resolved is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "asset not found"}},
        )
    return FileResponse(resolved)


@app.on_event("shutdown")
def shutdown():
    chat_handler.close()
