"""Chat completions handler: retrieval + generation + streaming."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncIterator

import httpx

from ..config import Config
from ..query_rewriter import QueryRewriter
from ..retrieval import RetrievalPipeline, RetrievedChunk
from .prompt import NO_RESULT_REPLY, build_messages, format_sources

logger = logging.getLogger(__name__)


class ChatHandler:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._pipeline = RetrievalPipeline(cfg)
        self._rewriter = QueryRewriter(
            llm_url=cfg.llm_url,
            model=cfg.llm_model,
        )
        self._llm_url = cfg.llm_url.rstrip("/") + "/v1/chat/completions"

    async def handle(self, body: dict) -> dict | AsyncIterator[str]:
        """Process a chat completion request.

        Returns either a full response dict (non-streaming) or an async
        iterator of SSE lines (streaming).
        """
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = self._cfg.rag_model_name

        # 1. Rewrite multi-turn to standalone query
        search_query = self._rewriter.rewrite_standalone(messages)
        logger.info("Search query: %r", search_query)

        # 2. Retrieve
        chunks = self._pipeline.retrieve(search_query, top_k=5)

        # 3. Score gate
        if not chunks or chunks[0].score < self._cfg.rag_score_threshold:
            logger.info("No relevant results (top score=%.3f)", chunks[0].score if chunks else 0)
            if stream:
                return self._stream_plain(NO_RESULT_REPLY, request_id, model_name)
            return self._sync_response(NO_RESULT_REPLY, request_id, model_name)

        logger.info(
            "Retrieved %d chunks (top: %s score=%.3f)",
            len(chunks), chunks[0].doc_id, chunks[0].score,
        )

        # 4. Build RAG prompt
        # Drop the last user message from history — we'll re-add it inside build_messages
        history = messages[:-1] if messages else []
        user_query = messages[-1].get("content", search_query) if messages else search_query
        llm_messages = build_messages(
            user_query, chunks, history, self._cfg.history_max_turns,
            asset_base_url=self._cfg.rag_asset_base_url,
        )

        # 5. Call vLLM
        sources_text = format_sources(chunks)

        if stream:
            return self._stream_rag(llm_messages, sources_text, request_id, model_name)
        return await self._sync_rag(llm_messages, sources_text, request_id, model_name)

    # ------------------------------------------------------------------
    # Streaming responses
    # ------------------------------------------------------------------

    async def _stream_rag(
        self,
        messages: list[dict],
        sources_text: str,
        request_id: str,
        model_name: str,
    ) -> AsyncIterator[str]:
        """SSE stream: proxy vLLM tokens, then append sources."""
        payload = {
            "model": self._cfg.llm_model,
            "messages": messages,
            "stream": True,
            "max_tokens": 2048,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", self._llm_url, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield self._sse_chunk(content, request_id, model_name)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        # Append sources after the answer
        if sources_text:
            yield self._sse_chunk(sources_text, request_id, model_name)

        # Final stop chunk + DONE
        yield self._sse_stop(request_id, model_name)
        yield "data: [DONE]\n\n"

    async def _stream_plain(
        self, text: str, request_id: str, model_name: str
    ) -> AsyncIterator[str]:
        """Stream a fixed text response."""
        yield self._sse_chunk(text, request_id, model_name)
        yield self._sse_stop(request_id, model_name)
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Non-streaming responses
    # ------------------------------------------------------------------

    async def _sync_rag(
        self,
        messages: list[dict],
        sources_text: str,
        request_id: str,
        model_name: str,
    ) -> dict:
        payload = {
            "model": self._cfg.llm_model,
            "messages": messages,
            "stream": False,
            "max_tokens": 2048,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self._llm_url, json=payload)
            r.raise_for_status()
            data = r.json()

        content = data["choices"][0]["message"]["content"] + sources_text
        return self._full_response(content, request_id, model_name)

    def _sync_response(self, text: str, request_id: str, model_name: str) -> dict:
        return self._full_response(text, request_id, model_name)

    # ------------------------------------------------------------------
    # SSE formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sse_chunk(content: str, request_id: str, model: str) -> str:
        obj = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    @staticmethod
    def _sse_stop(request_id: str, model: str) -> str:
        obj = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    @staticmethod
    def _full_response(content: str, request_id: str, model: str) -> dict:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def close(self) -> None:
        self._pipeline.close()
        self._rewriter.close()
