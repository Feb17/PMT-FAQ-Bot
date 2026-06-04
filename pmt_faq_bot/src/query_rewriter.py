"""LLM-based query rewriting using vLLM's OpenAI-compatible API.

Production RAG pattern: instead of embedding the user's raw query, first
ask an LLM to generate 2-3 paraphrases or alternative phrasings. Then run
hybrid recall for each rewrite and fuse the candidate lists.

This measurably boosts recall on:
 - abstract queries ("介绍下 X") where the user's phrasing doesn't match doc text
 - cross-lingual queries (CN query for EN doc, or vice versa)
 - vocabulary-mismatch queries ("网络设备的配置备份" vs "startup-config backup")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a retrieval query expansion assistant for a Bosch data center operations knowledge base. The docs are Confluence exports covering: Windows/Linux OS, networking (Cisco), Veeam/Oracle backup, Active Directory, Nutanix, Redis/MySQL/PostgreSQL/MongoDB, compliance and hardening.

Given a user's query, produce 2-3 alternative rewrites that will help retrieve relevant documents. Rules:

1. Include technical synonyms and concrete product/tool names where relevant (e.g. "WSUS" for "Windows patch", "Veeam" for "backup").
2. Produce both a Chinese and English version if the query is ambiguous or single-language.
3. Do NOT invent specifics the user didn't mention. Stay faithful to the original intent.
4. Each rewrite should be a SHORT phrase, not a long sentence.
5. Output strict JSON: {"queries": ["rewrite1", "rewrite2", "rewrite3"]}. No other text.

Example:
Input: "介绍下 DISC foundational services"
Output: {"queries": ["DISC foundational services overview", "DISC 基础服务有哪些", "WSUS RedHat Satellite Symantec centralized management"]}

Example:
Input: "网络设备的配置备份怎么做？"
Output: {"queries": ["network device configuration backup", "Cisco switch running-config startup-config backup", "Nexus IOS-XE backup procedure"]}"""


class QueryRewriter:
    def __init__(
        self,
        llm_url: str,
        model: str,
        timeout: float = 20.0,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        self._url = llm_url.rstrip("/") + "/v1/chat/completions"
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = httpx.Client(timeout=timeout)

    def rewrite(self, query: str) -> list[str]:
        """Return list of queries: [original] + rewrites. Always includes original.

        On any failure, returns just [original] — degrades gracefully.
        """
        try:
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "response_format": {"type": "json_object"},
            }
            r = self._client.post(self._url, json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            rewrites = _parse_rewrites(content)

            # Keep original first (anchor), then add deduped rewrites
            result = [query]
            seen = {query.strip().lower()}
            for rw in rewrites:
                norm = rw.strip().lower()
                if norm and norm not in seen:
                    result.append(rw.strip())
                    seen.add(norm)
            return result[:4]  # cap at original + 3 rewrites
        except Exception as exc:
            logger.warning("Query rewrite failed, using original only: %s", exc)
            return [query]

    def rewrite_standalone(self, messages: list[dict]) -> str:
        """Given multi-turn chat messages, return a standalone search query.

        If only one user message exists, returns it as-is (no LLM call).
        For multi-turn, calls the LLM to resolve pronouns and context.
        On failure, falls back to the last user message.
        """
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return ""
        last_user = user_msgs[-1].get("content", "").strip()
        if len(user_msgs) <= 1:
            return last_user

        try:
            history_text = "\n".join(
                f"{m['role'].upper()}: {m.get('content', '')}"
                for m in messages[-8:]  # last 4 turns max
                if m.get("role") in ("user", "assistant")
            )
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Given the following conversation, rewrite the user's "
                            "latest question into a standalone search query that "
                            "incorporates relevant context from prior turns. "
                            "Output ONLY the rewritten query, nothing else."
                        ),
                    },
                    {"role": "user", "content": history_text},
                ],
                "temperature": 0.0,
                "max_tokens": 128,
            }
            r = self._client.post(self._url, json=payload)
            r.raise_for_status()
            rewritten = r.json()["choices"][0]["message"]["content"].strip()
            if rewritten:
                logger.info("Standalone rewrite: %r -> %r", last_user, rewritten)
                return rewritten
        except Exception as exc:
            logger.warning("Standalone rewrite failed, using last user msg: %s", exc)

        return last_user

    def close(self) -> None:
        self._client.close()


def _parse_rewrites(content: str) -> list[str]:
    """Parse LLM JSON output robustly."""
    content = content.strip()
    # Strip markdown fences if present
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract a JSON object from within the text
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, dict):
        return []

    queries = data.get("queries") or data.get("rewrites") or []
    if not isinstance(queries, list):
        return []

    return [str(q) for q in queries if q and isinstance(q, str)]
