"""Small evaluator for image-aware RAG smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib import request


def evaluate_answer_text(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    contains_image = "![" in answer and "](" in answer
    missing_terms = [
        term for term in case.get("expected_terms", []) if str(term) not in answer
    ]
    missing_image_terms = [
        term
        for term in case.get("expected_image_terms", [])
        if str(term) not in answer
    ]
    image_required = bool(case.get("must_return_image", False))
    ok = not missing_terms and not missing_image_terms
    if image_required and not contains_image:
        ok = False
    return {
        "ok": ok,
        "contains_image_markdown": contains_image,
        "missing_terms": missing_terms,
        "missing_image_terms": missing_image_terms,
    }


def run_eval(api_url: str, model: str, cases_path: Path) -> dict[str, Any]:
    cases = _load_jsonl(cases_path)
    results = []
    for case in cases:
        answer = _call_rag_api(api_url, model, str(case["query"]))
        results.append(
            {
                "query": case["query"],
                **evaluate_answer_text(answer, case),
            }
        )
    passed = sum(1 for item in results if item["ok"])
    return {"total": len(results), "passed": passed, "results": results}


def _call_rag_api(api_url: str, model: str, query: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": False,
    }
    req = request.Request(
        api_url.rstrip("/") + "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"invalid JSONL record at {path}:{line_no}")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image RAG smoke evaluation.")
    parser.add_argument("--api-url", required=True, help="RAG API base URL")
    parser.add_argument("--model", default="pmt_faq_bot", help="RAG model name")
    parser.add_argument("--cases", required=True, help="JSONL image eval cases")
    args = parser.parse_args()

    result = run_eval(args.api_url, args.model, Path(args.cases))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
