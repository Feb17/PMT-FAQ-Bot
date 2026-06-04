"""Unit tests for image retrieval evaluation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_retrieval_eval import evaluate_answer_text


class ImageRetrievalEvalTests(unittest.TestCase):
    def test_detects_image_markdown_and_expected_terms(self) -> None:
        result = evaluate_answer_text(
            "答案\n\n![图1: 登录截图](http://rag-api/assets/login.png)\nQdrant 6333",
            {
                "must_return_image": True,
                "expected_image_terms": ["![", "]("],
                "expected_terms": ["Qdrant", "6333"],
            },
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["contains_image_markdown"])

    def test_fails_when_image_is_required_but_missing(self) -> None:
        result = evaluate_answer_text(
            "只有文字答案",
            {"must_return_image": True, "expected_terms": []},
        )

        self.assertFalse(result["ok"])
        self.assertFalse(result["contains_image_markdown"])


if __name__ == "__main__":
    unittest.main()
