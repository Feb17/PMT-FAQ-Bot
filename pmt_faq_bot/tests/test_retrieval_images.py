"""Tests for image-aware retrieval filtering and payload mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import RetrievalPipeline, RetrievedChunk


class RetrievalImageTests(unittest.TestCase):
    def test_base_filter_recalls_child_and_image_chunks(self) -> None:
        filt = RetrievalPipeline._base_filter(exclude_low_value=True)

        self.assertIn(
            {"key": "chunk_type", "match": {"any": ["child", "image"]}},
            filt["must"],
        )
        self.assertIn(
            {"key": "is_low_value", "match": {"value": True}},
            filt["must_not"],
        )

    def test_retrieved_chunk_can_carry_image_metadata(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c1",
            chunk_type="image",
            doc_id="doc",
            title="FAQ",
            section_title="Login",
            content="图片说明: 登录截图",
            source_url="https://inside-docupedia/page",
            score=0.9,
            images=[{"image_id": "img-1"}],
            image={"image_id": "img-1"},
        )

        self.assertEqual(chunk.chunk_type, "image")
        self.assertEqual(chunk.images[0]["image_id"], "img-1")


if __name__ == "__main__":
    unittest.main()
