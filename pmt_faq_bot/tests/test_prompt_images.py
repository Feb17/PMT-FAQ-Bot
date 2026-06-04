"""Tests for image rendering in RAG responses."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import RetrievedChunk
from src.server.prompt import format_related_images, format_sources


def _chunk(**overrides) -> RetrievedChunk:
    values = {
        "chunk_id": "c1",
        "chunk_type": "child",
        "doc_id": "doc",
        "title": "FAQ 中文版",
        "section_title": "Login",
        "content": "answer context",
        "source_url": "https://inside-docupedia/page",
        "score": 0.9,
        "images": [],
        "image": None,
    }
    values.update(overrides)
    return RetrievedChunk(**values)


class PromptImageTests(unittest.TestCase):
    def test_format_related_images_deduplicates_and_limits_markdown_images(self) -> None:
        first_image = {
            "image_id": "img-1",
            "caption": "登录截图",
            "relative_path": "assets/721033694/001-login.png",
        }
        duplicate_first = dict(first_image)
        chunks = [
            _chunk(chunk_type="image", image=first_image, images=[first_image]),
            _chunk(chunk_id="c2", images=[duplicate_first]),
            _chunk(
                chunk_id="c3",
                images=[
                    {
                        "image_id": "img-2",
                        "caption": "配置截图",
                        "relative_path": "assets/721033694/002-config.png",
                    },
                    {
                        "image_id": "img-3",
                        "caption": "第三张图",
                        "relative_path": "assets/721033694/003-third.png",
                    },
                    {
                        "image_id": "img-4",
                        "caption": "第四张图",
                        "relative_path": "assets/721033694/004-fourth.png",
                    },
                ],
            ),
        ]

        rendered = format_related_images(
            chunks,
            asset_base_url="http://rag-api:8090/assets/confluence",
            max_images=3,
        )

        self.assertIn("**相关图片:**", rendered)
        self.assertEqual(rendered.count("!["), 3)
        self.assertIn(
            "![图1: 登录截图](http://rag-api:8090/assets/confluence/assets/721033694/001-login.png)",
            rendered,
        )
        self.assertNotIn("004-fourth.png", rendered)

    def test_format_sources_includes_related_images_after_sources(self) -> None:
        chunks = [
            _chunk(
                images=[
                    {
                        "image_id": "img-1",
                        "caption": "登录截图",
                        "url": "http://rag-api/assets/confluence/721033694/login.png",
                    }
                ]
            )
        ]

        rendered = format_sources(chunks)

        self.assertIn("**Sources:**", rendered)
        self.assertIn("**相关图片:**", rendered)
        self.assertIn("![图1: 登录截图](http://rag-api/assets/confluence/721033694/login.png)", rendered)


if __name__ == "__main__":
    unittest.main()
