"""Tests for image-aware chunk creation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_document
from src.config import Config
from src.parser import ParsedDocument, Section


class ChunkerImageTests(unittest.TestCase):
    def test_creates_image_chunk_and_attaches_image_metadata_to_text_chunk(self) -> None:
        doc = ParsedDocument(
            doc_id="0001-faq-721033694",
            title="FAQ 中文版",
            path="FAQ",
            last_updated="",
            source_url="https://inside-docupedia/page",
            page_id=721033694,
            scope="",
            file_path="documents/0001-faq-721033694.md",
            doc_content_hash="sha256:test",
            body_sections=[
                Section(
                    title="Login",
                    level=2,
                    content=(
                        "Follow the login steps.\n\n"
                        "![登录截图](../assets/721033694/001-login.png)\n\n"
                        "Then continue."
                    ),
                )
            ],
        )
        image = {
            "image_id": "img-1",
            "filename": "login.png",
            "relative_path": "assets/721033694/001-login.png",
            "markdown_path": "../assets/721033694/001-login.png",
            "caption": "登录截图",
            "summary": "展示登录步骤",
            "ocr_text": "Login",
            "embedding_text": "图片说明: 登录截图\n图片OCR文本: Login",
            "is_decorative": False,
        }

        chunks = chunk_document(doc, Config(), images=[image])

        image_chunks = [chunk for chunk in chunks if chunk.chunk_type == "image"]
        text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "child"]

        self.assertEqual(len(image_chunks), 1)
        self.assertEqual(image_chunks[0].doc_id, "0001-faq-721033694")
        self.assertEqual(image_chunks[0].section_title, "Login")
        self.assertIn("图片OCR文本: Login", image_chunks[0].content)
        self.assertEqual(image_chunks[0].image["image_id"], "img-1")

        self.assertTrue(text_chunks)
        self.assertEqual(text_chunks[0].images[0]["image_id"], "img-1")


if __name__ == "__main__":
    unittest.main()
