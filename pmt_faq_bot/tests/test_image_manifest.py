"""Tests for processed image manifest loading."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_manifest import load_processed_images_by_doc_id


class ImageManifestTests(unittest.TestCase):
    def test_loads_processed_images_by_document_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "export"
            docs_dir = export_dir / "documents"
            docs_dir.mkdir(parents=True)
            (docs_dir / "0001-faq-721033694.md").write_text("# FAQ\n", encoding="utf-8")
            (export_dir / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "pageId": "721033694",
                        "relativePath": "documents/0001-faq-721033694.md",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (export_dir / "processed_image_manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "pageId": "721033694",
                                "imageId": "img-1",
                                "filename": "login.png",
                                "relativePath": "assets/721033694/001-login.png",
                                "markdownPath": "../assets/721033694/001-login.png",
                                "caption": "登录截图",
                                "summary": "展示登录步骤",
                                "ocrText": "Login",
                                "embeddingText": "图片说明: 登录截图\n图片OCR文本: Login",
                                "imageProcessingStatus": "processed",
                                "isDecorative": False,
                            }
                        ),
                        json.dumps(
                            {
                                "pageId": "721033694",
                                "imageId": "img-2",
                                "filename": "missing.png",
                                "relativePath": "assets/721033694/002-missing.png",
                                "imageProcessingStatus": "skipped_unavailable",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            by_doc = load_processed_images_by_doc_id(docs_dir)

            self.assertEqual(list(by_doc.keys()), ["0001-faq-721033694"])
            self.assertEqual(len(by_doc["0001-faq-721033694"]), 1)
            image = by_doc["0001-faq-721033694"][0]
            self.assertEqual(image["image_id"], "img-1")
            self.assertEqual(image["caption"], "登录截图")
            self.assertEqual(image["ocr_text"], "Login")
            self.assertEqual(
                image["markdown_path"], "../assets/721033694/001-login.png"
            )


if __name__ == "__main__":
    unittest.main()
