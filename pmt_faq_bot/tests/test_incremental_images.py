"""Tests for image-aware incremental hashing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.incremental import compute_file_hash


class IncrementalImageTests(unittest.TestCase):
    def test_file_hash_changes_when_image_metadata_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "doc.md"
            doc.write_text("# FAQ\n\nBody\n", encoding="utf-8")

            first = compute_file_hash(
                doc,
                [
                    {
                        "image_id": "img-1",
                        "sha256": "sha256:image",
                        "caption": "登录截图",
                        "ocr_text": "Login",
                    }
                ],
            )
            second = compute_file_hash(
                doc,
                [
                    {
                        "image_id": "img-1",
                        "sha256": "sha256:image",
                        "caption": "登录截图更新",
                        "ocr_text": "Login",
                    }
                ],
            )

            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
