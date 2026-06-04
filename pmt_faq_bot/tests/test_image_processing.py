"""Tests for exported Confluence image processing."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_processing import ImageProcessingConfig, process_image_manifest


class FakeOcr:
    def __init__(self) -> None:
        self.calls = 0

    def extract_text(self, image_path: Path) -> str:
        self.calls += 1
        return f"OCR from {image_path.name}"


class FakeCaptioner:
    def __init__(self) -> None:
        self.calls = 0

    def describe(self, image_path: Path) -> dict:
        self.calls += 1
        return {
            "caption": "登录页面截图",
            "image_type": "screenshot",
            "summary": "截图展示 FAQ 登录步骤。",
            "visible_text": "Login",
            "entities": ["Login", "FAQ"],
            "is_decorative": False,
        }


class ImageProcessingTests(unittest.TestCase):
    def test_processes_downloaded_images_and_skips_failed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "export"
            asset_dir = export_dir / "assets" / "721033694"
            asset_dir.mkdir(parents=True)
            (asset_dir / "001-login.png").write_bytes(
                b"\x89PNG\r\n\x1a\nfake-image-bytes"
            )
            (export_dir / "image_manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "pageId": "721033694",
                                "imageId": "img-1",
                                "filename": "login.png",
                                "relativePath": "assets/721033694/001-login.png",
                                "status": "downloaded",
                            }
                        ),
                        json.dumps(
                            {
                                "pageId": "721033694",
                                "imageId": "img-2",
                                "filename": "missing.png",
                                "relativePath": "assets/721033694/002-missing.png",
                                "status": "download_failed",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ocr = FakeOcr()
            captioner = FakeCaptioner()
            report = process_image_manifest(
                export_dir,
                ImageProcessingConfig(cache_dir=export_dir / "image_cache"),
                ocr_engine=ocr,
                captioner=captioner,
            )

            self.assertEqual(report.total_images, 2)
            self.assertEqual(report.processed, 1)
            self.assertEqual(report.skipped_unavailable, 1)
            self.assertEqual(ocr.calls, 1)
            self.assertEqual(captioner.calls, 1)

            output_path = export_dir / "processed_image_manifest.jsonl"
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(records[0]["imageProcessingStatus"], "processed")
            self.assertEqual(records[0]["ocrText"], "OCR from 001-login.png")
            self.assertEqual(records[0]["caption"], "登录页面截图")
            self.assertIn("图片OCR文本: OCR from 001-login.png", records[0]["embeddingText"])
            self.assertEqual(records[1]["imageProcessingStatus"], "skipped_unavailable")

    def test_reuses_cache_by_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "export"
            asset_dir = export_dir / "assets" / "721033694"
            asset_dir.mkdir(parents=True)
            (asset_dir / "001-login.png").write_bytes(b"same-image")
            (export_dir / "image_manifest.jsonl").write_text(
                json.dumps(
                    {
                        "pageId": "721033694",
                        "imageId": "img-1",
                        "filename": "login.png",
                        "relativePath": "assets/721033694/001-login.png",
                        "status": "downloaded",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = ImageProcessingConfig(cache_dir=export_dir / "image_cache")
            first_ocr = FakeOcr()
            first_captioner = FakeCaptioner()
            process_image_manifest(
                export_dir, cfg, ocr_engine=first_ocr, captioner=first_captioner
            )

            second_ocr = FakeOcr()
            second_captioner = FakeCaptioner()
            report = process_image_manifest(
                export_dir, cfg, ocr_engine=second_ocr, captioner=second_captioner
            )

            self.assertEqual(report.cached, 1)
            self.assertEqual(second_ocr.calls, 0)
            self.assertEqual(second_captioner.calls, 0)


if __name__ == "__main__":
    unittest.main()
