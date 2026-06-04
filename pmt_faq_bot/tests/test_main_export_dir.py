"""Tests for PMT FAQ Bot CLI export directory resolution."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import _fail_missing_image_manifest, _resolve_export_dir


class MainExportDirTests(unittest.TestCase):
    def test_resolve_export_dir_accepts_direct_export_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "export"
            export_dir.mkdir()
            (export_dir / "image_manifest.jsonl").write_text("", encoding="utf-8")

            self.assertEqual(_resolve_export_dir(export_dir), export_dir)

    def test_resolve_export_dir_accepts_single_nested_export_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "exports"
            export_dir = parent / "faq-export"
            export_dir.mkdir(parents=True)
            (export_dir / "image_manifest.jsonl").write_text("", encoding="utf-8")

            self.assertEqual(_resolve_export_dir(parent), export_dir)

    def test_missing_image_manifest_exits_with_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "export"
            export_dir.mkdir()

            with self.assertRaises(SystemExit) as exc:
                _fail_missing_image_manifest(export_dir)

            self.assertEqual(exc.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
