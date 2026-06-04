"""Tests for controlled image asset path resolution."""

from __future__ import annotations

import sys
import unittest
import importlib
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.server.assets import resolve_asset_path


class AssetServingTests(unittest.TestCase):
    def test_resolves_existing_asset_inside_root(self) -> None:
        root = Path(__file__).resolve().parent / "fixtures" / "asset_export"
        asset = root / "assets" / "721033694" / "login.png"

        resolved = resolve_asset_path(root, "assets/721033694/login.png")

        self.assertEqual(resolved, asset.resolve())

    def test_rejects_path_traversal(self) -> None:
        root = Path(__file__).resolve().parent / "fixtures" / "asset_export"

        resolved = resolve_asset_path(root, "../secret.png")

        self.assertIsNone(resolved)

    def test_fastapi_asset_route_serves_only_configured_root(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this local Python environment")

        old_asset_dir = os.environ.get("RAG_ASSET_DIR")
        try:
            root = Path(__file__).resolve().parent / "fixtures" / "asset_export"
            os.environ["RAG_ASSET_DIR"] = str(root)

            app_module = importlib.import_module("src.server.app")
            app_module = importlib.reload(app_module)

            with TestClient(app_module.app) as client:
                ok_response = client.get("/assets/confluence/assets/721033694/login.png")
                blocked_response = client.get("/assets/confluence/../secret.png")

            app_module.chat_handler.close()
        finally:
            if old_asset_dir is None:
                os.environ.pop("RAG_ASSET_DIR", None)
            else:
                os.environ["RAG_ASSET_DIR"] = old_asset_dir

        self.assertEqual(ok_response.status_code, 200)
        self.assertEqual(ok_response.content, b"png\n")
        self.assertEqual(blocked_response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
