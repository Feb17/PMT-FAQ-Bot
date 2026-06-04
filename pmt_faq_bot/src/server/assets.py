"""Controlled static asset helpers for exported Confluence images."""

from __future__ import annotations

from pathlib import Path


def resolve_asset_path(asset_root: Path, requested_path: str) -> Path | None:
    """Resolve an asset path if it stays inside ``asset_root`` and exists."""
    root = Path(asset_root).resolve()
    candidate = (root / requested_path.replace("\\", "/").lstrip("/")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate
