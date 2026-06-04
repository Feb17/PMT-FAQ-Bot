"""Tests for the hierarchical chunker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_document
from src.config import Config
from src.parser import parse_file

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _cfg() -> Config:
    return Config()


def test_short_doc_single_child_chunks():
    """A short document should produce child chunks only (no parents)."""
    doc = parse_file(FIXTURES / "0001-short-doc-1234567890.md", FIXTURES)
    chunks = chunk_document(doc, _cfg())

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    assert len(parents) == 0, "Short doc should not have parent chunks"
    assert len(children) >= 1, "Short doc should have at least 1 child chunk"

    for child in children:
        assert child.parent_chunk_id is None
        assert child.doc_id == "0001-short-doc-1234567890"


def test_long_doc_has_parents_and_children():
    """A long document should produce both parent and child chunks."""
    doc = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)
    chunks = chunk_document(doc, _cfg())

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    assert len(parents) >= 1, "Long doc should have parent chunks"
    assert len(children) >= 1, "Long doc should have child chunks"


def test_child_chunks_reference_parent():
    doc = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)
    chunks = chunk_document(doc, _cfg())

    parent_ids = {c.chunk_id for c in chunks if c.chunk_type == "parent"}
    children = [c for c in chunks if c.chunk_type == "child"]

    for child in children:
        if child.parent_chunk_id is not None:
            assert child.parent_chunk_id in parent_ids, (
                f"Child {child.chunk_id} references non-existent parent {child.parent_chunk_id}"
            )


def test_embed_text_has_context_prefix():
    doc = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)
    chunks = chunk_document(doc, _cfg())

    children = [c for c in chunks if c.chunk_type == "child"]
    assert len(children) > 0

    for child in children:
        assert child.content_for_embedding.startswith("文档: Database Service")
        assert "---" in child.content_for_embedding


def test_tables_are_preserved():
    """Tables should appear as recognizable chunks."""
    doc = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)
    chunks = chunk_document(doc, _cfg())

    children = [c for c in chunks if c.chunk_type == "child"]
    table_chunks = [c for c in children if "| " in c.content and "---" in c.content]
    assert len(table_chunks) >= 1, "Should have at least one table chunk"


def test_code_blocks_preserved():
    """Code blocks should be kept intact."""
    doc = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)
    chunks = chunk_document(doc, _cfg())

    children = [c for c in chunks if c.chunk_type == "child"]
    code_chunks = [c for c in children if "```" in c.content]
    assert len(code_chunks) >= 1, "Should have at least one code block chunk"


def test_no_meta_table_doc_chunks():
    doc = parse_file(FIXTURES / "0003-no-meta-table-5555555555.md", FIXTURES)
    chunks = chunk_document(doc, _cfg())

    assert len(chunks) >= 1
    all_doc_ids = {c.doc_id for c in chunks}
    assert all_doc_ids == {"0003-no-meta-table-5555555555"}


def test_chunk_ids_unique():
    """All chunk IDs within a document must be unique."""
    for fixture in FIXTURES.glob("*.md"):
        doc = parse_file(fixture, FIXTURES)
        chunks = chunk_document(doc, _cfg())
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), (
            f"Duplicate chunk IDs in {fixture.name}: {[x for x in ids if ids.count(x) > 1]}"
        )


if __name__ == "__main__":
    test_short_doc_single_child_chunks()
    test_long_doc_has_parents_and_children()
    test_child_chunks_reference_parent()
    test_embed_text_has_context_prefix()
    test_tables_are_preserved()
    test_code_blocks_preserved()
    test_no_meta_table_doc_chunks()
    test_chunk_ids_unique()
    print("All chunker tests passed!")
