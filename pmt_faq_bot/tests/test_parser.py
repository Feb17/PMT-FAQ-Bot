"""Tests for the Markdown parser."""

import sys
from pathlib import Path

# Ensure the src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import parse_file

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_short_doc_metadata():
    doc = parse_file(FIXTURES / "0001-short-doc-1234567890.md", FIXTURES)

    assert doc.doc_id == "0001-short-doc-1234567890"
    assert doc.title == "Short Document"
    assert "Home > Category > Short Document" in doc.path
    assert doc.last_updated == "2025-06-01T10:00:00.000+02:00"
    assert doc.page_id == 1234567890
    assert doc.scope == "Test scope"
    assert doc.source_url.endswith("1234567890/Short+Document")
    assert doc.doc_content_hash.startswith("sha256:")


def test_short_doc_sections():
    doc = parse_file(FIXTURES / "0001-short-doc-1234567890.md", FIXTURES)

    titles = [s.title for s in doc.body_sections]
    # "Others" (n.a), "References" (n.a), and "Revision History" should be excluded
    assert "Revision History" not in titles
    assert "Others" not in titles
    assert "References" not in titles

    # "Description & Purpose" and "Details" should remain
    assert "Description & Purpose" in titles
    assert "Details" in titles


def test_long_doc_metadata():
    doc = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)

    assert doc.title == "Database Service"
    assert doc.page_id == 9876543210
    assert doc.scope == "PAAS document"


def test_long_doc_has_content_sections():
    doc = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)

    titles = [s.title for s in doc.body_sections]
    assert "Description & Purpose" in titles
    assert "Details" in titles
    assert "Others" in titles  # has actual content in this doc
    assert "Revision History" not in titles


def test_long_doc_details_contains_tables_and_code():
    doc = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)

    details = [s for s in doc.body_sections if s.title == "Details"][0]
    assert "Component" in details.content
    assert "MySQL" in details.content
    assert "```bash" in details.content or "mysqldump" in details.content


def test_no_meta_table_doc():
    doc = parse_file(FIXTURES / "0003-no-meta-table-5555555555.md", FIXTURES)

    assert doc.title == "POC Network Setup"
    assert doc.page_id == 5555555555
    assert doc.scope == ""  # no metadata table

    titles = [s.title for s in doc.body_sections]
    assert "Goal" in titles
    assert "Tasks" in titles
    assert "Results" in titles


def test_content_hash_changes_with_content():
    doc1 = parse_file(FIXTURES / "0001-short-doc-1234567890.md", FIXTURES)
    doc2 = parse_file(FIXTURES / "0002-long-doc-9876543210.md", FIXTURES)
    assert doc1.doc_content_hash != doc2.doc_content_hash


if __name__ == "__main__":
    test_short_doc_metadata()
    test_short_doc_sections()
    test_long_doc_metadata()
    test_long_doc_has_content_sections()
    test_long_doc_details_contains_tables_and_code()
    test_no_meta_table_doc()
    test_content_hash_changes_with_content()
    print("All parser tests passed!")
