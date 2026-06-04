"""End-to-end test: parse and chunk real Confluence documents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_document
from src.config import Config
from src.parser import parse_file

REAL_DOCS = Path(__file__).resolve().parent.parent.parent / "confluence-dify-clean-export" / "documents"


def _cfg() -> Config:
    return Config()


def test_real_cctv():
    """CCTV — a very short document."""
    p = REAL_DOCS / "8-compliance-datacenter" / "0950-cctv-6019281145.md"
    if not p.exists():
        print(f"  SKIP (file not found): {p}")
        return

    doc = parse_file(p, REAL_DOCS)
    assert doc.title == "CCTV"
    assert doc.page_id == 6019281145
    assert doc.scope == "Describe the CCTV Information in CoDC"

    chunks = chunk_document(doc, _cfg())
    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    print(f"  CCTV: {len(parents)} parents, {len(children)} children")
    for c in children:
        print(f"    [{c.chunk_id}] ({len(c.content)} chars)")
    assert len(children) >= 1


def test_real_mysql():
    """MySQL — a very long document with huge parameter tables."""
    p = REAL_DOCS / "8-compliance-datacenter" / "0936-mysql-5384936508.md"
    if not p.exists():
        print(f"  SKIP (file not found): {p}")
        return

    doc = parse_file(p, REAL_DOCS)
    assert doc.title == "MySQL"
    assert doc.page_id == 5384936508

    chunks = chunk_document(doc, _cfg())
    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    print(f"  MySQL: {len(parents)} parents, {len(children)} children")
    for c in children[:5]:
        print(f"    [{c.chunk_id}] ({len(c.content)} chars) {c.content[:80]}...")

    assert len(parents) >= 1
    assert len(children) >= 3

    # Verify tables were split (the parameter table has 200+ rows)
    table_chunks = [c for c in children if "| " in c.content and "---" in c.content]
    print(f"  Table chunks: {len(table_chunks)}")
    assert len(table_chunks) >= 2, "Large parameter table should be split"


def test_real_poc_network():
    """POC Network — no metadata table."""
    p = REAL_DOCS / "8-compliance-datacenter" / "0923-01-poc-network-5180327622.md"
    if not p.exists():
        print(f"  SKIP (file not found): {p}")
        return

    doc = parse_file(p, REAL_DOCS)
    assert doc.title == "01 POC Network"
    assert doc.scope == ""  # no metadata table

    chunks = chunk_document(doc, _cfg())
    children = [c for c in chunks if c.chunk_type == "child"]

    print(f"  POC Network: {len(children)} children")
    assert len(children) >= 1


def test_batch_parse_sample():
    """Parse a batch of 20 real documents, report any failures."""
    cfg = _cfg()
    docs_dir = REAL_DOCS / "8-compliance-datacenter"
    if not docs_dir.exists():
        print(f"  SKIP (dir not found): {docs_dir}")
        return

    files = sorted(docs_dir.glob("*.md"))[:20]
    total_chunks = 0
    failures = []

    for f in files:
        try:
            doc = parse_file(f, REAL_DOCS)
            chunks = chunk_document(doc, cfg)
            n_children = sum(1 for c in chunks if c.chunk_type == "child")
            total_chunks += n_children
        except Exception as exc:
            failures.append((f.name, str(exc)))

    print(f"  Batch: {len(files)} files, {total_chunks} child chunks, {len(failures)} failures")
    for name, err in failures:
        print(f"    FAIL: {name}: {err}")

    assert len(failures) == 0, f"Failures: {failures}"


if __name__ == "__main__":
    print("Running end-to-end tests on real documents...\n")
    test_real_cctv()
    test_real_mysql()
    test_real_poc_network()
    test_batch_parse_sample()
    print("\nAll end-to-end tests passed!")
