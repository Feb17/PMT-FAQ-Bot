"""Parse Confluence-exported Markdown files and extract structured metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    path: str
    last_updated: str
    source_url: str
    page_id: Optional[int]
    scope: str
    file_path: str
    doc_content_hash: str
    body_sections: list[Section] = field(default_factory=list)


@dataclass
class Section:
    """A top-level section delimited by H1 or H2 headers."""

    title: str
    level: int  # 1 or 2
    content: str  # raw markdown content below the header (excluding the header line itself)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_PATH = re.compile(r"^Path:\s*(.+)$", re.MULTILINE)
_RE_UPDATED = re.compile(r"^Last updated:\s*(.+)$", re.MULTILINE)
_RE_SOURCE = re.compile(
    r"^Source:\s*\[?(https?://[^\s\]\)]+)",
    re.MULTILINE,
)
_RE_PAGE_ID_URL = re.compile(r"/pages/(\d+)")
_RE_PAGE_ID_FILENAME = re.compile(r"-(\d{7,})\.md$")

# Matches the metadata property table that appears right after the Source line.
# It is a markdown table whose first column key is one of the known properties.
_META_KEYS = {
    "scope",
    "responsible",
    "access level",
    "last review",
    "labels",
    "document  status",
    "document status",
}

_RE_HEADER = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)

_SKIP_SECTIONS = {"revision history"}

# Low-value section titles (typically references, placeholders, admin boilerplate).
# These sections are still kept but marked in payload so the retriever can down-weight them.
LOW_VALUE_SECTIONS = {
    "references",
    "reference",
    "others",
    "other",
    "revision history",
    "appendix",
}

# Content treated as empty: "n.a", "---", "xxx", "tbd", pure whitespace,
# short placeholder strings (<=4 non-whitespace chars), or just a number.
_EMPTY_CONTENT = re.compile(
    r"^\s*(n\.?a\.?|---|x{1,}|tbd|todo|none|\d+|\s)*$", re.IGNORECASE
)

# Confluence template/macro export artifacts that leak through as a single line of
# parameters, e.g. info panels or template includes.
# Examples:
#   "true DISC Foundational Services false 600 auto top 6106741452 true 2891 1181"
#   "ITOM_Glossary_Header .ITOM_Template_V3.0_Glossary_Include false  2 true square false pipe"
_RE_ITOM_TEMPLATE = re.compile(r"ITOM_\w+|\.ITOM_Template_", re.IGNORECASE)
_RE_LONG_PAGE_ID = re.compile(r"\b\d{7,}\b")
_RE_BOOL_TOKEN = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def is_low_value_section(section_title: str) -> bool:
    return section_title.strip().lower() in LOW_VALUE_SECTIONS


def is_placeholder_content(text: str) -> bool:
    """Return True if content is likely a placeholder without real information."""
    stripped = text.strip()
    if not stripped:
        return True
    if _EMPTY_CONTENT.match(stripped):
        return True
    # Remove whitespace and markdown symbols, count informative chars
    informative = re.sub(r"[\s\-*#`|_]+", "", stripped)
    if len(informative) <= 4:
        return True
    # Confluence template/macro artifacts (single-line parameter dumps)
    if _is_confluence_macro_noise(stripped):
        return True
    return False


def _is_confluence_macro_noise(text: str) -> bool:
    """Detect Confluence info-panel / template-include parameter dumps."""
    # Strong signal: ITOM template references
    if _RE_ITOM_TEMPLATE.search(text):
        return True
    # Info-panel heuristic: short text dominated by true/false + numeric page IDs
    if len(text) > 300:
        return False
    bool_count = len(_RE_BOOL_TOKEN.findall(text))
    has_page_id = bool(_RE_LONG_PAGE_ID.search(text))
    if bool_count >= 2 and has_page_id:
        return True
    return False


def _extract_meta_value(rows: list[list[str]], key_lower: str) -> str:
    for row in rows:
        if len(row) >= 2 and row[0].strip().lower() == key_lower:
            return row[1].strip()
    return ""


def _parse_table_rows(block: str) -> list[list[str]]:
    """Return cell values for each non-separator row of a markdown table."""
    rows: list[list[str]] = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        # first and last are empty from leading/trailing pipe
        cells = cells[1:-1] if len(cells) > 2 else cells
        if cells and all(re.fullmatch(r"-{3,}", c.strip()) for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def _compute_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_bold_markers(text: str) -> str:
    """Remove leading/trailing ** or * from header text."""
    return re.sub(r"^\*{1,2}(.*?)\*{1,2}$", r"\1", text.strip())


def _find_meta_table_end(lines: list[str], start: int) -> int:
    """Return the line index just past the metadata property table.

    The metadata table starts at *start* (first ``|`` line after the Source
    line) and continues as long as consecutive lines begin with ``|``.
    """
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        i += 1
    return i


def parse_file(filepath: Path, base_dir: Path) -> ParsedDocument:
    """Parse a single Confluence-exported Markdown file."""
    raw = filepath.read_text(encoding="utf-8")
    content_hash = _compute_hash(raw)

    # --- Title (first H1) ---
    title = ""
    title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    if title_match:
        title = _strip_bold_markers(title_match.group(1))

    # --- Inline metadata lines ---
    path_val = (_RE_PATH.search(raw) or _match_empty()).group(1).strip()
    updated_val = (_RE_UPDATED.search(raw) or _match_empty()).group(1).strip()
    source_val = (_RE_SOURCE.search(raw) or _match_empty()).group(1).strip()

    # --- Page ID ---
    page_id: Optional[int] = None
    pid_match = _RE_PAGE_ID_URL.search(source_val) or _RE_PAGE_ID_FILENAME.search(
        filepath.name
    )
    if pid_match:
        page_id = int(pid_match.group(1))

    # --- Doc ID (from filename without .md) ---
    doc_id = filepath.stem  # e.g. "0936-mysql-5384936508"

    # --- File path relative to base_dir ---
    try:
        rel = filepath.relative_to(base_dir)
    except ValueError:
        rel = Path(filepath.name)
    file_path = str(rel).replace("\\", "/")

    # --- Metadata table extraction ---
    lines = raw.splitlines()
    scope = ""
    meta_table_end = 0

    # Find the first table block whose first-column keys overlap with _META_KEYS
    in_table = False
    table_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and not in_table:
            in_table = True
            table_start = idx
        elif in_table and not stripped.startswith("|"):
            # End of table block — check if it's the meta table
            table_block = "\n".join(lines[table_start:idx])
            rows = _parse_table_rows(table_block)
            first_col_keys = {r[0].strip().lower() for r in rows if r}
            if first_col_keys & _META_KEYS:
                scope = _extract_meta_value(rows, "scope")
                meta_table_end = idx
                break
            in_table = False
    else:
        # Handle table that extends to end of file
        if in_table:
            table_block = "\n".join(lines[table_start:])
            rows = _parse_table_rows(table_block)
            first_col_keys = {r[0].strip().lower() for r in rows if r}
            if first_col_keys & _META_KEYS:
                scope = _extract_meta_value(rows, "scope")
                meta_table_end = len(lines)

    # --- Split into sections, skipping preamble + meta area ---
    body_text = _remove_preamble(raw, meta_table_end, lines)
    sections = _split_sections(body_text)

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        path=path_val,
        last_updated=updated_val,
        source_url=source_val,
        page_id=page_id,
        scope=scope,
        file_path=file_path,
        doc_content_hash=content_hash,
        body_sections=sections,
    )


def _remove_preamble(
    raw: str, meta_table_end_line: int, lines: list[str]
) -> str:
    """Remove title line, Path/Updated/Source lines, and metadata table.

    Returns the remaining body text starting after the preamble.
    """
    if meta_table_end_line > 0:
        body_lines = lines[meta_table_end_line:]
    else:
        # No meta table found — skip until after Source line or first H1 title
        start = 0
        for i, line in enumerate(lines):
            if _RE_SOURCE.match(line):
                start = i + 1
                break
            if i > 0 and re.match(r"^#\s", line):
                # First content header after the title header
                if i > 1:
                    start = i
                    break
        body_lines = lines[start:]

    return "\n".join(body_lines)


def _split_sections(body: str) -> list[Section]:
    """Split body text into sections by H1/H2 headers.

    Filters out empty sections and revision history.
    Ignores `#` lines that are inside fenced code blocks (e.g. bash comments).
    """
    sections: list[Section] = []
    code_ranges = _find_code_fence_ranges(body)
    matches = [
        m
        for m in _RE_HEADER.finditer(body)
        if not _is_inside(m.start(), code_ranges)
    ]

    if not matches:
        # No headers — treat the entire body as one section
        content = body.strip()
        if content and not is_placeholder_content(content):
            sections.append(Section(title="", level=1, content=content))
        return sections

    # Content before the first header
    pre = body[: matches[0].start()].strip()
    if pre and not is_placeholder_content(pre):
        sections.append(Section(title="", level=1, content=pre))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        header_text = _strip_bold_markers(m.group(2))

        if header_text.lower() in _SKIP_SECTIONS:
            continue
        if not header_text or header_text == "****":
            continue

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()

        # Remove trailing horizontal rules
        content = re.sub(r"\n---\s*$", "", content).strip()

        if is_placeholder_content(content):
            continue

        sections.append(Section(title=header_text, level=level, content=content))

    return sections


class _FakeMatch:
    def group(self, _n: int) -> str:
        return ""


def _match_empty() -> _FakeMatch:
    return _FakeMatch()


def _find_code_fence_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start_offset, end_offset) ranges for each fenced code block."""
    ranges: list[tuple[int, int]] = []
    in_code = False
    start = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            if not in_code:
                in_code = True
                start = offset
            else:
                ranges.append((start, offset + len(line)))
                in_code = False
        offset += len(line)
    if in_code:
        # Unclosed fence — treat rest of document as code
        ranges.append((start, len(text)))
    return ranges


def _is_inside(pos: int, ranges: list[tuple[int, int]]) -> bool:
    for a, b in ranges:
        if a <= pos < b:
            return True
    return False
