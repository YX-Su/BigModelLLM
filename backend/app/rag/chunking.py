"""Semantic-driven document chunking.

The knowledge base is authored in Markdown where every level-2 heading (``## ``)
delimits one self-contained semantic unit — a config item, a rule, or a
solution template. Each such section becomes a chunk, so a chunk always
expresses one complete unit rather than an arbitrary fixed window.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.config import settings
from app.schemas.knowledge import Chunk

# Config item codes: <DIM-PREFIX>(-SEGMENT)+, e.g. ASSET-EQUITY, STRAT-REBAL-Q
_CONFIG_CODE_RE = re.compile(r"\b(?:RP|HORIZON|ASSET|STRAT|RISK)(?:-[A-Z0-9]+)+\b")


def _extract_codes(text: str) -> list[str]:
    """Return the distinct config item codes mentioned in a piece of text."""
    seen: list[str] = []
    for match in _CONFIG_CODE_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def chunk_file(path: Path) -> list[Chunk]:
    """Split a single Markdown file into one chunk per level-2 section."""
    doc = path.stem
    chunks: list[Chunk] = []
    section = ""
    title = ""
    body: list[str] = []
    in_chunk = False
    seq = 0

    def flush() -> None:
        nonlocal seq
        if not in_chunk or not title:
            return
        text = title + "\n" + "\n".join(body).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc}#{seq:03d}",
                doc=doc,
                section=section,
                title=title,
                text=text.strip(),
                config_codes=_extract_codes(text),
            )
        )
        seq += 1

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            flush()
            title = line[3:].strip()
            body = []
            in_chunk = True
        elif line.startswith("# "):
            flush()
            section = line[2:].strip()
            title = ""
            body = []
            in_chunk = False
        elif in_chunk:
            body.append(line)
    flush()
    return chunks


def chunk_corpus(raw_dir: Path | None = None) -> list[Chunk]:
    """Chunk every Markdown file under the raw data directory."""
    raw_dir = raw_dir or settings.raw_data_dir
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.glob("*.md")):
        chunks.extend(chunk_file(path))
    return chunks
