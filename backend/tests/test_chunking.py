"""Unit tests for semantic chunking (pure functions, no external services)."""
from __future__ import annotations

from app.rag.chunking import chunk_corpus


def test_chunk_counts_per_document():
    chunks = chunk_corpus()
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1

    assert by_doc["00_overview"] == 5
    assert by_doc["01_config_items"] == 25
    assert by_doc["02_rules"] == 30
    assert by_doc["03_solution_templates"] == 5
    assert len(chunks) == 65


def test_chunk_ids_are_unique():
    chunks = chunk_corpus()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_config_codes_extracted():
    chunks = chunk_corpus()
    item = next(c for c in chunks if c.title.startswith("RP-CONSERVATIVE"))
    assert "RP-CONSERVATIVE" in item.config_codes
    assert item.section == "客户画像维度"

    multi = next(c for c in chunks if c.title.startswith("STRAT-REBAL-Q"))
    assert "STRAT-REBAL-Q" in multi.config_codes


def test_chunk_length_in_range():
    # Semantic units should stay reasonably compact for retrieval precision.
    for c in chunk_corpus():
        assert 40 <= len(c.text) <= 600, (c.chunk_id, len(c.text))
