"""Mock-mode bootstrap.

Builds the in-memory knowledge base on server startup: chunk / vector /
keyword indexes are built with the real offline routines (which transparently
use the mock embedder and in-memory stores), while the knowledge graph is
populated directly from the static definitions — no LLM extraction needed.
"""
from __future__ import annotations

import logging

from app.graph.store import get_graph_store
from app.mock.knowledge import all_entities, all_relations, config_item_entities
from app.mock.llm import build_idf
from app.rag.chunking import chunk_corpus
from app.rag.offline import build_entity_index, build_keyword_index, build_vector_index

logger = logging.getLogger(__name__)


def _build_mock_graph() -> None:
    """Populate the in-memory graph store from the static knowledge base."""
    store = get_graph_store()
    store.reset()
    for entity in all_entities():
        store.upsert_entity(entity)
    written = sum(store.upsert_relation(rel) for rel in all_relations())
    logger.info("mock graph: %d entities, %d relations", len(all_entities()), written)


def mock_bootstrap() -> None:
    """Build the full in-memory knowledge base for mock mode."""
    logger.info("mock mode: building in-memory knowledge base...")
    chunks = chunk_corpus()
    entities = config_item_entities()

    # Build the IDF table first so all mock embeddings are discriminative.
    build_idf([c.text for c in chunks] + [e.embedding_text() for e in entities])

    build_vector_index(chunks)
    build_keyword_index(chunks)
    _build_mock_graph()
    build_entity_index(entities)
    logger.info("mock mode: knowledge base ready (%d chunks)", len(chunks))
