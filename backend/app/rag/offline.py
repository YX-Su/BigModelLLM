"""Offline preparation pipeline.

Runs the full offline build of the GraphRAG system in one pass:

1. semantic chunking of the raw knowledge base;
2. chunk vector index (Milvus, semantic recall layer);
3. chunk BM25 index (Elasticsearch, precise fallback layer);
4. knowledge graph (Neo4j, structured reasoning layer);
5. entity vector index (Milvus, entity linking layer).
"""
from __future__ import annotations

import json
import logging

from app.config import settings
from app.core.llm import embed
from app.graph.builder import build_knowledge_graph
from app.graph.store import get_graph_store
from app.rag.chunking import chunk_corpus
from app.rag.keyword_store import get_keyword_store
from app.rag.vector_store import get_vector_store
from app.schemas.knowledge import Chunk, Entity

logger = logging.getLogger(__name__)

_EMBED_BATCH = 32


def _batch_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches to stay within provider request limits."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        vectors.extend(embed(texts[start : start + _EMBED_BATCH]))
    return vectors


def build_vector_index(chunks: list[Chunk]) -> None:
    """Step 2 — embed chunks and load the Milvus chunk collection."""
    store = get_vector_store()
    store.reset_collection(settings.milvus_chunk_collection)
    vectors = _batch_embed([c.text for c in chunks])
    rows = [
        {
            "id": chunk.chunk_id,
            "vector": vector,
            "text": chunk.text,
            "title": chunk.title,
            "doc": chunk.doc,
            "section": chunk.section,
            "config_codes": chunk.config_codes,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    store.upsert(settings.milvus_chunk_collection, rows)


def build_keyword_index(chunks: list[Chunk]) -> None:
    """Step 3 — load chunks into the Elasticsearch BM25 index."""
    store = get_keyword_store()
    store.reset_index(settings.es_chunk_index)
    docs = [
        {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "title": chunk.title,
            "doc": chunk.doc,
            "section": chunk.section,
            "config_codes": chunk.config_codes,
        }
        for chunk in chunks
    ]
    store.bulk_index(settings.es_chunk_index, docs)


def build_entity_index(entities: list[Entity]) -> None:
    """Step 5 — embed entities and load the Milvus entity collection."""
    store = get_vector_store()
    store.reset_collection(settings.milvus_entity_collection)
    vectors = _batch_embed([e.embedding_text() for e in entities])
    rows = [
        {
            "id": entity.entity_id,
            "vector": vector,
            "name": entity.name,
            "type": entity.type,
            "dimension": entity.dimension,
            "description": entity.description,
        }
        for entity, vector in zip(entities, vectors)
    ]
    store.upsert(settings.milvus_entity_collection, rows)


def build_all() -> None:
    """Run the complete offline preparation pipeline."""
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/5] semantic chunking")
    chunks = chunk_corpus()
    (settings.processed_data_dir / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("chunked corpus into %d chunks", len(chunks))

    logger.info("[2/5] building chunk vector index (Milvus)")
    build_vector_index(chunks)

    logger.info("[3/5] building chunk keyword index (Elasticsearch)")
    build_keyword_index(chunks)

    logger.info("[4/5] building knowledge graph (Neo4j)")
    entities = build_knowledge_graph(chunks, get_graph_store())

    logger.info("[5/5] building entity vector index (Milvus)")
    build_entity_index(entities)

    logger.info("offline build complete: %d chunks, %d entities", len(chunks), len(entities))


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    build_all()
