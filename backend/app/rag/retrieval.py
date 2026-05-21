"""Online GraphRAG retrieval pipeline.

Two stages:

* **Stage 1 — entity recognition & graph expansion.** Aligns the user query
  to standard graph entities via the entity vector index, then expands the
  knowledge graph by 1~2 hops to obtain a structured, explainable subgraph.
* **Stage 2 — subgraph-driven hybrid retrieval.** Uses the subgraph entity
  descriptions as queries for vector + BM25 recall, fuses the candidates with
  weighted ranking, and filters by a similarity threshold.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.core.llm import embed, embed_one
from app.graph.store import get_graph_store
from app.rag.keyword_store import get_keyword_store
from app.rag.vector_store import (
    CHUNK_OUTPUT_FIELDS,
    ENTITY_OUTPUT_FIELDS,
    get_vector_store,
)
from app.schemas.knowledge import Entity, Relation
from app.schemas.retrieval import RetrievalResult, ScoredChunk

logger = logging.getLogger(__name__)

# Cap on how many subgraph entities expand the stage-2 query set.
_MAX_QUERY_EXPANSION = 10

# Relative weight of the user query vs. subgraph-expansion queries in fusion.
_PRIMARY_WEIGHT = 0.7
_EXPANSION_WEIGHT = 0.3


def link_entities(query: str) -> list[str]:
    """Stage 1a — map the query to standard graph entities via vector search."""
    query_vector = embed_one(query)
    hits = get_vector_store().search(
        settings.milvus_entity_collection,
        query_vector,
        settings.entity_top_k,
        ENTITY_OUTPUT_FIELDS,
    )
    return [hit["id"] for hit in hits]


def expand_subgraph(seed_ids: list[str]) -> tuple[list[Entity], list[Relation]]:
    """Stage 1b — expand seed entities 1~2 hops into an entity subgraph."""
    graph = get_graph_store()
    relations = graph.expand(seed_ids, settings.graph_hop)

    entity_ids: set[str] = set(seed_ids)
    for relation in relations:
        entity_ids.add(relation.source)
        entity_ids.add(relation.target)

    entities = [e for e in (graph.get_entity(eid) for eid in entity_ids) if e]
    return entities, relations


def _norm(value: float, lo: float, hi: float) -> float:
    """Min-max normalisation; collapses to 1.0 when the range is degenerate."""
    if hi <= lo:
        return 1.0 if value > 0 else 0.0
    return (value - lo) / (hi - lo)


def _hybrid_retrieve(query: str, entities: list[Entity]) -> list[ScoredChunk]:
    """Stage 2 — subgraph-driven vector + BM25 recall with weighted fusion.

    The user query is the primary retrieval signal; subgraph entity
    descriptions provide supplementary expansion at a lower weight, so a
    chunk that merely self-matches an expanded entity cannot outrank a chunk
    that is genuinely relevant to the user's question.
    """
    config_entities = [e for e in entities if e.type == "ConfigItem"]
    expansion_texts = [
        f"{e.name} {e.description}".strip()
        for e in config_entities[:_MAX_QUERY_EXPANSION]
    ]

    vector_store = get_vector_store()
    keyword_store = get_keyword_store()
    pool: dict[str, dict[str, Any]] = {}

    def _slot(hit: dict[str, Any]) -> dict[str, Any]:
        return pool.setdefault(
            hit["id"],
            {
                "chunk_id": hit["id"],
                "title": hit.get("title", ""),
                "text": hit.get("text", ""),
                "doc": hit.get("doc", ""),
                "primary_vec": 0.0,
                "expansion_vec": 0.0,
                "bm25_score": 0.0,
            },
        )

    # Primary pass — the user query itself.
    for hit in vector_store.search(
        settings.milvus_chunk_collection, embed_one(query), settings.chunk_top_k, CHUNK_OUTPUT_FIELDS
    ):
        slot = _slot(hit)
        slot["primary_vec"] = max(slot["primary_vec"], hit["score"])
    for hit in keyword_store.search(settings.es_chunk_index, query, settings.chunk_top_k):
        slot = _slot(hit)
        slot["bm25_score"] = max(slot["bm25_score"], hit["score"])

    # Expansion pass — subgraph entity descriptions (supplementary signal).
    if expansion_texts:
        for vector in embed(expansion_texts):
            for hit in vector_store.search(
                settings.milvus_chunk_collection, vector, settings.chunk_top_k, CHUNK_OUTPUT_FIELDS
            ):
                slot = _slot(hit)
                slot["expansion_vec"] = max(slot["expansion_vec"], hit["score"])
        for text in expansion_texts:
            for hit in keyword_store.search(settings.es_chunk_index, text, settings.chunk_top_k):
                slot = _slot(hit)
                slot["bm25_score"] = max(slot["bm25_score"], hit["score"])

    if not pool:
        return []

    # Weighted fusion over min-max normalised channel scores.
    primary_values = [c["primary_vec"] for c in pool.values()]
    expansion_values = [c["expansion_vec"] for c in pool.values()]
    bm25_values = [c["bm25_score"] for c in pool.values()]
    p_lo, p_hi = min(primary_values), max(primary_values)
    e_lo, e_hi = min(expansion_values), max(expansion_values)
    b_lo, b_hi = min(bm25_values), max(bm25_values)

    scored: list[ScoredChunk] = []
    for cand in pool.values():
        vec_component = _PRIMARY_WEIGHT * _norm(cand["primary_vec"], p_lo, p_hi) + (
            _EXPANSION_WEIGHT * _norm(cand["expansion_vec"], e_lo, e_hi)
        )
        fused = settings.vector_weight * vec_component + (
            settings.bm25_weight * _norm(cand["bm25_score"], b_lo, b_hi)
        )
        scored.append(
            ScoredChunk(
                chunk_id=cand["chunk_id"],
                title=cand["title"],
                text=cand["text"],
                doc=cand["doc"],
                score=round(fused, 4),
                vec_score=round(cand["primary_vec"], 4),
                bm25_score=round(cand["bm25_score"], 4),
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    filtered = [c for c in scored if c.score >= settings.similarity_threshold]
    return filtered[: settings.chunk_top_k]


def retrieve_baseline(query: str) -> list[ScoredChunk]:
    """Pre-GraphRAG baseline: plain vector recall over chunks, no graph.

    Used by the evaluation harness to quantify the recall gain from adding
    the knowledge graph (entity linking + subgraph expansion).
    """
    query_vector = embed_one(query)
    hits = get_vector_store().search(
        settings.milvus_chunk_collection,
        query_vector,
        settings.chunk_top_k,
        CHUNK_OUTPUT_FIELDS,
    )
    return [
        ScoredChunk(
            chunk_id=hit["id"],
            title=hit.get("title", ""),
            text=hit.get("text", ""),
            doc=hit.get("doc", ""),
            score=round(hit["score"], 4),
            vec_score=round(hit["score"], 4),
        )
        for hit in hits
    ]


def retrieve(query: str, exclude_entities: set[str] | None = None) -> RetrievalResult:
    """Run the full two-stage GraphRAG retrieval pipeline for a user query.

    ``exclude_entities`` lets the context cache prune entities already
    expanded in earlier dialogue turns, reducing redundant retrieval. If
    pruning would remove every seed (a follow-up about an already-seen
    entity), the full seed set is kept so retrieval still has signal.
    """
    seed_ids = link_entities(query)
    if exclude_entities:
        pruned = [eid for eid in seed_ids if eid not in exclude_entities]
        if pruned:
            logger.info("cache pruning: %d -> %d seed entities", len(seed_ids), len(pruned))
            seed_ids = pruned

    entities, relations = expand_subgraph(seed_ids)
    chunks = _hybrid_retrieve(query, entities)

    logger.info(
        "retrieve: query=%r seeds=%d subgraph=%d/%d chunks=%d",
        query,
        len(seed_ids),
        len(entities),
        len(relations),
        len(chunks),
    )
    return RetrievalResult(
        query=query,
        seed_entity_ids=seed_ids,
        entities=entities,
        relations=relations,
        chunks=chunks,
    )
