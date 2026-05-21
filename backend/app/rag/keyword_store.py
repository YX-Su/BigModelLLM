"""Elasticsearch BM25 keyword index.

The precise fallback layer of GraphRAG: it reliably hits config item names,
rule descriptions and parameter thresholds even when semantic recall drifts.
The ``cjk`` analyzer is used so Chinese text is tokenised into bigrams without
requiring an external analysis plugin.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from elasticsearch import Elasticsearch, helpers

from app.config import settings

logger = logging.getLogger(__name__)

_MAPPING = {
    "properties": {
        "text": {"type": "text", "analyzer": "cjk"},
        "title": {"type": "text", "analyzer": "cjk"},
        "doc": {"type": "keyword"},
        "section": {"type": "keyword"},
        "config_codes": {"type": "keyword"},
    }
}


class KeywordStore:
    """Thin wrapper over the Elasticsearch client for BM25 retrieval."""

    def __init__(self) -> None:
        self.client = Elasticsearch(settings.es_url)

    def reset_index(self, name: str) -> None:
        """Drop and recreate the index with the CJK-aware mapping."""
        if self.client.indices.exists(index=name):
            self.client.indices.delete(index=name)
        self.client.indices.create(index=name, mappings=_MAPPING)
        logger.info("elasticsearch index %s recreated", name)

    def bulk_index(self, name: str, docs: list[dict[str, Any]]) -> None:
        """Bulk index documents. Each doc must carry an ``id`` key."""
        if not docs:
            return
        actions = [
            {"_index": name, "_id": doc["id"], "_source": {k: v for k, v in doc.items() if k != "id"}}
            for doc in docs
        ]
        helpers.bulk(self.client, actions)
        self.client.indices.refresh(index=name)
        logger.info("elasticsearch index %s: indexed %d docs", name, len(docs))

    def search(self, name: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 search over text + title; returns hits with raw ``score``."""
        response = self.client.search(
            index=name,
            size=top_k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "text"],
                }
            },
        )
        hits: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            hits.append({"id": hit["_id"], "score": float(hit["_score"]), **hit["_source"]})
        return hits


@lru_cache
def get_keyword_store():  # type: ignore[no-untyped-def]
    """Return the Elasticsearch-backed store, or the in-memory store in mock mode."""
    if settings.mock_mode:
        from app.mock.stores import InMemoryKeywordStore

        return InMemoryKeywordStore()
    return KeywordStore()
