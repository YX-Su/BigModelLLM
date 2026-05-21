"""Milvus vector store.

Hosts two collections built by the offline pipeline:

* the **chunk collection** — semantic index over knowledge base chunks
  (semantic recall layer);
* the **entity collection** — vector index over knowledge graph entities
  (entity linking layer, the first stage of GraphRAG).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from pymilvus import DataType, MilvusClient

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper over :class:`MilvusClient` for collection-keyed access."""

    def __init__(self) -> None:
        self.client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    def reset_collection(self, name: str) -> None:
        """Drop and recreate a collection with a string PK + HNSW cosine index."""
        if self.client.has_collection(name):
            self.client.drop_collection(name)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("milvus collection %s recreated", name)

    def upsert(self, name: str, rows: list[dict[str, Any]]) -> None:
        """Insert rows. Each row must carry ``id``, ``vector`` and metadata keys."""
        if not rows:
            return
        self.client.insert(collection_name=name, data=rows)
        logger.info("milvus collection %s: upserted %d rows", name, len(rows))

    def search(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        output_fields: list[str],
    ) -> list[dict[str, Any]]:
        """Cosine similarity search; returns hits with ``id``, ``score`` and fields."""
        results = self.client.search(
            collection_name=name,
            data=[vector],
            limit=top_k,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE"},
        )
        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            hits.append({"id": hit["id"], "score": float(hit["distance"]), **entity})
        return hits


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
