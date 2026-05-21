"""Knowledge graph construction orchestration.

Builds the Neo4j graph from knowledge base chunks:

1. seed the controlled-vocabulary Dimension and Effect entities;
2. extract ConfigItem entities from config item chunks (LLM);
3. link each ConfigItem to its Dimension (BELONGS_TO);
4. extract dependency / mutex / influence relations from rule chunks (LLM).
"""
from __future__ import annotations

import logging

from app.graph.extraction import extract_config_item, extract_relations
from app.graph.schema import DIMENSIONS, EFFECTS
from app.graph.store import GraphStore
from app.schemas.knowledge import Chunk, Entity, Relation

logger = logging.getLogger(__name__)

CONFIG_ITEM_DOC = "01_config_items"
RULES_DOC = "02_rules"


def build_knowledge_graph(chunks: list[Chunk], store: GraphStore) -> list[Entity]:
    """Build the full knowledge graph and return the ConfigItem entities.

    The returned entities feed the entity vector index (the entity linking
    layer of GraphRAG).
    """
    store.reset()

    # 1. controlled-vocabulary entities
    for dim in DIMENSIONS:
        store.upsert_entity(Entity(entity_id=dim, name=dim, type="Dimension"))
    for effect in EFFECTS:
        store.upsert_entity(Entity(entity_id=effect, name=effect, type="Effect"))
    logger.info("seeded %d dimensions, %d effects", len(DIMENSIONS), len(EFFECTS))

    # 2-3. config item entities + BELONGS_TO edges
    entities: list[Entity] = []
    for chunk in (c for c in chunks if c.doc == CONFIG_ITEM_DOC):
        entity = extract_config_item(chunk)
        store.upsert_entity(entity)
        entities.append(entity)
        if entity.dimension in DIMENSIONS:
            store.upsert_relation(
                Relation(source=entity.entity_id, target=entity.dimension, type="BELONGS_TO")
            )
    logger.info("extracted %d config item entities", len(entities))

    # 4. dependency / mutex / influence relations
    relations: list[Relation] = []
    for chunk in (c for c in chunks if c.doc == RULES_DOC):
        relations.extend(extract_relations(chunk))

    written = sum(store.upsert_relation(rel) for rel in relations)
    logger.info("extracted %d relations, %d written to graph", len(relations), written)

    return entities
