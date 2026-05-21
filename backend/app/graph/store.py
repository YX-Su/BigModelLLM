"""Neo4j knowledge graph store.

Persists entities and relations with Cypher and supports the 1~2 hop relation
expansion used by the first stage of online GraphRAG retrieval.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from neo4j import GraphDatabase

from app.config import settings
from app.graph.schema import ENTITY_TYPES, EXPANDABLE_RELATIONS, RELATION_TYPES
from app.schemas.knowledge import Entity, Relation

logger = logging.getLogger(__name__)


class GraphStore:
    """Thin wrapper over the Neo4j driver for entity / relation persistence."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def reset(self) -> None:
        """Delete the whole graph."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("neo4j graph cleared")

    def upsert_entity(self, entity: Entity) -> None:
        """Merge an entity node, keyed by stable ``entity_id``."""
        label = entity.type if entity.type in ENTITY_TYPES else "Entity"
        query = (
            f"MERGE (e:Entity {{entity_id: $entity_id}}) "
            f"SET e:{label}, e.name=$name, e.type=$type, "
            f"e.aliases=$aliases, e.description=$description, e.dimension=$dimension"
        )
        with self.driver.session() as session:
            session.run(
                query,
                entity_id=entity.entity_id,
                name=entity.name,
                type=entity.type,
                aliases=entity.aliases,
                description=entity.description,
                dimension=entity.dimension,
            )

    def upsert_relation(self, relation: Relation) -> bool:
        """Merge a relation edge between two existing entities.

        Returns ``False`` when either endpoint is missing so the offline
        pipeline can report relations that failed entity disambiguation.
        """
        if relation.type not in RELATION_TYPES:
            logger.warning("skip relation with unknown type: %s", relation.type)
            return False
        query = (
            "MATCH (s:Entity {entity_id: $source}) "
            "MATCH (t:Entity {entity_id: $target}) "
            f"MERGE (s)-[r:{relation.type}]->(t) "
            "SET r.rule_id=$rule_id "
            "RETURN r"
        )
        with self.driver.session() as session:
            result = session.run(
                query,
                source=relation.source,
                target=relation.target,
                rule_id=relation.rule_id,
            )
            created = result.single() is not None
        if not created:
            logger.warning(
                "relation %s-[%s]->%s skipped: endpoint missing",
                relation.source,
                relation.type,
                relation.target,
            )
        return created

    def get_entity(self, entity_id: str) -> Entity | None:
        """Fetch a single entity by id."""
        with self.driver.session() as session:
            record = session.run(
                "MATCH (e:Entity {entity_id: $id}) RETURN e", id=entity_id
            ).single()
        if not record:
            return None
        node = record["e"]
        return Entity(
            entity_id=node["entity_id"],
            name=node.get("name", node["entity_id"]),
            type=node.get("type", ""),
            aliases=node.get("aliases", []),
            description=node.get("description", ""),
            dimension=node.get("dimension", ""),
        )

    def expand(self, entity_ids: list[str], hop: int = 2) -> list[Relation]:
        """Return relations reachable within ``hop`` hops of the seed entities.

        Drives GraphRAG stage one: maps seed entities to a structured,
        explainable subgraph of dependency / mutex / influence relations.
        """
        if not entity_ids:
            return []
        hop = max(1, min(int(hop), 3))
        rel_filter = "|".join(EXPANDABLE_RELATIONS)
        query = (
            f"MATCH (e:Entity)-[rels:{rel_filter}*1..{hop}]-(:Entity) "
            "WHERE e.entity_id IN $ids "
            "UNWIND rels AS r "
            "RETURN DISTINCT startNode(r).entity_id AS source, "
            "type(r) AS rtype, endNode(r).entity_id AS target, r.rule_id AS rule_id"
        )
        with self.driver.session() as session:
            records = session.run(query, ids=entity_ids)
            return [
                Relation(
                    source=rec["source"],
                    target=rec["target"],
                    type=rec["rtype"],
                    rule_id=rec["rule_id"] or "",
                )
                for rec in records
            ]

    def all_relations(self, rel_type: str) -> list[Relation]:
        """Return every relation of a given type — used to load the rule set."""
        if rel_type not in RELATION_TYPES:
            raise ValueError(f"unknown relation type: {rel_type}")
        query = (
            f"MATCH (s:Entity)-[r:{rel_type}]->(t:Entity) "
            "RETURN s.entity_id AS source, t.entity_id AS target, r.rule_id AS rule_id"
        )
        with self.driver.session() as session:
            records = session.run(query)
            return [
                Relation(
                    source=rec["source"],
                    target=rec["target"],
                    type=rel_type,
                    rule_id=rec["rule_id"] or "",
                )
                for rec in records
            ]

    def config_item_ids(self) -> set[str]:
        """Return the ids of every ConfigItem entity in the graph."""
        with self.driver.session() as session:
            records = session.run("MATCH (e:ConfigItem) RETURN e.entity_id AS id")
            return {rec["id"] for rec in records}


@lru_cache
def get_graph_store():  # type: ignore[no-untyped-def]
    """Return the Neo4j-backed store, or the in-memory store in mock mode."""
    if settings.mock_mode:
        from app.mock.stores import InMemoryGraphStore

        return InMemoryGraphStore()
    return GraphStore()
