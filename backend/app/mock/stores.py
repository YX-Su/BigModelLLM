"""In-memory stand-ins for Milvus / Elasticsearch / Neo4j / Redis.

Each class mirrors the method surface of its real counterpart, so the
retrieval, generation and pipeline code runs unchanged in mock mode.
"""
from __future__ import annotations

import math
from typing import Any

from app.graph.schema import EXPANDABLE_RELATIONS, RELATION_TYPES
from app.schemas.dialogue import DialogueTurn, SessionState
from app.schemas.knowledge import Entity, Relation


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _bigrams(text: str) -> set[str]:
    cleaned = "".join(text.split())
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _overlap(query_bigrams: set[str], doc_bigrams: set[str]) -> float:
    if not query_bigrams or not doc_bigrams:
        return 0.0
    return len(query_bigrams & doc_bigrams) / len(query_bigrams)


class InMemoryVectorStore:
    """Brute-force cosine search over in-memory vectors."""

    def __init__(self) -> None:
        self.collections: dict[str, list[dict[str, Any]]] = {}

    def reset_collection(self, name: str) -> None:
        self.collections[name] = []

    def upsert(self, name: str, rows: list[dict[str, Any]]) -> None:
        self.collections.setdefault(name, []).extend(rows)

    def search(
        self, name: str, vector: list[float], top_k: int, output_fields: list[str]
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.collections.get(name, []):
            score = _cosine(vector, row["vector"])
            hit: dict[str, Any] = {"id": row["id"], "score": score}
            for field in output_fields:
                if field in row:
                    hit[field] = row[field]
            scored.append((score, hit))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in scored[:top_k]]


class InMemoryKeywordStore:
    """Bigram-overlap scoring as an in-memory stand-in for BM25."""

    def __init__(self) -> None:
        self.indices: dict[str, list[dict[str, Any]]] = {}

    def reset_index(self, name: str) -> None:
        self.indices[name] = []

    def bulk_index(self, name: str, docs: list[dict[str, Any]]) -> None:
        self.indices.setdefault(name, []).extend(docs)

    def search(self, name: str, query: str, top_k: int) -> list[dict[str, Any]]:
        query_bigrams = _bigrams(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.indices.get(name, []):
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            score = _overlap(query_bigrams, _bigrams(text))
            if score <= 0:
                continue
            hit = {"id": doc["id"], "score": score}
            hit.update({k: v for k, v in doc.items() if k != "id"})
            scored.append((score, hit))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in scored[:top_k]]


class InMemoryGraphStore:
    """In-memory entity / relation graph with BFS-based hop expansion."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []

    def close(self) -> None:
        pass

    def reset(self) -> None:
        self.entities = {}
        self.relations = []

    def upsert_entity(self, entity: Entity) -> None:
        self.entities[entity.entity_id] = entity

    def upsert_relation(self, relation: Relation) -> bool:
        if relation.type not in RELATION_TYPES:
            return False
        if relation.source not in self.entities or relation.target not in self.entities:
            return False
        self.relations.append(relation)
        return True

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def all_relations(self, rel_type: str) -> list[Relation]:
        return [r for r in self.relations if r.type == rel_type]

    def config_item_ids(self) -> set[str]:
        return {eid for eid, e in self.entities.items() if e.type == "ConfigItem"}

    def expand(self, entity_ids: list[str], hop: int = 2) -> list[Relation]:
        if not entity_ids:
            return []
        hop = max(1, min(int(hop), 3))
        frontier = set(entity_ids)
        visited = set(entity_ids)
        collected: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()
        for _ in range(hop):
            next_frontier: set[str] = set()
            for rel in self.relations:
                if rel.type not in EXPANDABLE_RELATIONS:
                    continue
                if rel.source not in frontier and rel.target not in frontier:
                    continue
                key = (rel.source, rel.target, rel.type)
                if key not in seen:
                    seen.add(key)
                    collected.append(rel)
                for node in (rel.source, rel.target):
                    if node not in visited:
                        visited.add(node)
                        next_frontier.add(node)
            frontier = next_frontier
            if not frontier:
                break
        return collected


class InMemorySessionStore:
    """In-memory session state as an in-memory stand-in for Redis."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        state = self.sessions.get(session_id)
        if state is None:
            return SessionState(session_id=session_id)
        return state.model_copy(deep=True)

    def save_session(self, state: SessionState) -> None:
        self.sessions[state.session_id] = state.model_copy(deep=True)

    def append_turn(self, session_id: str, turn: DialogueTurn) -> SessionState:
        state = self.get_session(session_id)
        turn.turn_id = len(state.turns) + 1
        state.turns.append(turn)
        self.save_session(state)
        return state

    def cached_entity_ids(self, session_id: str) -> set[str]:
        return self.get_session(session_id).all_entity_ids()

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
