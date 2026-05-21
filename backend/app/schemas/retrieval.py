"""Data models for the online GraphRAG retrieval result."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.knowledge import Entity, Relation


class ScoredChunk(BaseModel):
    """A knowledge base chunk with its hybrid retrieval scores."""

    chunk_id: str
    title: str = ""
    text: str
    doc: str = ""
    score: float = 0.0
    vec_score: float = 0.0
    bm25_score: float = 0.0


class RetrievalResult(BaseModel):
    """Output of the two-stage GraphRAG retrieval pipeline.

    Carries everything needed to build the LLM generation context: the
    high-confidence reference chunks plus the entity subgraph triples.
    """

    query: str
    seed_entity_ids: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    chunks: list[ScoredChunk] = Field(default_factory=list)

    def _entity_name(self, entity_id: str) -> str:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return f"{entity.name}({entity_id})" if entity.name != entity_id else entity_id
        return entity_id

    def reference_context(self) -> str:
        """Render the retrieved chunks as a numbered reference block."""
        return "\n\n".join(
            f"[参考{i + 1}] {chunk.text}" for i, chunk in enumerate(self.chunks)
        )

    def triples_context(self) -> str:
        """Render the entity subgraph as human-readable relation triples."""
        verb = {
            "BELONGS_TO": "属于",
            "DEPENDS_ON": "依赖",
            "MUTEX": "互斥于",
            "ENHANCES": "增强",
            "WEAKENS": "减弱",
        }
        lines = []
        for rel in self.relations:
            tag = f"[{rel.rule_id}] " if rel.rule_id else ""
            lines.append(
                f"- {tag}{self._entity_name(rel.source)} "
                f"{verb.get(rel.type, rel.type)} {self._entity_name(rel.target)}"
            )
        return "\n".join(lines)
