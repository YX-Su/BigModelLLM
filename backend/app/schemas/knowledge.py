"""Core knowledge data models shared by the offline pipeline and online retrieval."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A semantic unit of the knowledge base (one config item / rule / solution)."""

    chunk_id: str
    doc: str
    section: str = ""
    title: str = ""
    text: str
    config_codes: list[str] = Field(default_factory=list)


class Entity(BaseModel):
    """A knowledge graph entity: a config item, a dimension or a business effect."""

    entity_id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    dimension: str = ""

    def embedding_text(self) -> str:
        """Text used to build the entity vector index (entity linking layer)."""
        parts = [self.name, *self.aliases]
        if self.description:
            parts.append(self.description)
        if self.dimension:
            parts.append(self.dimension)
        return " | ".join(p for p in parts if p)


class Relation(BaseModel):
    """A directed knowledge graph relation between two entities."""

    source: str
    target: str
    type: str
    rule_id: str = ""

    def as_triple(self) -> str:
        """Human-readable triple used as part of the LLM generation context."""
        verb = {
            "BELONGS_TO": "属于",
            "DEPENDS_ON": "依赖",
            "MUTEX": "与之互斥",
            "ENHANCES": "增强",
            "WEAKENS": "减弱",
        }.get(self.type, self.type)
        return f"({self.source}) {verb} ({self.target})"
