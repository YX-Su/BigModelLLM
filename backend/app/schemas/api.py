"""Request / response models for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.generation import ValidationReport
from app.schemas.knowledge import Relation
from app.schemas.retrieval import ScoredChunk


class ChatRequest(BaseModel):
    """A single chat request within a session."""

    session_id: str = "default"
    query: str


class ChatResponse(BaseModel):
    """The assistant reply plus a trace of the GraphRAG inference pipeline."""

    session_id: str
    intent: str
    intent_label: str
    answer: str
    config_plan: list[str] = Field(default_factory=list)
    validation: ValidationReport | None = None
    seed_entities: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    references: list[ScoredChunk] = Field(default_factory=list)
