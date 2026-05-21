"""Dialogue state models for multi-turn conversation management."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DialogueTurn(BaseModel):
    """One question/answer round, plus the entities and config items it touched."""

    turn_id: int = 0
    question: str
    answer: str = ""
    intent: str = ""
    hit_entities: list[str] = Field(default_factory=list)
    config_codes: list[str] = Field(default_factory=list)


class SessionState(BaseModel):
    """The full conversation state for one session, persisted in Redis."""

    session_id: str
    turns: list[DialogueTurn] = Field(default_factory=list)

    def all_entity_ids(self) -> set[str]:
        """Union of every entity hit across the conversation so far."""
        seen: set[str] = set()
        for turn in self.turns:
            seen.update(turn.hit_entities)
        return seen

    def all_config_codes(self) -> set[str]:
        """Union of every config item touched across the conversation so far."""
        seen: set[str] = set()
        for turn in self.turns:
            seen.update(turn.config_codes)
        return seen
