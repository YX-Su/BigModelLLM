"""Redis-backed context cache for multi-turn dialogue state.

Each session is stored as a single JSON blob keyed by ``session:<id>`` with a
TTL. The cache maintains the intermediate state of a conversation — questions,
answers, hit entities and touched config items — so later turns can reuse it.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import redis

from app.config import settings
from app.schemas.dialogue import DialogueTurn, SessionState

logger = logging.getLogger(__name__)


class SessionStore:
    """Thin wrapper over Redis for session-scoped dialogue state."""

    def __init__(self) -> None:
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    def get_session(self, session_id: str) -> SessionState:
        """Load a session, returning a fresh empty state when absent."""
        raw = self.client.get(self._key(session_id))
        if not raw:
            return SessionState(session_id=session_id)
        return SessionState.model_validate_json(raw)

    def save_session(self, state: SessionState) -> None:
        """Persist a session with a sliding TTL."""
        self.client.set(
            self._key(state.session_id),
            state.model_dump_json(),
            ex=settings.session_ttl,
        )

    def append_turn(self, session_id: str, turn: DialogueTurn) -> SessionState:
        """Append a completed turn, assigning it the next turn id."""
        state = self.get_session(session_id)
        turn.turn_id = len(state.turns) + 1
        state.turns.append(turn)
        self.save_session(state)
        logger.info("session %s: stored turn %d", session_id, turn.turn_id)
        return state

    def cached_entity_ids(self, session_id: str) -> set[str]:
        """Entities already expanded in earlier turns — used for retrieval pruning."""
        return self.get_session(session_id).all_entity_ids()

    def clear(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore()
