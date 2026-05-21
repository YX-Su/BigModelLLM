"""Session endpoints — inspect and reset multi-turn dialogue state."""
from __future__ import annotations

from fastapi import APIRouter

from app.cache.session_store import get_session_store
from app.schemas.dialogue import SessionState

router = APIRouter(prefix="/session", tags=["session"])


@router.get("/{session_id}", response_model=SessionState)
def get_session(session_id: str) -> SessionState:
    """Return the stored dialogue state for a session."""
    return get_session_store().get_session(session_id)


@router.delete("/{session_id}")
def clear_session(session_id: str) -> dict[str, str]:
    """Clear a session's cached dialogue state."""
    get_session_store().clear(session_id)
    return {"status": "cleared", "session_id": session_id}
