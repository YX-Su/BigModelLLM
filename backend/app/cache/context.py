"""Related-turn selection and condensed history context assembly.

Within a session, a follow-up question is judged against earlier turns; only
related turns are condensed and concatenated into a compact context for the
LLM, so multi-turn coherence is kept without unbounded context growth.
"""
from __future__ import annotations

from app.schemas.dialogue import DialogueTurn, SessionState

# Char-bigram Jaccard above this counts as "related to the current query".
_RELATED_THRESHOLD = 0.08
_MAX_RELATED_TURNS = 3
_ANSWER_SUMMARY_LEN = 120


def _bigrams(text: str) -> set[str]:
    cleaned = "".join(text.split())
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _overlap(a: str, b: str) -> float:
    """Char-bigram Jaccard similarity between two Chinese strings."""
    ba, bb = _bigrams(a), _bigrams(b)
    union = ba | bb
    if not union:
        return 0.0
    return len(ba & bb) / len(union)


def select_related_turns(state: SessionState, query: str) -> list[DialogueTurn]:
    """Pick the historical turns related to the current query.

    A turn is related when it has enough lexical overlap with the query. When
    nothing scores as related the most recent turn is still kept, so generic
    follow-ups (e.g. "那它呢") retain conversational coherence.
    """
    if not state.turns:
        return []

    related = [
        turn
        for turn in state.turns
        if _overlap(query, turn.question + turn.answer) >= _RELATED_THRESHOLD
    ]
    if not related:
        related = state.turns[-1:]
    return related[-_MAX_RELATED_TURNS:]


def build_history_context(turns: list[DialogueTurn]) -> str:
    """Condense related turns into a compact history block for the prompt."""
    if not turns:
        return ""
    blocks = []
    for turn in turns:
        answer = turn.answer.strip().replace("\n", " ")
        if len(answer) > _ANSWER_SUMMARY_LEN:
            answer = answer[:_ANSWER_SUMMARY_LEN] + "…"
        blocks.append(f"第{turn.turn_id}轮 用户:{turn.question}\n第{turn.turn_id}轮 助手:{answer}")
    return "\n".join(blocks)
