"""Unit tests for related-turn selection (pure functions, no Redis needed)."""
from __future__ import annotations

from app.cache.context import build_history_context, select_related_turns
from app.schemas.dialogue import DialogueTurn, SessionState


def _state(*questions: str) -> SessionState:
    turns = [
        DialogueTurn(turn_id=i + 1, question=q, answer=f"对{q}的回答")
        for i, q in enumerate(questions)
    ]
    return SessionState(session_id="s1", turns=turns)


def test_no_history_returns_empty():
    assert select_related_turns(SessionState(session_id="s1"), "任意问题") == []


def test_related_turn_is_selected():
    state = _state("股票型基金配置的作用是什么", "货币基金怎么买")
    related = select_related_turns(state, "股票型基金配置可以设置止盈吗")
    assert any("股票型基金" in t.question for t in related)


def test_unrelated_query_keeps_last_turn_for_coherence():
    state = _state("货币基金怎么买", "债券基金的风险")
    related = select_related_turns(state, "完全无关的天气问题")
    assert len(related) == 1
    assert related[0].turn_id == 2


def test_history_context_condenses_long_answers():
    turns = [DialogueTurn(turn_id=1, question="问题", answer="答" * 300)]
    context = build_history_context(turns)
    assert "第1轮" in context
    assert "…" in context
