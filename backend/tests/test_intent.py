"""Unit tests for rule-based intent classification (no LLM call needed)."""
from __future__ import annotations

from app.intent.classifier import (
    INTENT_EXPLAIN,
    INTENT_GENERATE,
    INTENT_INVALID,
    INTENT_OPTIMIZE,
    _rule_classify,
)


def test_explain_intent():
    result = _rule_classify("股票型基金配置是什么意思")
    assert result.intent == INTENT_EXPLAIN
    assert result.confidence >= 0.7


def test_generate_intent():
    result = _rule_classify("帮我给一个保守型客户配一套投资组合方案")
    assert result.intent == INTENT_GENERATE
    assert result.confidence >= 0.7


def test_optimize_intent():
    result = _rule_classify("在现有方案基础上优化一下,降低风险")
    assert result.intent == INTENT_OPTIMIZE
    assert result.confidence >= 0.7


def test_invalid_out_of_domain():
    result = _rule_classify("今天天气怎么样")
    assert result.intent == INTENT_INVALID
    assert not result.is_valid


def test_invalid_too_short():
    assert _rule_classify("嗯").intent == INTENT_INVALID
