"""Lightweight intent recognition.

Classifies a user query into one of four task types before the full
retrieval + reasoning pipeline runs. A fast rule-based pass handles the
clear-cut cases (including filtering out-of-scope questions cheaply); only
ambiguous queries fall through to an LLM few-shot classifier.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel

from app.core.llm import chat_json

logger = logging.getLogger(__name__)

INTENT_EXPLAIN = "explain"
INTENT_GENERATE = "generate"
INTENT_OPTIMIZE = "optimize"
INTENT_INVALID = "invalid"

INTENT_META: dict[str, dict[str, str]] = {
    INTENT_EXPLAIN: {
        "label": "效果解释",
        "description": "解释配置项或规则的定义、适用范围与组合效果",
    },
    INTENT_GENERATE: {
        "label": "方案生成",
        "description": "根据业务目标与约束生成一套新的投资组合配置方案",
    },
    INTENT_OPTIMIZE: {
        "label": "方案优化",
        "description": "在已有配置方案的基础上进行调整与优化",
    },
    INTENT_INVALID: {
        "label": "无效问题",
        "description": "超出投资组合配置领域或系统能力范围的问题",
    },
}

INVALID_GUIDANCE = (
    "抱歉,我是投资组合配置助手,只能解答与投资组合 / 保险方案配置相关的问题。"
    "你可以问我:某个配置项的作用、配置项之间的依赖与互斥关系,"
    "或让我根据客户的风险偏好与预算生成一套配置方案。"
)

# Domain vocabulary — a query touching none of these is treated as out of scope.
_DOMAIN_KEYWORDS = (
    "基金", "配置", "组合", "投资", "理财", "风险", "收益", "止盈", "止损", "定投",
    "杠杆", "保险", "债券", "股票", "货币", "黄金", "QDII", "养老", "再平衡", "仓位",
    "客户", "方案", "资产", "流动性", "偏好", "保守", "进取", "平衡", "期限", "分红",
    "年金", "回撤", "配比",
)

_OPTIMIZE_KEYWORDS = (
    "优化", "调整", "修改", "改成", "改为", "已有", "现有", "现在的", "基础上",
    "替换", "升级", "降低风险", "再加", "去掉",
)
_GENERATE_KEYWORDS = (
    "生成", "推荐", "设计", "配一个", "配一套", "做一个", "做一套", "怎么配",
    "帮我配", "制定", "出一套", "出一个", "给一套", "给个方案", "搭一个",
)
_EXPLAIN_KEYWORDS = (
    "是什么", "什么意思", "含义", "作用", "解释", "为什么", "区别", "能不能",
    "可以吗", "是否", "如何理解", "影响", "怎么理解", "啥意思", "有什么用",
)


class IntentResult(BaseModel):
    """The recognised intent of a user query."""

    intent: str
    confidence: float
    reason: str = ""

    @property
    def label(self) -> str:
        return INTENT_META.get(self.intent, {}).get("label", self.intent)

    @property
    def is_valid(self) -> bool:
        return self.intent != INTENT_INVALID


def _rule_classify(query: str) -> IntentResult:
    """Fast keyword-based classification; low confidence defers to the LLM."""
    text = query.strip()
    if len(text) < 4:
        return IntentResult(intent=INTENT_INVALID, confidence=0.85, reason="问题过短")
    if not any(kw in text for kw in _DOMAIN_KEYWORDS):
        return IntentResult(
            intent=INTENT_INVALID, confidence=0.85, reason="未涉及投资组合配置领域"
        )

    scores = {
        INTENT_OPTIMIZE: sum(kw in text for kw in _OPTIMIZE_KEYWORDS),
        INTENT_GENERATE: sum(kw in text for kw in _GENERATE_KEYWORDS),
        INTENT_EXPLAIN: sum(kw in text for kw in _EXPLAIN_KEYWORDS),
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return IntentResult(
            intent=INTENT_EXPLAIN, confidence=0.5, reason="领域问题但意图线索不足"
        )
    confidence = 0.7 if scores[best] == 1 else 0.85
    return IntentResult(intent=best, confidence=confidence, reason="命中意图关键词")


_LLM_PROMPT = """你是投资组合配置助手的意图识别模块。请将用户问题分类为以下之一:
- explain: 解释配置项 / 规则的定义或组合效果
- generate: 根据业务目标生成一套新的配置方案
- optimize: 在已有配置方案基础上进行调整优化
- invalid: 与投资组合 / 保险配置无关,或超出系统能力范围

示例:
- "股票型基金配置是什么" -> explain
- "给一个保守型客户配一套理财组合" -> generate
- "现在的方案波动太大,帮我调一下" -> optimize
- "帮我写一首诗" -> invalid

仅输出 JSON:{{"intent": "...", "reason": "..."}}

用户问题:{query}"""


def _llm_classify(query: str) -> IntentResult:
    """LLM few-shot classification for ambiguous queries."""
    data = chat_json([{"role": "user", "content": _LLM_PROMPT.format(query=query)}])
    intent = str(data.get("intent", "")).strip().lower()
    if intent not in INTENT_META:
        intent = INTENT_EXPLAIN
    return IntentResult(
        intent=intent, confidence=0.8, reason=str(data.get("reason", "LLM 分类"))
    )


def classify(query: str) -> IntentResult:
    """Recognise the intent of a query (rule-based fast path + LLM fallback)."""
    rule_result = _rule_classify(query)
    if rule_result.confidence >= 0.7:
        logger.info("intent(rule): %s -> %s", query, rule_result.intent)
        return rule_result
    try:
        llm_result = _llm_classify(query)
        logger.info("intent(llm): %s -> %s", query, llm_result.intent)
        return llm_result
    except Exception as exc:  # noqa: BLE001 - fall back to the rule result
        logger.warning("llm intent classification failed (%s), using rule result", exc)
        return rule_result
