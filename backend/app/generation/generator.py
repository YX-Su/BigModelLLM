"""LLM generation stage.

Assembles the retrieval-augmented context, runs the intent-specific prompt
template through the LLM, and returns the answer plus — for plan-producing
intents — the structured config plan consumed by rule validation.
"""
from __future__ import annotations

import logging

from app.core.llm import chat, chat_json
from app.generation.context_builder import build_generation_context
from app.generation.templates import build_messages
from app.intent.classifier import INTENT_GENERATE, INTENT_OPTIMIZE
from app.schemas.generation import GenerationResult
from app.schemas.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

_PLAN_INTENTS = {INTENT_GENERATE, INTENT_OPTIMIZE}


def generate(
    query: str,
    intent: str,
    retrieval: RetrievalResult,
    history_context: str = "",
) -> GenerationResult:
    """Generate an answer for the query under the given intent."""
    context = build_generation_context(retrieval, history_context)
    messages = build_messages(intent, query, context)

    if intent in _PLAN_INTENTS:
        data = chat_json(messages)
        answer = str(data.get("answer", "")).strip()
        raw_plan = data.get("config_plan", [])
        plan = [str(code).strip() for code in raw_plan if str(code).strip()] if isinstance(raw_plan, list) else []
        logger.info("generate(%s): plan has %d config items", intent, len(plan))
        return GenerationResult(intent=intent, answer=answer, config_plan=plan)

    answer = chat(messages)
    return GenerationResult(intent=intent, answer=answer)
