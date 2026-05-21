"""Online inference pipeline orchestration.

Wires the four backend modules into one request flow:

    query
      -> intent recognition        (filter out-of-scope questions)
      -> context cache             (related history + cached entities)
      -> GraphRAG retrieval        (entity linking -> subgraph -> hybrid recall)
      -> context assembly + LLM    (intent-specific generation)
      -> rule-system validation    (dependency / mutex backstop)
      -> persist turn -> response
"""
from __future__ import annotations

import logging

from app.cache.context import build_history_context, select_related_turns
from app.cache.session_store import get_session_store
from app.generation.generator import generate
from app.generation.validator import cached_ruleset, validate
from app.intent.classifier import INVALID_GUIDANCE, classify
from app.rag.retrieval import retrieve
from app.schemas.api import ChatResponse
from app.schemas.dialogue import DialogueTurn

logger = logging.getLogger(__name__)


def run_pipeline(session_id: str, query: str) -> ChatResponse:
    """Execute the full online inference pipeline for one user query."""
    query = query.strip()

    # 1. Intent recognition — out-of-scope questions short-circuit here.
    intent = classify(query)
    if not intent.is_valid:
        logger.info("query rejected as invalid: %r", query)
        return ChatResponse(
            session_id=session_id,
            intent=intent.intent,
            intent_label=intent.label,
            answer=INVALID_GUIDANCE,
        )

    # 2. Context cache — related history + entities already expanded before.
    store = get_session_store()
    state = store.get_session(session_id)
    history_context = build_history_context(select_related_turns(state, query))
    cached_entities = state.all_entity_ids()

    # 3. GraphRAG retrieval (with cache-driven entity pruning).
    retrieval = retrieve(query, exclude_entities=cached_entities)

    # 4-5. Context assembly + intent-specific LLM generation.
    generation = generate(query, intent.intent, retrieval, history_context)

    # 6. Rule-system backstop for plan-producing intents.
    final_answer = generation.answer
    validation = None
    if generation.config_plan:
        validation = validate(generation.config_plan, cached_ruleset())
        final_answer = f"{generation.answer}\n\n---\n{validation.render_note()}"

    # 7. Persist the turn so later turns can reuse this context.
    store.append_turn(
        session_id,
        DialogueTurn(
            question=query,
            answer=final_answer,
            intent=intent.intent,
            hit_entities=retrieval.seed_entity_ids,
            config_codes=generation.config_plan,
        ),
    )

    return ChatResponse(
        session_id=session_id,
        intent=intent.intent,
        intent_label=intent.label,
        answer=final_answer,
        config_plan=generation.config_plan,
        validation=validation,
        seed_entities=retrieval.seed_entity_ids,
        relations=retrieval.relations,
        references=retrieval.chunks,
    )
