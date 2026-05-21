"""Generation context assembly.

Combines the three context sources into one prompt context block:
recalled domain knowledge chunks, entity relation triples, and the condensed
dialogue history — keeping the LLM inside the intended business semantics.
"""
from __future__ import annotations

from app.schemas.retrieval import RetrievalResult


def build_generation_context(result: RetrievalResult, history_context: str = "") -> str:
    """Assemble the reference context passed to the generation LLM."""
    sections: list[str] = []

    if history_context.strip():
        sections.append("【历史对话摘要】\n" + history_context.strip())

    triples = result.triples_context()
    if triples:
        sections.append("【实体关系三元组】\n" + triples)

    references = result.reference_context()
    if references:
        sections.append("【领域知识参考】\n" + references)

    return "\n\n".join(sections) if sections else "(未检索到相关参考资料)"
