"""Mock LLM — deterministic stand-ins for chat / chat_json / embed.

Lets the whole pipeline run with no API key. Embeddings are hashed character
bigram vectors, so lexically similar texts get similar vectors and the
cosine-based retrieval still produces relevant results.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

# IDF table over the knowledge base, built by build_idf() at bootstrap time.
# Down-weights common bigrams (基金 / 配置) so distinctive ones drive similarity.
_IDF: dict[str, float] = {}
_DEFAULT_IDF = 3.0


def _clean(text: str) -> str:
    return "".join(text.split())


def _bigram_list(text: str) -> list[str]:
    cleaned = _clean(text)
    if len(cleaned) < 2:
        cleaned = (cleaned or "_") * 2
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def build_idf(documents: list[str]) -> None:
    """Build the bigram IDF table from the knowledge base corpus."""
    global _IDF, _DEFAULT_IDF
    doc_freq: dict[str, int] = {}
    for doc in documents:
        for bigram in set(_bigram_list(doc)):
            doc_freq[bigram] = doc_freq.get(bigram, 0) + 1
    total = max(len(documents), 1)
    _IDF = {bg: math.log((total + 1) / (count + 1)) + 1.0 for bg, count in doc_freq.items()}
    _DEFAULT_IDF = math.log(total + 1) + 1.0


def mock_embed_one(text: str, dim: int) -> list[float]:
    """Hashed, IDF-weighted character-bigram vector; cosine ~ weighted overlap."""
    vec = [0.0] * dim
    for bigram in _bigram_list(text):
        weight = _IDF.get(bigram, _DEFAULT_IDF)
        bucket = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += weight
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def mock_embed(texts: list[str], dim: int) -> list[list[float]]:
    return [mock_embed_one(text, dim) for text in texts]


def _user_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def _extract_section(text: str, header: str) -> str:
    """Return the body of a 【header】 block from the assembled prompt."""
    start = text.find(f"【{header}】")
    if start < 0:
        return ""
    start += len(header) + 2
    rest = text[start:]
    nxt = rest.find("【")
    return (rest[:nxt] if nxt >= 0 else rest).strip()


def mock_chat(messages: list[dict[str, str]]) -> str:
    """Mock explanation answer, synthesised from the retrieved context."""
    user = _user_message(messages)
    question = _extract_section(user, "用户问题") or "你的问题"
    references = _extract_section(user, "领域知识参考")
    triples = _extract_section(user, "实体关系三元组")

    if not references:
        return (
            f"(Mock 模式)关于「{question}」,知识库中暂未检索到足够相关的资料,"
            "建议换一种说法或补充更具体的配置项名称。"
        )

    # Pick the reference paragraph most relevant to the question.
    paragraphs = [p.strip() for p in references.split("\n\n") if p.strip()]
    question_bigrams = set(_bigram_list(question))
    best = max(
        paragraphs,
        key=lambda p: len(question_bigrams & set(_bigram_list(p))),
    )
    answer = f"(Mock 模式回答)关于「{question}」,根据知识库检索到的资料:\n\n{best}"

    if triples:
        lines = [ln for ln in triples.splitlines() if ln.strip()]
        answer += "\n\n相关配置项关系:\n" + "\n".join(lines[:6])
        if len(lines) > 6:
            answer += f"\n…(共 {len(lines)} 条关系,详见检索链路面板)"
    answer += "\n\n注:本回答由 Mock 模式基于检索内容拼装,用于演示完整链路,非大模型生成。"
    return answer


def mock_chat_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Mock structured output for intent / generation prompts."""
    prompt = " ".join(m.get("content", "") for m in messages)

    # Intent classification prompt.
    if '"intent"' in prompt:
        return {"intent": "explain", "reason": "Mock 模式默认意图"}

    # Generation prompt — pick the solution template matching the query.
    from app.mock.knowledge import match_template

    user = _user_message(messages)
    question = _extract_section(user, "用户问题") or user
    label, plan = match_template(question)
    answer = (
        f"(Mock 模式方案)根据业务诉求,判定为「{label}」组合,建议配置以下配置项:"
        + "、".join(plan)
        + "。该方案由 Mock 模式依据方案模版生成,并已交由规则系统做依赖与互斥校验。"
    )
    return {"answer": answer, "config_plan": plan}
