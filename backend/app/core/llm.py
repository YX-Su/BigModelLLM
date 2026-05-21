"""OpenAI-compatible client wrappers for chat completion and embeddings.

Both the LLM and the embedding model are accessed through the OpenAI SDK, so
any OpenAI-compatible gateway (local vLLM, third-party proxy, etc.) works by
pointing the ``*_BASE_URL`` settings at it.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

_llm_client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    timeout=settings.llm_timeout,
)

_embedding_client = OpenAI(
    base_url=settings.embedding_base_url,
    api_key=settings.embedding_api_key,
    timeout=settings.llm_timeout,
)

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    reraise=True,
)


@_RETRY
def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run a chat completion and return the assistant message text."""
    if settings.mock_mode:
        from app.mock.llm import mock_chat

        return mock_chat(messages)
    response = _llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


@_RETRY
def chat_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
) -> Any:
    """Run a chat completion constrained to JSON output and parse the result."""
    if settings.mock_mode:
        from app.mock.llm import mock_chat_json

        return mock_chat_json(messages)
    response = _llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM JSON parse failed, returning raw content")
        return {"_raw": content}


@_RETRY
def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts and return their vectors."""
    if not texts:
        return []
    if settings.mock_mode:
        from app.mock.llm import mock_embed

        return mock_embed(texts, settings.embedding_dim)
    response = _embedding_client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_one(text: str) -> list[float]:
    """Embed a single text and return its vector."""
    return embed([text])[0]
