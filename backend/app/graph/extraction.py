"""LLM-based entity and relation extraction.

Each knowledge base chunk is passed to the LLM together with the predefined
graph schema. Config item chunks yield :class:`Entity` objects; rule chunks
yield :class:`Relation` objects. Stable ids (config codes / rule ids) are
parsed deterministically from the chunk title so extraction results stay
anchored regardless of LLM phrasing.
"""
from __future__ import annotations

import logging
import re

from app.core.llm import chat_json
from app.graph.schema import DIMENSIONS, EFFECTS
from app.schemas.knowledge import Chunk, Entity, Relation

logger = logging.getLogger(__name__)

_RULE_ID_RE = re.compile(r"\bR-(?:DEP|MEX|AFF)-\d+\b")

_CONFIG_ITEM_PROMPT = """你是金融投资领域的知识抽取助手。请从给定的「配置项说明文本」中抽取一个配置项实体。

需要抽取的字段:
- name: 配置项的标准中文名称
- dimension: 所属维度,必须严格取以下之一:{dimensions}
- aliases: 该配置项常见的口语化别名,字符串数组,给出 2~4 个
- description: 用一句话概括该配置项的作用

仅输出 JSON 对象,键为 name、dimension、aliases、description。

配置项编码:{code}
配置项文本:
{text}"""

_RELATION_PROMPT = """你是金融投资领域的知识抽取助手。请从给定的「配置项规则文本」中抽取配置项之间的关系。

关系类型(type 字段取值):
- DEPENDS_ON: 依赖,source 配置项必须在 target 配置项开启后才能开启
- MUTEX: 互斥,source 与 target 两个配置项不能同时开启
- ENHANCES: 增强,source 配置项增强 target 业务效果
- WEAKENS: 减弱,source 配置项减弱 target 业务效果

约定:
- DEPENDS_ON 与 MUTEX 关系中,source 和 target 都是配置项编码(大写英文,如 ASSET-EQUITY)。
- ENHANCES 与 WEAKENS 关系中,source 是配置项编码,target 是业务效果名称,
  业务效果必须严格取以下之一:{effects}
- 一条规则可能产生多条关系。例如三个配置项两两互斥应输出 3 条 MUTEX 关系;
  「A 依赖 B 或 C」应输出 A->B 与 A->C 两条 DEPENDS_ON 关系。

仅输出 JSON 对象,格式为 {{"relations": [{{"source": ..., "target": ..., "type": ...}}]}}。

规则文本:
{text}"""


def _title_code(chunk: Chunk) -> str:
    """The leading token of a chunk title is its stable code."""
    return chunk.title.split()[0] if chunk.title else chunk.chunk_id


def extract_config_item(chunk: Chunk) -> Entity:
    """Extract a single ConfigItem entity from a config item chunk."""
    code = _title_code(chunk)
    prompt = _CONFIG_ITEM_PROMPT.format(
        dimensions="、".join(DIMENSIONS),
        code=code,
        text=chunk.text,
    )
    data = chat_json([{"role": "user", "content": prompt}])

    dimension = str(data.get("dimension", "")).strip()
    if dimension not in DIMENSIONS:
        logger.warning("entity %s: dimension '%s' off-schema", code, dimension)

    aliases = data.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []

    return Entity(
        entity_id=code,
        name=str(data.get("name", code)).strip() or code,
        type="ConfigItem",
        aliases=[str(a).strip() for a in aliases if str(a).strip()],
        description=str(data.get("description", "")).strip(),
        dimension=dimension,
    )


def extract_relations(chunk: Chunk) -> list[Relation]:
    """Extract relations from a rule chunk."""
    rule_match = _RULE_ID_RE.search(chunk.title) or _RULE_ID_RE.search(chunk.text)
    rule_id = rule_match.group(0) if rule_match else ""

    prompt = _RELATION_PROMPT.format(effects="、".join(EFFECTS), text=chunk.text)
    data = chat_json([{"role": "user", "content": prompt}])

    raw = data.get("relations", []) if isinstance(data, dict) else []
    relations: list[Relation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        rtype = str(item.get("type", "")).strip().upper()
        if not source or not target or not rtype:
            continue
        relations.append(
            Relation(source=source, target=target, type=rtype, rule_id=rule_id)
        )
    return relations
