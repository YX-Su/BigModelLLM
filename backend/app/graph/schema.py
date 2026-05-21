"""Knowledge graph schema: controlled vocabulary for entity and relation types.

Used as the predefined schema for LLM-based entity / relation extraction and as
the validation whitelist when writing to Neo4j.
"""
from __future__ import annotations

# Entity types -> Neo4j labels
ENTITY_CONFIG_ITEM = "ConfigItem"
ENTITY_DIMENSION = "Dimension"
ENTITY_EFFECT = "Effect"
ENTITY_TYPES = [ENTITY_CONFIG_ITEM, ENTITY_DIMENSION, ENTITY_EFFECT]

# Controlled vocabulary for the customer/asset/strategy/risk dimensions
DIMENSIONS = [
    "客户画像维度",
    "资产类别维度",
    "交易策略维度",
    "风控约束维度",
]

# Controlled vocabulary for business effects
EFFECTS = [
    "流动性",
    "收益预期",
    "本金稳定性",
    "波动率",
    "择时风险",
    "抗通胀能力",
    "下行保护",
    "收益锁定",
]

# Relation types -> Neo4j relationship types
REL_BELONGS_TO = "BELONGS_TO"
REL_DEPENDS_ON = "DEPENDS_ON"
REL_MUTEX = "MUTEX"
REL_ENHANCES = "ENHANCES"
REL_WEAKENS = "WEAKENS"
RELATION_TYPES = [
    REL_BELONGS_TO,
    REL_DEPENDS_ON,
    REL_MUTEX,
    REL_ENHANCES,
    REL_WEAKENS,
]

# Relation types meaningful for online graph expansion (excludes structural BELONGS_TO)
EXPANDABLE_RELATIONS = [
    REL_DEPENDS_ON,
    REL_MUTEX,
    REL_ENHANCES,
    REL_WEAKENS,
]
