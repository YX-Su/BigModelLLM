"""Static knowledge base for mock mode.

The same 25 config items, 4 dimensions, 8 effects and 30 rules as the authored
documents — encoded as data so the mock graph can be built without any LLM
extraction or external graph database.
"""
from __future__ import annotations

from app.graph.schema import DIMENSIONS, EFFECTS
from app.schemas.knowledge import Entity, Relation

# (code, name, dimension, description, aliases)
CONFIG_ITEMS: list[tuple[str, str, str, str, list[str]]] = [
    ("RP-CONSERVATIVE", "保守型风险偏好", "客户画像维度",
     "将客户标记为保守型,以本金安全为首要目标,接受较低收益预期", ["保守型", "低风险偏好", "稳健保本"]),
    ("RP-BALANCED", "平衡型风险偏好", "客户画像维度",
     "在收益与风险之间取得平衡,接受中等程度的波动", ["平衡型", "稳健型", "中等风险"]),
    ("RP-AGGRESSIVE", "进取型风险偏好", "客户画像维度",
     "以追求较高收益为目标,接受较大的净值波动", ["进取型", "激进型", "高风险偏好"]),
    ("HORIZON-SHORT", "短期投资期限", "客户画像维度",
     "投资期限通常在1年以内,资金随时可能取用", ["短期", "一年内", "短线投资"]),
    ("HORIZON-LONG", "长期投资期限", "客户画像维度",
     "投资期限通常在3年以上,可承受期间净值波动", ["长期", "长线投资", "长期持有"]),
    ("ASSET-MONEY", "货币基金配置", "资产类别维度",
     "纳入货币市场基金,提供高流动性的低风险底仓", ["货币基金", "货基", "现金管理"]),
    ("ASSET-BOND", "债券基金配置", "资产类别维度",
     "纳入债券型基金,提供相对稳健的票息收益", ["债券基金", "债基", "固收基金"]),
    ("ASSET-EQUITY", "股票型基金配置", "资产类别维度",
     "纳入股票型基金,作为收益增长的主要来源", ["股票基金", "股基", "权益基金"]),
    ("ASSET-HYBRID", "混合型基金配置", "资产类别维度",
     "纳入混合型基金,由基金经理在股债之间动态调整", ["混合基金", "混基", "平衡型基金"]),
    ("ASSET-QDII", "QDII海外资产配置", "资产类别维度",
     "通过QDII基金配置海外资产,分散单一市场风险", ["QDII", "海外基金", "出海资产"]),
    ("ASSET-GOLD", "黄金及商品配置", "资产类别维度",
     "纳入黄金及大宗商品类资产,对冲通胀与极端风险", ["黄金", "贵金属", "商品资产"]),
    ("ASSET-INSURANCE", "保险理财配置", "资产类别维度",
     "纳入年金险、增额终身寿等保险理财,锁定长期确定收益", ["保险理财", "年金险", "增额终身寿"]),
    ("ASSET-PENSION", "个人养老金账户配置", "资产类别维度",
     "通过个人养老金账户配置养老目标基金,享受税收递延", ["养老金账户", "个人养老金", "养老目标基金"]),
    ("STRAT-LUMPSUM", "一次性买入", "交易策略维度",
     "建仓时将计划资金一次性买入目标资产", ["一次性买入", "整笔买入", "单笔建仓"]),
    ("STRAT-DCA", "定投策略", "交易策略维度",
     "将计划资金分批、定期投入,平滑买入成本", ["定投", "定期定额", "分批买入"]),
    ("STRAT-REBAL-Q", "季度再平衡", "交易策略维度",
     "每季度将组合各类资产比例调整回目标配置", ["季度再平衡", "季度调仓", "按季再平衡"]),
    ("STRAT-REBAL-Y", "年度再平衡", "交易策略维度",
     "每年将组合各类资产比例调整回目标配置", ["年度再平衡", "年度调仓", "按年再平衡"]),
    ("STRAT-DIVREINVEST", "分红再投资", "交易策略维度",
     "将基金分红自动转为再次申购,实现收益复利", ["分红再投资", "红利再投", "复利再投"]),
    ("STRAT-LEVERAGE", "杠杆融资配置", "交易策略维度",
     "通过融资方式放大组合的投资仓位以放大收益", ["杠杆", "加杠杆", "融资买入", "配资"]),
    ("RISK-TAKEPROFIT", "止盈线设置", "风控约束维度",
     "设置目标收益线,达到后触发部分或全部止盈", ["止盈", "止盈线", "目标收益线"]),
    ("RISK-STOPLOSS", "止损线设置", "风控约束维度",
     "设置最大可接受亏损线,触发后强制减仓", ["止损", "止损线", "下行止损"]),
    ("RISK-LIQUIDITY", "流动性保障", "风控约束维度",
     "强制预留一定比例的高流动性资产以应对应急用款", ["流动性保障", "应急资金", "备用金"]),
    ("RISK-LOCKUP", "长期锁定配置", "风控约束维度",
     "配置带封闭期的产品,以更低流动性换取更高收益", ["长期锁定", "封闭期配置", "锁定持有"]),
    ("RISK-MAXEQUITY", "权益仓位上限", "风控约束维度",
     "为组合设置权益类资产占比的硬性上限", ["权益上限", "仓位上限", "权益封顶"]),
    ("RISK-CONCENTRATION", "单一资产集中度限制", "风控约束维度",
     "限制单一资产在组合中的最大占比,分散非系统性风险", ["集中度限制", "单一上限", "分散度约束"]),
]

# (rule_id, relation_type, source, target)
RELATIONS_RAW: list[tuple[str, str, str, str]] = [
    ("R-DEP-01", "DEPENDS_ON", "STRAT-REBAL-Q", "ASSET-EQUITY"),
    ("R-DEP-01", "DEPENDS_ON", "STRAT-REBAL-Q", "ASSET-HYBRID"),
    ("R-DEP-02", "DEPENDS_ON", "STRAT-REBAL-Y", "ASSET-EQUITY"),
    ("R-DEP-02", "DEPENDS_ON", "STRAT-REBAL-Y", "ASSET-HYBRID"),
    ("R-DEP-03", "DEPENDS_ON", "RISK-TAKEPROFIT", "ASSET-EQUITY"),
    ("R-DEP-04", "DEPENDS_ON", "RISK-STOPLOSS", "ASSET-EQUITY"),
    ("R-DEP-05", "DEPENDS_ON", "STRAT-LEVERAGE", "RP-AGGRESSIVE"),
    ("R-DEP-06", "DEPENDS_ON", "STRAT-DIVREINVEST", "ASSET-BOND"),
    ("R-DEP-06", "DEPENDS_ON", "STRAT-DIVREINVEST", "ASSET-EQUITY"),
    ("R-DEP-06", "DEPENDS_ON", "STRAT-DIVREINVEST", "ASSET-HYBRID"),
    ("R-DEP-07", "DEPENDS_ON", "RISK-LOCKUP", "HORIZON-LONG"),
    ("R-DEP-08", "DEPENDS_ON", "ASSET-PENSION", "HORIZON-LONG"),
    ("R-DEP-09", "DEPENDS_ON", "ASSET-QDII", "RP-BALANCED"),
    ("R-DEP-09", "DEPENDS_ON", "ASSET-QDII", "RP-AGGRESSIVE"),
    ("R-MEX-01", "MUTEX", "RP-CONSERVATIVE", "RP-BALANCED"),
    ("R-MEX-01", "MUTEX", "RP-BALANCED", "RP-AGGRESSIVE"),
    ("R-MEX-01", "MUTEX", "RP-CONSERVATIVE", "RP-AGGRESSIVE"),
    ("R-MEX-02", "MUTEX", "HORIZON-SHORT", "HORIZON-LONG"),
    ("R-MEX-03", "MUTEX", "STRAT-LUMPSUM", "STRAT-DCA"),
    ("R-MEX-04", "MUTEX", "STRAT-REBAL-Q", "STRAT-REBAL-Y"),
    ("R-MEX-05", "MUTEX", "RP-CONSERVATIVE", "STRAT-LEVERAGE"),
    ("R-MEX-06", "MUTEX", "HORIZON-SHORT", "RISK-LOCKUP"),
    ("R-MEX-07", "MUTEX", "RISK-LIQUIDITY", "RISK-LOCKUP"),
    ("R-MEX-08", "MUTEX", "RP-CONSERVATIVE", "ASSET-QDII"),
    ("R-AFF-01", "ENHANCES", "ASSET-MONEY", "流动性"),
    ("R-AFF-02", "WEAKENS", "ASSET-MONEY", "收益预期"),
    ("R-AFF-03", "ENHANCES", "RISK-LOCKUP", "收益预期"),
    ("R-AFF-04", "WEAKENS", "RISK-LOCKUP", "流动性"),
    ("R-AFF-05", "ENHANCES", "ASSET-EQUITY", "收益预期"),
    ("R-AFF-06", "ENHANCES", "ASSET-EQUITY", "波动率"),
    ("R-AFF-07", "ENHANCES", "ASSET-BOND", "本金稳定性"),
    ("R-AFF-08", "WEAKENS", "STRAT-DCA", "择时风险"),
    ("R-AFF-09", "ENHANCES", "STRAT-LEVERAGE", "收益预期"),
    ("R-AFF-10", "ENHANCES", "STRAT-LEVERAGE", "波动率"),
    ("R-AFF-11", "ENHANCES", "ASSET-GOLD", "抗通胀能力"),
    ("R-AFF-12", "ENHANCES", "RISK-STOPLOSS", "下行保护"),
    ("R-AFF-13", "ENHANCES", "RISK-TAKEPROFIT", "收益锁定"),
]


# Curated, rule-valid solution templates used for mock-mode plan generation.
SOLUTION_TEMPLATES: dict[str, list[str]] = {
    "conservative": [
        "RP-CONSERVATIVE", "ASSET-MONEY", "ASSET-BOND", "ASSET-GOLD",
        "RISK-LIQUIDITY", "RISK-MAXEQUITY",
    ],
    "balanced": [
        "RP-BALANCED", "HORIZON-LONG", "ASSET-MONEY", "ASSET-BOND", "ASSET-EQUITY",
        "ASSET-QDII", "ASSET-GOLD", "STRAT-DCA", "STRAT-REBAL-Y", "RISK-STOPLOSS",
        "RISK-MAXEQUITY",
    ],
    "aggressive": [
        "RP-AGGRESSIVE", "HORIZON-LONG", "ASSET-BOND", "ASSET-EQUITY", "ASSET-QDII",
        "ASSET-GOLD", "STRAT-DCA", "STRAT-REBAL-Q", "RISK-TAKEPROFIT",
        "RISK-STOPLOSS", "RISK-CONCENTRATION",
    ],
    "pension": [
        "RP-BALANCED", "HORIZON-LONG", "ASSET-PENSION", "ASSET-INSURANCE",
        "ASSET-BOND", "ASSET-EQUITY", "ASSET-GOLD", "STRAT-DCA",
        "STRAT-DIVREINVEST", "RISK-LOCKUP",
    ],
}

_TEMPLATE_LABELS = {
    "conservative": "保守稳健型",
    "balanced": "平衡增值型",
    "aggressive": "进取成长型",
    "pension": "养老储备型",
}


def match_template(query: str) -> tuple[str, list[str]]:
    """Pick the solution template best matching a query; returns (label, plan)."""
    if any(k in query for k in ("保守", "低风险", "稳健保本", "本金安全", "求稳")):
        key = "conservative"
    elif any(k in query for k in ("养老", "退休", "传承")):
        key = "pension"
    elif any(k in query for k in ("进取", "成长", "高收益", "激进", "高风险")):
        key = "aggressive"
    else:
        key = "balanced"
    return _TEMPLATE_LABELS[key], list(SOLUTION_TEMPLATES[key])


def config_item_entities() -> list[Entity]:
    """The 25 config item entities (also feed the entity vector index)."""
    return [
        Entity(entity_id=code, name=name, type="ConfigItem",
               dimension=dimension, description=description, aliases=aliases)
        for code, name, dimension, description, aliases in CONFIG_ITEMS
    ]


def all_entities() -> list[Entity]:
    """Config item + dimension + effect entities."""
    entities = config_item_entities()
    entities += [Entity(entity_id=d, name=d, type="Dimension") for d in DIMENSIONS]
    entities += [Entity(entity_id=e, name=e, type="Effect") for e in EFFECTS]
    return entities


def all_relations() -> list[Relation]:
    """Dependency / mutex / influence relations plus BELONGS_TO edges."""
    relations = [
        Relation(source=source, target=target, type=rtype, rule_id=rule_id)
        for rule_id, rtype, source, target in RELATIONS_RAW
    ]
    relations += [
        Relation(source=code, target=dimension, type="BELONGS_TO")
        for code, _, dimension, _, _ in CONFIG_ITEMS
    ]
    return relations
