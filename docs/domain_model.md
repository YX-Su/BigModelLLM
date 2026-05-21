# 领域建模:实体与关系 Schema

本文档定义投资组合配置领域的知识图谱 Schema,是离线知识图谱构建(实体抽取、关系
抽取)与在线检索的共同依据。

## 配置维度

| 维度 | 编码前缀 | 说明 |
| --- | --- | --- |
| 客户画像维度 | `RP-` / `HORIZON-` | 风险偏好与投资期限 |
| 资产类别维度 | `ASSET-` | 组合可纳入的各类资产 |
| 交易策略维度 | `STRAT-` | 买入、再平衡、分红等操作策略 |
| 风控约束维度 | `RISK-` | 止盈止损、流动性、集中度等风控手段 |

## 实体类型(Entity)

| 实体类型 | Neo4j Label | 说明 | 数量 |
| --- | --- | --- | --- |
| 配置项 | `ConfigItem` | 可被开启/关闭的最小配置单元 | 25 |
| 业务维度 | `Dimension` | 配置项所属维度 | 4 |
| 业务效果 | `Effect` | 配置项最终作用的效果指标 | 8 |

每个实体具备稳定的 `entity_id`(等于配置项编码,如 `ASSET-EQUITY`)、`name`
(标准名称)、`aliases`(别名列表)、`description`(简要描述)、`type`(实体类型)。

## 关系类型(Relation)

| 关系类型 | Neo4j 关系 | 方向 | 说明 |
| --- | --- | --- | --- |
| 归属 | `BELONGS_TO` | ConfigItem → Dimension | 配置项归属于某维度 |
| 依赖 | `DEPENDS_ON` | ConfigItem → ConfigItem | 源配置项依赖目标配置项 |
| 互斥 | `MUTEX` | ConfigItem ↔ ConfigItem | 两个配置项不可同时开启 |
| 增强 | `ENHANCES` | ConfigItem → Effect | 配置项增强某业务效果 |
| 减弱 | `WEAKENS` | ConfigItem → Effect | 配置项减弱某业务效果 |

`DEPENDS_ON` 与 `MUTEX` 携带 `rule_id` 属性(如 `R-DEP-01`),指向规则文档中的
具体规则,供生成阶段的规则校验引用。`MUTEX` 为无向语义,存储时按对称写入两条边。

## 业务效果实体清单

流动性、收益预期、本金稳定性、波动率、择时风险、抗通胀能力、下行保护、收益锁定。

## 实体消歧

用户表达口语化,同一配置项可能有多种说法。实体的 `aliases` 字段收录常见别名,
例如:

- `ASSET-EQUITY` 股票型基金配置 ←「股票基金」「权益基金」「股基」
- `STRAT-DCA` 定投策略 ←「定期定额」「分批买入」「基金定投」
- `STRAT-LEVERAGE` 杠杆融资配置 ←「加杠杆」「融资买入」「配资」

实体向量索引的 embedding 文本由「标准名称 + 别名 + 描述 + 维度标签」拼接而成,
用于将用户 query 稳定映射到标准实体。

## GraphRAG 两跳扩展示例

以用户 query「想给客户做股票基金的止盈」为例:

1. 实体识别召回 `ASSET-EQUITY`、`RISK-TAKEPROFIT`。
2. 沿关系扩展 1~2 跳:
   - `RISK-TAKEPROFIT --DEPENDS_ON--> ASSET-EQUITY`
   - `ASSET-EQUITY --ENHANCES--> 收益预期`、`--ENHANCES--> 波动率`
   - `RISK-TAKEPROFIT --ENHANCES--> 收益锁定`
3. 得到的实体子图作为第二阶段混合检索的查询输入。
