# 在线推理链路示例

以一条真实查询为例,完整走一遍在线推理链路。

**用户输入**

- 会话:`s-demo01`(第 1 轮)
- query:`想给客户的股票基金设置止盈,有什么需要注意的`

---

## 步骤 1 · 意图识别

`app/intent/classifier.py` → `classify()`

- 规则分类命中领域关键词(基金、止盈)与意图关键词(需要注意 → 解释类线索)。
- 结果:`intent = explain`(效果解释),`confidence = 0.7`,规则路径直接返回,不调用 LLM。
- 非无效问题,进入后续链路。

## 步骤 2 · 上下文缓存

`app/cache/` → `select_related_turns()` / `build_history_context()`

- 会话 `s-demo01` 为首轮,历史为空。
- `history_context = ""`,`cached_entities = {}`(无可剪枝实体)。

## 步骤 3 · GraphRAG 检索 — 阶段一(实体识别与图谱扩展)

`app/rag/retrieval.py` → `link_entities()` + `expand_subgraph()`

- query 向量化后,在 Milvus 实体索引中召回 top-k 实体:
  - `ASSET-EQUITY`(股票型基金配置)— 由"股票基金"对齐
  - `RISK-TAKEPROFIT`(止盈线设置)— 由"设置止盈"对齐
- 在 Neo4j 中沿依赖 / 互斥 / 影响关系做 1~2 跳扩展,得到实体子图:
  - `RISK-TAKEPROFIT --DEPENDS_ON--> ASSET-EQUITY`（规则 R-DEP-03）
  - `ASSET-EQUITY --ENHANCES--> 收益预期`
  - `ASSET-EQUITY --ENHANCES--> 波动率`
  - `RISK-TAKEPROFIT --ENHANCES--> 收益锁定`

> 关键点:即使用户没有提到"依赖",图谱扩展也补全了"止盈依赖股票型基金"这一约束,
> 这是纯向量检索难以稳定召回的结构化知识。

## 步骤 4 · GraphRAG 检索 — 阶段二(子图驱动的混合检索)

`app/rag/retrieval.py` → `_hybrid_retrieve()`

- 以原始 query 与子图实体描述为查询集合,分别做:
  - 向量召回(Milvus chunk 集合)→ 语义相关片段
  - BM25 召回(Elasticsearch)→ 精确命中"止盈线""股票型基金"等
- 候选合并去重,按 `0.6 × 向量分 + 0.4 × BM25 分` 加权融合排序。
- 按相似度阈值 `0.35` 过滤,保留高置信片段,例如:
  - `RISK-TAKEPROFIT 止盈线设置` 配置项说明
  - `ASSET-EQUITY 股票型基金配置` 配置项说明
  - `R-DEP-03 止盈线依赖股票型基金` 规则说明

## 步骤 5 · 上下文组装与生成

`app/generation/` → `build_generation_context()` + `generate()`

- 组装上下文:历史摘要(空)+ 实体关系三元组 + 召回的领域知识片段。
- 意图为 `explain`,选用解释类提示词模版,调用大模型生成回答。
- 回答要点:止盈线作用、需先配置股票型基金(规则 R-DEP-03)、止盈线取值建议、
  止盈对"收益锁定"的增强作用。

## 步骤 6 · 规则校验

`app/generation/validator.py`

- 本轮意图为 `explain`,未产出结构化配置方案,`config_plan` 为空,跳过规则校验。
- (若为 generate / optimize 意图,则会校验方案是否违反依赖 / 互斥规则。)

## 步骤 7 · 持久化与返回

`app/pipeline.py` → `SessionStore.append_turn()`

- 将本轮问题、回答、命中实体(`ASSET-EQUITY`、`RISK-TAKEPROFIT`)写入 Redis。
- 返回 `ChatResponse`,包含回答正文、命中实体、子图三元组、参考片段。

下一轮若用户追问"那止损呢",上下文缓存会带入本轮历史,检索阶段会跳过已扩展的
`ASSET-EQUITY` 等实体,减少冗余检索。

---

## 对比:方案生成类查询

若 query 为 `给一位保守型客户配一套低风险组合`(意图 `generate`):

- 步骤 5 使用方案生成模版,大模型按"确认画像 → 选资产 → 选策略 → 加风控 → 自检"
  的思维路径输出 JSON:`{"answer": ..., "config_plan": ["RP-CONSERVATIVE", ...]}`。
- 步骤 6 规则校验生效:若方案中同时出现互斥项(如 `RP-CONSERVATIVE` 与
  `STRAT-LEVERAGE`),校验器标注冲突(规则 R-MEX-05)并给出修正后的建议方案。
