# 基于 GraphRAG 的投资组合配置助手

面向金融 / 保险投资组合配置场景的智能助手。将散落在文档中的产品与规则知识结构化为
**知识图谱 + 向量索引 + 关键词索引**,通过 GraphRAG 混合检索召回领域知识,再由大模型
完成自然语言理解与可控推理,帮助理财顾问快速查询配置定义、理解组合效果,并自动生成
符合风险偏好与预算约束的投资组合方案。

## 问题背景

投资组合 / 保险方案配置项多、维度多(产品类型、风险等级、预算约束、保障范围、收益
目标、流动性等),且配置项之间存在复杂的**依赖、互斥、影响**关系。顾问在为客户配置
方案时需要频繁查阅文档或咨询他人,效率低且易错。本项目把依赖人工经验的配置过程,
升级为"结构化知识 + 可控推理"的智能配置系统。

## 整体架构

前后端分离。后端由四个核心模块组成:

| 模块 | 职责 | 技术 |
| --- | --- | --- |
| RAG 检索 | GraphRAG 混合召回(语义 + 关键词 + 知识图谱) | Milvus / Elasticsearch / Neo4j |
| 上下文缓存 | 多轮对话状态管理、检索去重剪枝 | Redis |
| 意图识别 | 问题分类、无效问题过滤、提示词路由 | 轻量分类 |
| 生成 | 大模型生成 + 规则系统兜底校验 | OpenAI 兼容接口 |

### 在线推理链路

```
用户 query
  → 意图识别(分类 / 无效过滤)
  → GraphRAG 检索
      阶段一:实体识别 → 知识图谱 1~2 跳扩展 → 实体子图
      阶段二:子图驱动的向量召回 + BM25 召回 → 加权融合 → 阈值过滤
  → 上下文组装(知识文本段 + 实体关系三元组 + 历史对话摘要)
  → 大模型生成
  → 规则系统兜底校验(依赖 / 互斥 / 维度合法性)
  → 返回
```

## 目录结构

```
BigModelLLM/
├── backend/            后端服务(自主实现)
│   └── app/
│       ├── api/        FastAPI 路由
│       ├── core/       LLM / 存储客户端等基础设施
│       ├── rag/        GraphRAG 离线构建与在线检索
│       ├── graph/      知识图谱构建与查询
│       ├── cache/      Redis 上下文缓存
│       ├── intent/     意图识别
│       ├── generation/ 生成与规则校验
│       └── schemas/    数据模型
├── frontend/           对话前端(AI Coding 实现)
├── data/               领域知识文档与离线产物
├── deploy/             docker-compose 编排
├── scripts/            离线流水线 / 评测脚本
└── docs/               架构与设计文档
```

## 快速开始

```bash
# 1. 启动存储后端(Milvus / Elasticsearch / Neo4j / Redis)
docker compose -f deploy/docker-compose.yml up -d

# 2. 安装后端依赖
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env   # 填入 OpenAI 兼容接口的 BASE_URL / API_KEY

# 4. 启动服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 技术栈

- **后端**:Python + FastAPI
- **向量库**:Milvus(chunk 索引 + 实体索引)
- **关键词索引**:Elasticsearch(BM25)
- **知识图谱**:Neo4j(Cypher)
- **缓存**:Redis
- **模型**:OpenAI 兼容的 Chat / Embedding 接口
