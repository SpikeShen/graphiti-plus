# Graphiti 改造版 — 代码架构总览

基于 [Zep Graphiti](https://github.com/getzep/graphiti) 开源项目的 AWS 全栈改造版本。
将原生 OpenAI + Neo4j 向量检索替换为 Bedrock LLM + Nova Embeddings + S3 Vectors，
并扩展了多模态文档处理、深度搜索、原文溯源等能力。

---

## 1. 技术栈

| 层 | 组件 | 说明 |
|----|------|------|
| LLM | Kimi K2.5 via Bedrock Mantle | OpenAI 兼容端点，`aws-bedrock-token-generator` 鉴权 |
| Embedding | Nova Multimodal Embeddings v1 | boto3 `invoke_model`，1024 维，支持文本+图片 |
| Reranker | Kimi K2.5 (布尔分类) | 复用 LLM client，True/False 相关性判断 |
| 图数据库 | Neo4j 5.x | Docker 本地部署，BM25 全文索引 + 图遍历 |
| 向量检索 | S3 Vectors | Serverless ANN 索引，metadata 过滤，替代 Neo4j 暴力余弦 |
| 资产存储 | S3 | 多模态二进制文件（图片/音频/视频） |
| 日志 | S3 + Athena | LLM/Embedding 调用日志，JSON Lines gzip 格式 |

---

## 2. 目录结构

```
graphiti_core/
├── graphiti.py              # 主入口：Graphiti 类，add_episode / search_ / build_indices
├── nodes.py                 # 节点模型：EpisodicNode, EntityNode, CommunityNode, ContentBlock
├── edges.py                 # 边模型：EpisodicEdge, EntityEdge, DescribesEdge
├── helpers.py               # semaphore_gather 等并发工具
├── errors.py                # 自定义异常
│
├── llm_client/
│   ├── client.py            # LLMClient 抽象基类
│   ├── bedrock_client.py    # ★ BedrockLLMClient — Mantle 端点 + 超时重试
│   ├── openai_client.py     # 原生 OpenAI client（保留）
│   ├── config.py            # LLMConfig (model, temperature, etc.)
│   └── token_tracker.py     # Token 用量追踪
│
├── embedder/
│   ├── client.py            # EmbedderClient 抽象基类
│   ├── bedrock_nova.py      # ★ BedrockNovaEmbedder — Nova MME，支持文本+图片 embedding
│   └── openai.py            # 原生 OpenAI embedder（保留）
│
├── cross_encoder/
│   ├── client.py            # CrossEncoderClient 抽象基类
│   └── bedrock_reranker_client.py  # ★ BedrockRerankerClient — LLM 布尔分类 reranker
│
├── vector_store/
│   └── s3_vectors_client.py # ★ S3VectorsClient — 8 个向量索引的 CRUD + 相似度检索
│
├── driver/
│   ├── driver.py            # GraphDriver 抽象基类 + GraphProvider 枚举
│   ├── neo4j_driver.py      # Neo4jDriver — 主要使用的图驱动
│   ├── neo4j/operations/    # Neo4j 专用 CRUD 操作实现
│   ├── operations/          # 通用操作接口定义
│   └── ...                  # FalkorDB, Kuzu, Neptune 驱动（保留）
│
├── search/
│   ├── search.py            # search() 主入口 — 并行执行 edge/node/episode/community 搜索
│   ├── search_config.py     # SearchConfig, EdgeSearchConfig 等配置模型
│   ├── search_config_recipes.py  # 预置搜索配方（COMBINED_HYBRID, EDGE_DEEP_SEARCH 等）
│   ├── search_utils.py      # ★ s3_vectors_*_similarity_search — S3 Vectors 检索实现
│   ├── search_filters.py    # 搜索过滤条件
│   └── search_helpers.py    # RRF, MMR, BFS 等排序算法
│
├── prompts/
│   ├── extract_nodes.py     # 实体提取 prompt
│   ├── extract_edges.py     # 关系提取 prompt（含 source_excerpt）
│   ├── dedupe_nodes.py      # 节点去重 prompt
│   ├── dedupe_edges.py      # 边去重 prompt
│   ├── summarize_nodes.py   # 节点摘要 prompt
│   └── models.py            # Prompt 输出的 Pydantic 模型
│
├── preprocessing/
│   ├── parser.py            # DocumentParserRegistry — 文档解析器注册表
│   ├── word_parser.py       # Word/DOCX 解析器
│   ├── description.py       # ★ generate_image_descriptions — 视觉 LLM 图片描述
│   └── asset_storage.py     # ★ MultimodalAssetStorage — S3 二进制资产上传
│
├── logging/
│   └── s3_logger.py         # ★ S3InvocationLogger — LLM/Embedding 调用日志
│
├── models/
│   ├── nodes/               # Neo4j 节点 CRUD 查询
│   └── edges/               # Neo4j 边 CRUD 查询
│
├── utils/
│   ├── bulk_utils.py        # 批量写入（节点+边+S3 Vectors 同步）
│   ├── content_chunking.py  # 内容分块
│   └── datetime_utils.py    # 时间工具
│
└── namespaces/              # graphiti.nodes.entity / graphiti.edges 命名空间 API
```

> ★ 标记为本次改造新增或重大修改的文件

---

## 3. 数据模型

### 3.1 节点类型

```
Node (抽象基类)
├── EpisodicNode          # 原始输入（一段对话/一篇文档）
│   ├── content           # 原始文本
│   ├── content_blocks    # 多模态内容块列表 (ContentBlock[])
│   ├── narrative_excerpts # 未归属到实体的叙事片段
│   └── describes_edges   # 关联的 DescribesEdge UUID 列表
│
├── EntityNode            # 提取出的实体（人物、地点、概念等）
│   ├── name_embedding    # 名称向量
│   ├── summary           # 周边关系摘要
│   ├── labels            # 类型标签
│   └── attributes        # 扩展属性（可自定义 Pydantic 模型）
│
├── CommunityNode         # 实体社区（自动聚类）
│   ├── name_embedding    # 名称向量
│   └── summary           # 社区摘要
│
└── SagaNode              # 故事线/文档序列
```

### 3.2 边类型

```
Edge (抽象基类)
├── EpisodicEdge          # Episode → Entity (MENTIONS)
│
├── EntityEdge            # Entity → Entity (RELATES_TO)
│   ├── fact              # 关系事实描述
│   ├── fact_embedding    # fact 向量
│   ├── source_excerpt    # ★ 原文片段（深度搜索用）
│   ├── source_excerpt_embedding  # ★ 原文片段向量
│   ├── episodes          # 引用的 episode UUID 列表
│   ├── valid_at / invalid_at / expired_at  # 时序有效性
│   └── attributes        # 扩展属性
│
├── DescribesEdge         # ★ Episode → Entity (DESCRIBES)
│   ├── fact              # LLM 生成的描述摘要
│   ├── fact_embedding    # fact 向量
│   ├── excerpt           # 原文片段
│   └── excerpt_embedding # 原文片段向量
│
├── HasEpisodeEdge        # Saga → Episode (HAS_EPISODE)
├── NextEpisodeEdge       # Episode → Episode (NEXT_EPISODE)
└── CommunityEdge         # Entity → Community (HAS_MEMBER)
```

### 3.3 ContentBlock（多模态内容块）

```python
ContentBlock:
    index: int              # 文档中的顺序位置
    block_type: text | image | table | audio | video | ...
    text: str | None        # 文本内容
    s3_uri: str | None      # S3 对象 URI（二进制资产）
    description: str | None # LLM 生成的内容描述
    semantic_role: heading | body | caption | footnote | ...
    source_page: int | None # 来源页码
    _raw_bytes: bytes       # 临时二进制数据（处理后清除）
    _embedding: list[float] # 预计算的图片向量
```

---

## 4. Ingest Pipeline（数据摄入流水线）

### 4.1 add_episode — 文本/对话摄入

```
输入: episode_body (文本) + metadata
  │
  ├─ 去重检测: episode 内容 embedding → S3 Vectors 近似查询 → 跳过重复
  │
  ├─ Phase 1: 提取节点 (extract_nodes)
  │   └─ LLM 从文本中识别实体 → EntityNode[]
  │
  ├─ Phase 2: 解析节点 (resolve_extracted_nodes)
  │   └─ 与已有节点去重合并 → uuid_map
  │
  ├─ Phase 3: 提取 & 解析边 (extract_and_resolve_edges)
  │   └─ LLM 提取关系 → 去重 → 时序冲突处理
  │   └─ 输出: resolved_edges + invalidated_edges + narrative_excerpts
  │
  ├─ Phase 4: 提取节点属性 (extract_attributes_from_nodes)
  │   └─ LLM 补充实体的详细属性
  │
  ├─ Phase 5: 保存到图 (save to Neo4j)
  │   └─ 写入节点、边、episodic edges
  │
  └─ Phase 6: 同步到 S3 Vectors
      ├─ 节点 name_embedding → entity-name-embeddings 索引
      ├─ 边 fact_embedding → edge-fact-embeddings 索引
      ├─ 边 source_excerpt_embedding → edge-excerpt-embeddings 索引
      ├─ DescribesEdge → describes-fact/excerpt-embeddings 索引
      ├─ narrative_excerpts → episode-narrative-embeddings 索引
      └─ episode content_embedding → episode-content-embeddings 索引（去重用）
```

### 4.2 add_document_episode — 多模态文档摄入

```
输入: file_path 或 content_blocks
  │
  ├─ Step 1: 解析文档 → ContentBlock[] (DocumentParserRegistry)
  ├─ Step 2: 图片 embedding (Nova MME create_image)
  ├─ Step 3: 上传二进制资产到 S3 (MultimodalAssetStorage)
  ├─ Step 4: 视觉 LLM 生成图片描述 (generate_image_descriptions)
  ├─ Step 5: 拼接纯文本表示 (build_content_from_blocks)
  ├─ Step 6: 设置 image_embedding_map（供边 embedding 使用）
  └─ Step 7: 委托给 add_episode()
```

---

## 5. Search Pipeline（检索流水线）

```
查询: query (文本) + SearchConfig
  │
  ├─ 生成查询向量: embedder.create(query)
  │
  ├─ 并行执行 4 路搜索:
  │   ├─ edge_search
  │   │   ├─ BM25 全文 (Neo4j fulltext index)
  │   │   ├─ 向量相似度 (S3 Vectors edge-fact-embeddings)
  │   │   ├─ 原文相似度 (S3 Vectors edge-excerpt-embeddings)  ← 深度搜索
  │   │   ├─ BFS 图遍历 (Neo4j Cypher)
  │   │   └─ Rerank: RRF / MMR / Cross-Encoder / Node Distance
  │   │
  │   ├─ node_search
  │   │   ├─ BM25 全文 (Neo4j fulltext index)
  │   │   ├─ 向量相似度 (S3 Vectors entity-name-embeddings)
  │   │   ├─ 原文相似度 (S3 Vectors describes-excerpt-embeddings)  ← 深度搜索
  │   │   ├─ BFS 图遍历 (Neo4j Cypher)
  │   │   └─ Rerank: RRF / MMR / Cross-Encoder
  │   │
  │   ├─ episode_search
  │   │   ├─ BM25 全文 (Neo4j fulltext index)
  │   │   └─ Rerank: RRF / Cross-Encoder
  │   │
  │   └─ community_search
  │       ├─ BM25 全文 (Neo4j fulltext index)
  │       ├─ 向量相似度 (S3 Vectors community-name-embeddings)
  │       └─ Rerank: RRF / MMR / Cross-Encoder
  │
  ├─ 叙事片段检索 (深度搜索模式)
  │   └─ S3 Vectors episode-narrative-embeddings
  │
  └─ 返回: SearchResults (edges, nodes, episodes, communities, narrative_excerpts)
```

### 5.1 S3 Vectors 索引一览

| 索引名 | 内容 | 来源 |
|--------|------|------|
| entity-name-embeddings | 实体名称向量 | EntityNode.name_embedding |
| edge-fact-embeddings | 关系事实向量 | EntityEdge.fact_embedding |
| edge-excerpt-embeddings | 关系原文片段向量 | EntityEdge.source_excerpt_embedding |
| community-name-embeddings | 社区名称向量 | CommunityNode.name_embedding |
| episode-narrative-embeddings | 叙事片段向量 | EpisodicNode.narrative_excerpts |
| describes-fact-embeddings | 描述边事实向量 | DescribesEdge.fact_embedding |
| describes-excerpt-embeddings | 描述边原文片段向量 | DescribesEdge.excerpt_embedding |
| episode-content-embeddings | Episode 内容向量 | 去重检测用 |

---

## 6. 关键改造点

### 6.1 向量检索路径替换

```
原始: Neo4j vector.similarity.cosine() — O(n) 暴力扫描
改造: S3 Vectors ANN 索引 — metadata 过滤 + top-K 检索 → UUID 列表 → Neo4j 回查完整数据
```

### 6.2 深度搜索（原文溯源）

- EntityEdge 新增 `source_excerpt` + `source_excerpt_embedding`
- DescribesEdge 新增 `excerpt` + `excerpt_embedding`（Episode→Entity 描述边）
- EpisodicNode 新增 `narrative_excerpts`（未归属到实体的叙事片段）
- 搜索时通过 `source_similarity` 方法检索原文片段，与 fact 向量检索 RRF 融合

### 6.3 多模态文档处理

- ContentBlock 模型支持 text/image/table/audio/video 等模态
- DocumentParserRegistry 可注册自定义解析器
- 图片通过 Nova MME 生成 embedding，视觉 LLM 生成文字描述
- 二进制资产上传到 S3，ContentBlock.s3_uri 记录位置

### 6.4 LLM 超时与重试

- 按 prompt 类型配置超时（edge 提取 60s，其他 15s）
- `asyncio.timeout()` 包装，超时后自动重试（最多 2 次）
- 应对 Bedrock Mantle 网关层间歇性 ~300s 卡死问题

### 6.5 可观测性

- S3InvocationLogger: 缓冲写入 S3，JSON Lines gzip 格式
- Athena DDL 支持 Hive 分区查询
- `[LLM_TRACE]` 日志（`GRAPHITI_LLM_TRACE=true` 启用）
- 6 阶段 ingest 进度日志

---

## 7. 使用示例

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.bedrock_client import BedrockLLMClient
from graphiti_core.embedder.bedrock_nova import BedrockNovaEmbedder, BedrockNovaEmbedderConfig
from graphiti_core.cross_encoder.bedrock_reranker_client import BedrockRerankerClient
from graphiti_core.vector_store.s3_vectors_client import S3VectorsClient, S3VectorsConfig

# 初始化
llm = BedrockLLMClient(region_name='us-east-1')
embedder = BedrockNovaEmbedder(BedrockNovaEmbedderConfig(region_name='us-east-1'))
reranker = BedrockRerankerClient(client=llm.client)
s3v = S3VectorsClient(S3VectorsConfig(vector_bucket_name='my-bucket'))

g = Graphiti(
    uri='bolt://localhost:7687', user='neo4j', password='password',
    llm_client=llm, embedder=embedder, cross_encoder=reranker,
    s3_vectors=s3v,
)
await g.build_indices_and_constraints()

# 摄入
await g.add_episode(name='ep1', episode_body='...', ...)

# 多模态文档摄入
await g.add_document_episode(name='report', file_path='report.docx')

# 检索
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
results = await g.search_(query='...', config=COMBINED_HYBRID_SEARCH_RRF)
```

---

## 8. 与上游 Graphiti 的主要差异

| 维度 | 上游 Graphiti | 本改造版 |
|------|-------------|---------|
| LLM | OpenAI GPT-4 | Bedrock Kimi K2.5 (Mantle) |
| Embedding | OpenAI text-embedding | Nova Multimodal Embeddings |
| 向量检索 | Neo4j 暴力余弦 | S3 Vectors ANN |
| Reranker | OpenAI | Kimi K2.5 布尔分类 |
| 深度搜索 | 无 | source_excerpt + DescribesEdge + narrative_excerpts |
| 多模态 | 无 | ContentBlock + S3 资产 + 视觉 LLM 描述 |
| 调用日志 | 无 | S3InvocationLogger + Athena |
| 超时重试 | 无 | 按 prompt 类型配置超时 + 自动重试 |
| Episode 去重 | 无 | 内容 embedding 近似查询 |
