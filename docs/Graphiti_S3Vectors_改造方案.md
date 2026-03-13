# Graphiti + S3 Vectors 向量检索改造方案

## 目录

- [1. 背景与问题](#1-背景与问题)
- [2. 改造范围分析](#2-改造范围分析)
- [3. Ingest Pipeline LLM 调用分析](#3-ingest-pipeline-llm-调用分析)
- [4. S3 Vectors 数据结构设计](#4-s3-vectors-数据结构设计)
- [5. 检索流程改造](#5-检索流程改造)
- [6. 深度搜索改造方案](#6-深度搜索改造方案)
- [7. S3 调用日志系统](#7-s3-调用日志系统)
- [8. 测试与可观测性](#8-测试与可观测性)
- [9. TODO](#9-todo)
- [10. 实施概况](#10-实施概况)
- [11. 注意事项](#11-注意事项)
- [12. 深度搜索泛化改造：原文溯源从 Edge 扩展到 Node](#12-深度搜索泛化改造原文溯源从-edge-扩展到-node)
- [13. 多模态数据支持](#13-多模态数据支持)
- [14. 音视频数据处理](#14-音视频数据处理)

---

## 1. 背景与问题

### 1.1 Graphiti 当前向量检索的瓶颈

Graphiti 的向量检索直接在 Neo4j 内部做暴力余弦计算（`vector.similarity.cosine()`），没有使用 ANN 索引：

```cypher
MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
WHERE e.group_id IN $group_ids
WITH DISTINCT e, n, m, vector.similarity.cosine(e.fact_embedding, $search_vector) AS score
WHERE score > $min_score
ORDER BY score DESC
LIMIT $limit
```

- 时间复杂度 O(n)，每次查询遍历所有候选边/节点
- Neptune 后端更差：把所有 embedding 拉到 Python 端逐条计算
- 数据量大时（如整本小说的知识图谱，几千实体 + 几万关系边）性能不可接受
- Neo4j 的原生向量索引（`db.index.vector.queryNodes()`）不支持属性过滤与向量检索同时进行

### 1.2 为什么选择 S3 Vectors

| 维度 | S3 Vectors | OpenSearch | Neptune Analytics |
|------|-----------|------------|-------------------|
| 属性过滤 + 向量检索 | ✅ 搜索中过滤（tandem） | ✅ 三种策略（efficient/post/pre） | ✅ vertexFilter |
| 边上的向量 | ✅ 扁平文档模型 | ✅ | ❌ 只支持节点 |
| 成本 | 极低，Serverless 按量付费 | 中等，需维护集群 | 中等 |
| 运维复杂度 | 极低，纯 API 调用 | 中等 | 中等 |
| 成熟度 | 2025.12 GA | 非常成熟 | 成熟 |

S3 Vectors 的 metadata filtering 在向量搜索过程中同步执行过滤条件验证，不是先搜后过滤，能保证返回满足条件的 top-K 结果。支持 `$eq`、`$ne`、`$gt`、`$lt`、`$in`、`$and`、`$or` 等操作符，完全覆盖 Graphiti 的 `group_id` 过滤需求。

S3 Vectors 适合 POC 验证，后续如需更强的全文检索能力或更高写入吞吐，可平滑迁移到 OpenSearch（S3 Vectors 支持一键导出到 OpenSearch）。

## 2. 改造范围分析

### 2.1 不需要修改的部分

- **数据模型**：`EntityNode`、`CommunityNode` 的 Pydantic 模型字段不变（`EntityEdge` 新增 `source_excerpt`/`source_excerpt_embedding`，`EpisodicNode` 新增 `uncovered_excerpts`，见 Section 6.6）

> **⚠️ 章节12改造后更新**：数据模型变更范围扩大。除上述字段外，还新增了 `DescribesEdge` 类型（Episode→Entity 描述边，见 12.4 Phase 2）；`EpisodicNode` 新增 `describes_edges` 字段；`uncovered_excerpts` 从 `list[str]` 改为 `list[UncoveredExcerpt]` 结构化模型（见 12.4 Phase 1）；存储格式从 `|` 分隔改为 JSON（见 12.8）。
>
> **⚠️ 术语重构**：本节中的 `uncovered_excerpts` / `UncoveredExcerpt` 已在后续重构中统一更名为 `narrative_excerpts` / `NarrativeExcerpt`，详见 12.9.5。
- **Neo4j 图结构**：节点类型、关系类型、属性字段全部保留
- **时序更新逻辑**：`resolve_edge_contradictions()` 纯时间比较，不涉及向量
- **LLM 提取与去重判断**：实体/关系提取、`dedupe_edges.resolve_edge` prompt 调用不变
- **BM25 全文检索**：继续走 Neo4j 的 fulltext index
- **RRF 融合排序**：`rrf()` 函数不变，只是输入来源从 Neo4j 向量检索改为 S3 Vectors
- **图遍历（BFS）**：继续走 Neo4j Cypher

### 2.2 需要修改的部分

#### A. 新增 S3 Vectors Client（新文件）

文件：`graphiti_core/vector_store/s3_vectors_client.py`

职责：
- 管理 5 个 vector index：`entity-name-embeddings`（节点）、`edge-fact-embeddings`（边 fact）、`edge-excerpt-embeddings`（边原文片段）、`community-name-embeddings`（社区）、`episode-narrative-embeddings`（纯叙事文本）
- 提供 `upsert_vectors()`、`query_vectors()`、`delete_vectors()` 方法
- 封装 boto3 S3 Vectors API 调用

> **⚠️ 章节12改造后更新**：索引数量从 5 个增加到 7 个，新增 `describes-fact-embeddings`（DescribesEdge fact 向量）和 `describes-excerpt-embeddings`（DescribesEdge excerpt 向量）。详见 12.4 Phase 3 和 12.9.1。

#### B. 修改向量写入路径

涉及文件：
- `graphiti_core/nodes.py` — `EntityNode.save()`、`CommunityNode.save()`
- `graphiti_core/edges.py` — `EntityEdge.save()`
- `graphiti_core/models/nodes/node_db_queries.py` — 跳过 `setNodeVectorProperty`
- `graphiti_core/models/edges/edge_db_queries.py` — 跳过 `setRelationshipVectorProperty`
- `graphiti_core/utils/bulk_utils.py` — 批量写入时同步写 S3 Vectors

改动：Neo4j 不再存 embedding（复用 `has_aoss=True` 逻辑），embedding 写入 S3 Vectors。

#### C. 修改向量检索路径

涉及文件：
- `graphiti_core/search/search_utils.py` — `edge_similarity_search()`、`node_similarity_search()`、`community_similarity_search()`

改动：向量检索改为调 S3 Vectors query API，拿到 UUID 列表后回 Neo4j 查完整数据。

#### D. 修改向量删除路径

涉及文件：
- `graphiti_core/nodes.py` — `EntityNode.delete()`、`Node.delete_by_group_id()`
- `graphiti_core/edges.py` — `EntityEdge` 删除相关

改动：删除节点/边时同步删除 S3 Vectors 中的向量。

#### E. 修改初始化路径

涉及文件：
- `graphiti_core/graphiti.py` — `Graphiti.__init__()`、`build_indices_and_constraints()`

改动：初始化时创建 S3 Vectors 的 vector bucket 和 index。

## 3. Ingest Pipeline LLM 调用分析

### 3.1 单段落（add_episode）调用流程

每个段落的 ingest 经过以下阶段，每个阶段的 LLM/Embedding 调用次数如下：

| Phase | 阶段 | LLM 调用 | Embedding 调用 | 说明 |
|-------|------|---------|---------------|------|
| 1 | `extract_nodes` | 1 | 0 | 从原文提取实体，单次 LLM 调用 |
| 2 | `resolve_extracted_nodes` | 0~1 | N（节点数） | 先做 embedding 相似度匹配，未匹配的批量发给 LLM 去重（1 次调用处理所有节点） |
| 3 | `_extract_and_resolve_edges` | 1+E（边数） | 2E+ | 先单次 LLM 调用提取关系边 + uncovered excerpts，再对每条边各 1 次 LLM 去重调用（**并行执行**，受 `SEMAPHORE_LIMIT` 控制，默认 20）；每条边还需 embedding 用于相似度搜索 |
| 4 | `extract_attributes_from_nodes` | 若干 | 0 | 为每个节点生成 summary，涉及 LLM 调用 |
| 5 | `_process_episode_data` | 0 | 0 | 写入 Neo4j，纯数据库操作 |
| 6 | S3 Vectors sync | 0 | U（uncovered 数） | 同步向量到 S3 Vectors；uncovered excerpts 需批量 embedding |

> **⚠️ 术语重构**：本节中的 `uncovered excerpts` 已在后续重构中统一更名为 `narrative excerpts`，详见 12.9.5。

如果启用 `update_communities=True`（默认 False），每个节点额外 2 次 LLM 调用（`summarize_pair` + `generate_summary_description`）。

#### 3.1.1 `extract_edges.edge` Prompt 结构

这是单段落 ingest 中最耗时的 LLM 调用，其 prompt 结构如下：

```
<PREVIOUS_MESSAGES>
[ep.content for ep in previous_episodes]  # 最多 RELEVANT_SCHEMA_LIMIT=10 个历史 episode 的完整内容
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
episode.content  # 当前段落原文
</CURRENT_MESSAGE>

<ENTITIES>
[{'name': node.name, 'entity_types': node.labels} for node in nodes]  # 当前段落提取的实体列表
</ENTITIES>

<REFERENCE_TIME>
episode.valid_at  # 用于解析相对时间表述
</REFERENCE_TIME>
```

**`previous_episodes` 的作用**：
- 代词消歧：如果当前段落说"他"，需要前文知道"他"是谁
- 时间连续性：理解事件发生的先后顺序
- 上下文补充：某些关系可能跨段落描述

**重要**：`previous_episodes` 只用于 `extract_edges.edge`，不用于边去重（`dedupe_edges.resolve_edge`）。边去重的输入是从 Neo4j 图数据库查出的已有边，与 `previous_episodes` 无关。

#### 3.1.2 `resolve_extracted_edges` 处理流程

边去重是 Phase 3 中最复杂的部分，处理流程如下：

```
extract_edges.edge (LLM)
    ↓
    输出: N 条新边 + uncovered_excerpts
    ↓
resolve_extracted_edges
    ↓
    1. 内存去重：对 (source_uuid, target_uuid, normalized_fact) 三元组去重
    ↓
    2. Embedding：对所有新边的 fact 做 embedding（批量调用 Nova MME）
    ↓
    3. 对每条新边并行执行：
       ├─ get_between_nodes: 查找同一对节点间的已有边
       ├─ search (related_edges): 用 fact embedding 搜索相似边（用于去重判断）
       └─ search (invalidation_candidates): 全局搜索相似边（用于矛盾检测）
    ↓
    4. resolve_extracted_edge (LLM) x N 并行
       ├─ 输入: EXISTING FACTS + INVALIDATION CANDIDATES + NEW FACT
       └─ 输出: duplicate_facts[], contradicted_facts[]
    ↓
    5. 最终 Embedding：对 resolved_edges 和 invalidated_edges 做 embedding
```

**关键参数**：
- `SEMAPHORE_LIMIT=20`：控制步骤 3-4 的最大并发数
- `RELEVANT_SCHEMA_LIMIT=10`：控制搜索返回的候选边数量

#### 3.1.3 实测 Token 消耗与耗时分布

以三国演义第一回前 3 段落为例（2026-03-03 测试数据，排除超时调用后）：

| prompt_name | 调用次数 | avg | p50 | p90 | max | 说明 |
|-------------|---------|-----|-----|-----|-----|------|
| `extract_edges.edge` | 3 | 19.3s | 28.8s | 28.8s | 28.8s | 输出 1000-5000 tokens |
| `dedupe_nodes.nodes` | 3 | 5.4s | 6.2s | 6.2s | 6.2s | 批量节点去重 |
| `extract_nodes.extract_text` | 3 | 4.9s | 4.7s | 5.8s | 5.8s | 实体提取 |
| `dedupe_edges.resolve_edge` | 67 | 1.2s | 1.0s | 1.9s | 2.7s | 单边去重，并行执行 |

**Token 消耗分析**（`extract_edges.edge`）：

| 段落 | input_tokens | output_tokens | 耗时 | 提取边数 |
|------|-------------|---------------|------|---------|
| 0 | 2147 | 1030 | 9.8s | 13 |
| 1 | 2683 | 5280 | 28.8s | 43 |
| 2 | 3103 | 3021 | 28.8s | ~25 |

**发现**：
- Input tokens 增长不大（每段落增加 ~500 tokens），因为 `previous_episodes` 的内容相对稳定
- Output tokens 是主要变量，取决于提取出的边数量
- 段落 1 输出 5280 tokens（43 条边），是段落 0 的 5 倍

#### 3.1.4 Phase 3 详细耗时分解

以段落 1（43 条边）为例的 Phase 3 耗时分解：

| 子阶段 | 耗时 | 说明 |
|--------|------|------|
| `extract_edges.edge` LLM | 302.0s | 单次 LLM 调用（本次碰到超时） |
| embed_edges | 29.4s | 43 条边的 fact embedding |
| fetch_between_nodes | 0.1s | Neo4j 查询同节点对已有边 |
| search_related | 25.2s | 向量搜索相似边（去重候选） |
| search_invalidation | 24.8s | 向量搜索相似边（矛盾候选） |
| LLM_RESOLVE | 1-2s | 43 个 `dedupe_edges.resolve_edge` 并行调用 |
| final_embed | 9.5s | resolved + invalidated 边的 embedding |

**正常情况下**（无超时），Phase 3 总耗时约 100-150s，其中：
- `extract_edges.edge` LLM 调用占 10-30s
- 向量搜索占 50s
- Embedding 占 40s

#### 3.1.5 Bedrock Mantle 间歇性延迟问题

**现象**：在多次测试中观察到 Bedrock Mantle（Kimi K2.5）存在间歇性的极端延迟，单次调用耗时达到 ~300s 后返回。

**典型案例**（2026-03-03 测试）：

| 时间 | prompt | in_tok | out_tok | 耗时 | 分析 |
|------|--------|--------|---------|------|------|
| 07:04→07:09 | `extract_edges.edge` | 2683 | 5280 | 302.0s | 输出 5280 tokens，但 benchmark 测试同等输出只需 ~15s |
| 07:10→07:15 | `dedupe_edges.resolve_edge` | 495 | 19 | 308.0s | 极端异常：仅 495 input + 19 output |

**关键证据**：
- 308s 的 `dedupe_edges.resolve_edge` 调用：输入只有 495 tokens，输出只有 19 tokens
- 同批次其他 42 个 resolve_edge 调用全部在 0.5-1.9s 内完成
- 这些调用是同时发出的（并行 `semaphore_gather`），说明不是客户端排队
- 302s 和 308s 接近某个超时上限（可能是 Bedrock Mantle 网关层的 300s 超时）

**结论**：
- Kimi K2.5 模型本身处理能力没问题（benchmark 测试证明）
- 问题出在 Bedrock Mantle 网关层偶发性地把请求"卡住"，直到超时才返回
- 这不是"慢"，而是"卡死后超时返回"

**排除超时调用后的正常耗时**：

| prompt_name | avg | p50 | p90 | max |
|-------------|-----|-----|-----|-----|
| `extract_edges.edge` | 19.3s | 28.8s | 28.8s | 28.8s |
| `dedupe_nodes.nodes` | 5.4s | 6.2s | 6.2s | 6.2s |
| `extract_nodes.extract_text` | 4.9s | 4.7s | 5.8s | 5.8s |
| `dedupe_edges.resolve_edge` | 1.2s | 1.0s | 1.9s | 2.7s |

**应对策略**：
1. ✅ 已实现：设置更短的客户端超时（`extract_edges.edge` 60s，其他调用 15s），超时后立即重试
2. 后续可测试其他 Bedrock Mantle 支持的模型（如 DeepSeek V3.2）对比稳定性
3. 已添加 `[LLM_TRACE]` 日志用于持续监控（通过 `GRAPHITI_LLM_TRACE=true` 启用）

**超时配置**（`graphiti_core/llm_client/bedrock_client.py`）：

```python
PROMPT_TIMEOUTS = {
    'extract_edges.edge': 60.0,  # 边提取可能输出大量 tokens
    'default': 15.0,  # 其他所有 prompt
}
```

超时后会自动触发重试（最多 2 次），避免长时间卡死在 Bedrock Mantle 网关层。

#### 3.1.6 Streaming 模式分析

**问题**：是否可以通过 streaming 模式优化 LLM 调用延迟？

**分析**：
- Streaming 模式允许 LLM 逐 token 返回结果，客户端可以更早开始处理
- 但 Graphiti 的 LLM 调用场景（实体提取、边提取、去重判断）都需要完整的结构化 JSON 输出
- 必须等待完整响应才能解析 Pydantic 模型，streaming 无法提前开始后续处理
- 唯一的潜在收益是"感知延迟"降低（用户看到进度），但对实际处理时间无帮助

**结论**：Streaming 模式对 Graphiti ingest pipeline 没有实际优化价值。

#### 3.1.7 优化空间总结

经过详细分析，当前 ingest pipeline 的 LLM 调用已经是核心功能，无法通过精简来优化：

| 调用 | 是否可精简 | 原因 |
|------|-----------|------|
| `extract_edges.edge` | ❌ | 核心功能，`previous_episodes` 用于代词消歧和上下文连续性 |
| `dedupe_edges.resolve_edge` | ❌ | 核心功能，确保图谱数据质量，避免重复和矛盾 |
| `extract_nodes.extract_text` | ❌ | 核心功能，实体提取 |
| `dedupe_nodes.nodes` | ❌ | 核心功能，实体去重 |

**可行的优化方向**：
1. **模型选择**：测试其他 Bedrock Mantle 支持的模型（如 DeepSeek V3.2），对比稳定性和延迟
2. **超时重试**：设置更短的客户端超时（60s），超时后立即重试
3. **并发调优**：根据 Bedrock 限流情况调整 `SEMAPHORE_LIMIT`
4. **批量导入模式**：首次灌入语料时跳过去重逻辑（`skip_dedup=True`），大幅降低 LLM 调用量

### 3.2 调用量估算

以三国演义第一回第 2 段落为例（提取 33 个实体、约 30 条边）：

| 类型 | 调用次数 | 并发度 |
|------|---------|--------|
| LLM 调用 | ~35 次（3 固定 + ~30 边去重 + 若干属性提取） | 边去重阶段最高 20 并发 |
| Embedding 调用 | ~70 次（节点 + 边 + uncovered） | 批量调用 |

### 3.3 性能瓶颈

根据实测数据（3.1.3、3.1.4），真正的瓶颈在 Phase 3 的 `extract_edges.edge` 调用：

**主要瓶颈：边提取的单次 LLM 调用**

| 段落 | 边数 | `extract_edges.edge` 耗时 | Phase 3 总耗时 | 占比 |
|------|------|--------------------------|---------------|------|
| 段落 0 | 13 | 9.8s | ~60s | 16% |
| 段落 1 | 43 | 302.0s（超时）/ 正常 ~30s | ~330s | 91% / 正常 ~30% |
| 段落 2 | ~25 | 28.8s | ~120s | 24% |

**次要瓶颈：向量搜索**
- `search_related` + `search_invalidation`：每段落 50s 左右
- 随边数线性增长（每条边 2 次搜索）

**非瓶颈：边去重的并发 LLM 调用**
- `dedupe_edges.resolve_edge` 平均 1.2s，p90 仅 1.9s
- 虽然有 N 次调用（N=边数），但并行执行（`SEMAPHORE_LIMIT=20`）
- 段落 1 的 43 条边去重总耗时仅 1-2s（并行）

**瓶颈分析**：
1. `extract_edges.edge` 是单次调用，无法并行，output tokens 是主要变量（段落 1 输出 5280 tokens）
2. 碰到 Bedrock Mantle 超时时（~300s），这一次调用就占据了整个 Phase 3 的 90% 时间
3. 正常情况下（无超时），边提取仍占 Phase 3 的 20-30%，是最大单项开销

实测数据（三国演义第一回前 3 段落）：

| 段落 | 实体数 | 边数 | Phase 3 耗时 | 总耗时 |
|------|--------|------|-------------|--------|
| 段落 0（空图） | 18 | 13 | ~60s | 113s |
| 段落 1（图中已有数据） | 33 | 43 | ~330s（含超时）| 276s |
| 段落 2 | ~25 | ~25 | ~120s | 194s |

### 3.4 可调参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| 并发限制 | `SEMAPHORE_LIMIT` | 20 | 控制 `semaphore_gather` 最大并发协程数，降低可减少 Bedrock 限流风险 |
| 并发限制（实例级） | `Graphiti(max_coroutines=N)` | None（使用全局值） | 实例级覆盖 |

## 4. S3 Vectors 数据结构设计

### 4.1 Vector Bucket

```
bucket: graphiti-vectors-{region}
```

### 4.2 Entity Name Embeddings Index

```
index: entity-name-embeddings
dimension: 1024
distance_metric: cosine

每条向量:
  key: "{entity_uuid}"
  data: float[1024]  # name_embedding
  metadata (filterable):
    uuid: string
    group_id: string
    name: string
    created_at: number (unix timestamp)
```

### 4.3 Edge Fact Embeddings Index

```
index: edge-fact-embeddings
dimension: 1024
distance_metric: cosine

每条向量:
  key: "{edge_uuid}"
  data: float[1024]  # fact_embedding
  metadata (filterable):
    uuid: string
    group_id: string
    source_node_uuid: string
    target_node_uuid: string
    fact: string (截断到filterable metadata限制)
    created_at: number (unix timestamp)
```

### 4.4 Community Name Embeddings Index

```
index: community-name-embeddings
dimension: 1024
distance_metric: cosine

每条向量:
  key: "{community_uuid}"
  data: float[1024]  # name_embedding
  metadata (filterable):
    uuid: string
    group_id: string
    name: string
    created_at: number (unix timestamp)
```

> **⚠️ 章节6+12改造后更新**：S3 Vectors 索引从上述 3 个扩展到 7 个。章节6新增了 `edge-excerpt-embeddings`（EntityEdge source_excerpt 向量，见 6.5）和 `episode-narrative-embeddings`（纯叙事文本向量，见 6.8）。章节12新增了 `describes-fact-embeddings` 和 `describes-excerpt-embeddings`（DescribesEdge 向量，见 12.4 Phase 3）。章节12还新增了 `episode-content-embeddings`（Episode 内容向量，用于重复导入检测）。完整索引清单见 12.9.1。

## 5. 检索流程改造

### 5.1 改造前（Neo4j 暴力计算）

```
query text → embedder → query_vector
                          ↓
              Neo4j: MATCH + cosine(embedding, query_vector) → 候选集
                          ↓
              Neo4j: BM25 fulltext search → 候选集
                          ↓
              RRF 融合 → 排序结果
```

### 5.2 改造后（S3 Vectors + Neo4j）

```
query text → embedder → query_vector
                          ↓
              S3 Vectors: query(vector, metadata_filter={group_id}) → UUID 列表
                          ↓
              Neo4j: MATCH WHERE uuid IN $uuids → 完整数据
                          ↓
              Neo4j: BM25 fulltext search → 候选集
                          ↓
              RRF 融合 → 排序结果
```

> **⚠️ 章节12改造后更新**：上述流程仅描述标准搜索（Edge cosine_similarity + BM25）。章节12引入 `deep_search=True` 参数后，搜索流程扩展为：Edge 增加 source_similarity 路（查 `edge-excerpt-embeddings`）；Node 增加 source_similarity 路（查 `describes-excerpt-embeddings`，通过 DescribesEdge 的 `target_node_uuid` 关联回 EntityNode）；同时触发 uncovered excerpts 搜索（查 `episode-narrative-embeddings`）。完整流程见 6.3 深度搜索流程图和 12.4 Phase 4-5。

### 5.3 去重流程中的向量检索改造

`resolve_extracted_edges()` 中的两次 `search()` 调用不需要直接修改，因为它们调用的是 `EDGE_HYBRID_SEARCH_RRF` 配置，最终走到 `edge_similarity_search()`。只要改了底层的 `edge_similarity_search()` 实现，去重流程自动生效。

## 6. 深度搜索改造方案

### 6.1 背景与动机

当前 Graphiti 的检索流程只对 LLM 提取后的 fact 做向量检索，原文（episode_body）在提取阶段使用后就不再参与检索。这在大多数场景下足够，但存在以下局限：

- fact 是 LLM 对原文的理解和概括，可能丢失原文中的细节信息
- 用户如果用接近原文的表述去搜索，fact embedding 的匹配度可能不如原文 embedding
- 默认 limit=10 的情况下，某些关联关系可能被排在 top-10 之外而丢失

### 6.2 设计思路

引入"深度搜索"模式，在 LLM 提取 edge 时同时标注对应的原文片段（`source_excerpt`），对其单独做 embedding 并存入 S3 Vectors。深度搜索时多一路原文向量检索参与 RRF 融合，提高召回率。

关键设计决策：
- 原文向量检索不绑定到每次查询，仅在深度搜索模式下启用
- 深度搜索模式下 limit 翻倍（默认 20），增加召回量
- `source_excerpt` 是 edge 级别的原文片段（一两句话），不是整个 episode，粒度可控

### 6.3 检索流程对比

#### 标准搜索（现有逻辑不变）

```
query "刘备是什么人？"
    │
    ├─ 1. Embedding: query → Nova MME → 1024维向量
    │
    ├─ 2. 两路并行检索 Edge:
    │   ├─ BM25 全文检索 (Neo4j fulltext index)
    │   │   → 对 edge 的 fact 字段做关键词匹配
    │   │   → 返回 2*limit 条候选
    │   │
    │   └─ Cosine 向量检索 (S3 Vectors)
    │       → query embedding vs fact embedding
    │       → metadata filter: group_id
    │       → sim_min_score = 0.6
    │       → 返回 2*limit 条候选
    │
    ├─ 3. RRF 融合排序
    │   → score(doc) = Σ 1/(rank + k)
    │   → 两路都命中的结果得分更高
    │
    └─ 4. 截断 → 返回 top limit(默认10) 条 edge
```

#### 深度搜索（新增）

```
query "刘备是什么人？"
    │
    ├─ 1. Embedding: query → Nova MME → 1024维向量
    │
    ├─ 2. 三路并行检索 Edge:
    │   ├─ BM25 全文检索 (Neo4j fulltext index)
    │   │   → 对 edge 的 fact 字段做关键词匹配
    │   │   → 返回 2*limit 条候选
    │   │
    │   ├─ Cosine 向量检索 - fact (S3 Vectors)
    │   │   → query embedding vs fact embedding
    │   │   → index: edge-fact-embeddings
    │   │   → 返回 2*limit 条候选
    │   │
    │   └─ Cosine 向量检索 - source_excerpt (S3 Vectors) [新增]
    │       → query embedding vs source_excerpt embedding
    │       → index: edge-excerpt-embeddings
    │       → 返回 2*limit 条候选
    │
    ├─ 3. RRF 三路融合排序
    │   → 三路中出现次数越多、排名越靠前的结果得分越高
    │
    └─ 4. 截断 → 返回 top limit(默认20，即标准模式的2倍) 条 edge
```

> **⚠️ 章节12改造后更新**：上述流程仅描述 Edge 维度的深度搜索。章节12将深度搜索泛化为正交参数 `deep_search=True`（见 12.4 Phase 5），自动给 SearchConfig 中已配置的所有搜索对象追加 `source_similarity`：
>
> - Edge: 追加 `EdgeSearchMethod.source_similarity`（查 `edge-excerpt-embeddings`），与上述流程一致
> - Node: 追加 `NodeSearchMethod.source_similarity`（查 `describes-excerpt-embeddings`），通过 DescribesEdge 的 `target_node_uuid` 返回关联的 EntityNode
> - 注意：`_apply_deep_search()` 只对已配置的搜索对象追加，不会凭空启用。如果 SearchConfig 只配了 `edge_config`，则只有 Edge 加 source_similarity，Node 的 DescribesEdge 搜索不会被触发。要同时搜索 DescribesEdge，需使用包含 `node_config` 的配置（如 `COMBINED_HYBRID_SEARCH_RRF`）。
> - 旧的 `EDGE_DEEP_SEARCH_RRF` 配方已标记 deprecated，建议使用 `search_(config=..., deep_search=True)` 替代。

### 6.4 配置参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `SEARCH_LIMIT` | `.env` | `10` | 标准搜索返回条数 |
| `DEEP_SEARCH_LIMIT` | `.env` | `20` | 深度搜索返回条数（标准的2倍） |
| `deep_search` | `Graphiti.search()` 参数 | `False` | 是否启用深度搜索模式 |

### 6.5 S3 Vectors 新增 Index

```
index: edge-excerpt-embeddings
dimension: 1024
distance_metric: cosine

每条向量:
  key: "{edge_uuid}"
  data: float[1024]  # source_excerpt embedding
  metadata (filterable):
    uuid: string
    group_id: string
    source_excerpt: string (截断到200字符)
    created_at: number (unix timestamp)
```

### 6.6 数据模型变更

#### Edge 提取模型（`prompts/extract_edges.py`）

```python
class Edge(BaseModel):
    source_entity_name: str
    target_entity_name: str
    relation_type: str
    fact: str
    source_excerpt: str = Field(
        ...,
        description='The exact sentence(s) from the source text that support this fact'
    )
    valid_at: str | None = None
    invalid_at: str | None = None
```

#### EntityEdge 数据模型（`edges.py`）

在 `EntityEdge` 中新增字段：

```python
source_excerpt: str = Field(default='', description='Original text excerpt supporting this fact')
source_excerpt_embedding: list[float] | None = Field(default=None)
```

> **⚠️ 章节12改造后更新**：`ExtractedEdges.uncovered_excerpts` 的类型从 `list[str]` 改为 `list[UncoveredExcerpt]`。`UncoveredExcerpt` 是结构化模型，包含 `excerpt`（原文片段）、`related_entity`（关联实体名，可选）、`fact`（对该实体的描述摘要，可选）三个字段。当 `related_entity` 非空时，该 excerpt 会被提升为 `DescribesEdge`（Episode→Entity 描述边），而非作为纯 uncovered excerpt 存储。详见 12.4 Phase 1（UncoveredExcerpt 模型）和 Phase 2（DescribesEdge 提升流程）。
>
> **⚠️ 术语重构**：本节中的 `uncovered_excerpts` / `UncoveredExcerpt` 已在后续重构中统一更名为 `narrative_excerpts` / `NarrativeExcerpt`，详见 12.9.5。

### 6.7 改造文件清单

深度搜索相关的改动文件已纳入 Section 10.3 的差异对比范围，改造稳定后统一生成。以下为设计阶段的参考清单：

| 文件 | 说明 |
|------|------|
| `prompts/extract_edges.py` | Edge 模型加 `source_excerpt`；`ExtractedEdges` 加 `uncovered_excerpts` |
| `edges.py` | EntityEdge 加 `source_excerpt` + embedding 字段 |
| `edge_operations.py` | `extract_edges()` 返回 `tuple[list[EntityEdge], list[str]]` |
| `vector_store/s3_vectors_client.py` | 新增 `edge-excerpt-embeddings`、`episode-narrative-embeddings` index |
| `graphiti.py` | source_excerpt/uncovered excerpts 同步；`search()` 加 `deep_search` 参数 |
| `search/search_config.py` | `EdgeSearchMethod` 加 `source_similarity`；`SearchResults` 加 `uncovered_excerpts` |
| `search/search.py` | `edge_search()` 支持 source_similarity；深度搜索自动搜索 uncovered excerpts |
| `search/search_utils.py` | 新增 source/uncovered 桥接函数 |
| `nodes.py` | `EpisodicNode` 新增 `uncovered_excerpts` 字段 |

> **⚠️ 章节12改造后更新**：章节12 Phase 2-4 新增以下文件/改动，未在上表中列出：
>
> | 文件 | 说明 |
> |------|------|
> | `describes_edges.py`（新增） | `DescribesEdge` 类定义（Episode→Entity 描述边），含 `fact`、`excerpt`、`target_node_uuid` 等字段 |
> | `nodes.py` | `EpisodicNode` 新增 `describes_edges` 字段（`list[DescribesEdge]`） |
> | `vector_store/s3_vectors_client.py` | 新增 `describes-fact-embeddings`、`describes-excerpt-embeddings` 两个索引（总计 7 个） |
> | `search/search_config.py` | `NodeSearchMethod` 新增 `source_similarity`；`SearchConfig` 新增 `_apply_deep_search()` 方法 |
> | `search/search_utils.py` | 新增 `s3_vectors_node_source_similarity_search()`（查 describes-excerpt-embeddings → EntityNode） |
> | `graphiti.py` | DescribesEdge 写入/删除同步；`search_()` 新增 `deep_search` 正交参数 |
>
> **⚠️ 术语重构**：上述表格中的 `uncovered_excerpts` 相关符号已在后续重构中统一更名为 `narrative_excerpts`，详见 12.9.5。

### 6.8 Uncovered Excerpts（残余文本保留）

**问题**：LLM 边提取只捕获命名实体之间的关系。描述能力、特征、匿名群体的文本（如"角有徒弟五百余人，云游四方，皆能书符念咒"）会丢失，因为无法表示为两个命名实体之间的三元组。

**方案**：零额外 LLM 成本，搭载在现有 `extract_edges` 调用中。LLM 在提取边的同时输出未被任何 edge 覆盖的原文片段，对其做 embedding 存入 S3 Vectors，深度搜索时作为独立的传统 RAG 补充上下文。

**存储结构**：

| 字段 | 说明 |
|------|------|
| key | `{episode_uuid}:uncovered:{content_hash_8}` (SHA256前8位) |
| embedding | 1024 维向量（Nova MME） |
| metadata.group_id | 分组 ID |
| metadata.episode_uuid | 来源 episode UUID |
| metadata.excerpt | 原文片段（截断至 200 字符） |
| metadata.created_at | 创建时间戳 |

**搜索行为**：仅在深度搜索模式（`deep_search=True` 或 `source_similarity` 在 search methods 中）时触发。结果通过 `SearchResults.uncovered_excerpts` 返回，不参与 RRF 融合排序，作为独立的传统 RAG 补充上下文附加给 LLM，由 LLM 自行判断其价值。

**设计决策**：uncovered excerpts 本质上是非结构化文本片段，与知识图谱的结构化实体关系检索是不同的信息维度。不应将其包装为虚拟 edge 强行参与 RRF 排序，而是保持两条路径各司其职——图谱负责结构化关系，uncovered excerpts 走传统 RAG 路径作为补充。

> **⚠️ 章节12改造后更新**：`uncovered_excerpts` 的类型从 `list[str]` 改为 `list[UncoveredExcerpt]` 结构化模型（见 6.6 注释）。存储格式从 `|` 分隔的纯字符串改为 JSON 序列化。此外，章节12引入了"实体归属提升"机制：当 `UncoveredExcerpt.related_entity` 非空时，该 excerpt 不再作为纯 uncovered excerpt 存储到 `episode-narrative-embeddings` 索引，而是被提升为 `DescribesEdge`（Episode→Entity 描述边），存入 `describes-fact-embeddings` 和 `describes-excerpt-embeddings` 索引。只有 `related_entity` 为空的纯叙事文本才保留在 uncovered excerpts 中。这使得原本"丢失"的实体描述信息（如能力、特征、背景）可以通过 Node 的 `source_similarity` 搜索被召回。详见 12.4 Phase 1-2。
>
> **⚠️ 术语重构**：本节（6.8）中的 `uncovered excerpts` / `uncovered_excerpts` / `UncoveredExcerpt` 已在后续重构中统一更名为 `narrative excerpts` / `narrative_excerpts` / `NarrativeExcerpt`，S3 Vectors key 前缀从 `uncovered:` 改为 `narrative:`，`SearchResults.uncovered_excerpts` 改为 `narrative_excerpts`，详见 12.9.5。

## 7. S3 调用日志系统

### 7.0 LLM 调用超时机制

**背景**：Bedrock Mantle 网关层存在间歇性延迟问题，单次调用可能卡死 300s 后才返回（见 3.1.5）。为避免长时间等待，需要客户端超时机制快速失败并重试。

**实现**（2026-03-03）：

在 `BedrockLLMClient._generate_response()` 中使用 `asyncio.timeout()` 包装 API 调用：

```python
async with asyncio.timeout(timeout_seconds):
    response = await self.client.chat.completions.create(...)
```

**超时配置**：

| Prompt 类型 | 超时时间 | 原因 |
|------------|---------|------|
| `extract_edges.edge` | 60s | 可能输出大量 tokens（如段落 1 输出 5280 tokens） |
| 其他所有 prompt | 15s | 正常情况下都在 10s 内完成 |

**重试逻辑**：
- 超时后抛出 `TimeoutError`，触发重试（最多 2 次）
- 与 JSON 解析错误不同，超时重试不附加错误上下文到 prompt（直接重试即可）
- 日志记录：`[LLM_TRACE] !!! TIMEOUT  prompt=xxx  attempt=N  elapsed=XXs  timeout_limit=XXs`

**预期效果**：
- 正常调用不受影响（都在超时限制内完成）
- 碰到 Bedrock Mantle 卡死时，60s 后快速失败并重试，大概率第二次正常
- 避免单次调用卡死 300s 导致整个 ingest 流程长时间阻塞

**测试脚本**：`examples/sanguo/bench_llm.py`（包含超时机制的端到端验证）

### 7.1 背景与目标

Graphiti 的 ingest pipeline 涉及大量 LLM 和 Embedding 调用，但原始代码没有调用级别的日志记录能力。为了分析性能瓶颈、追踪 token 消耗、排查超时问题，需要一套结构化的调用日志系统。

设计目标：
- 记录每次 LLM/Embedding 调用的 model、prompt_name、token 数、延迟、状态
- 日志写入 S3，支持 Athena 查询分析
- 对 ingest 性能影响最小（缓冲写入，不阻塞主流程）
- 零配置可关闭（不设环境变量即不启用）

### 7.2 架构设计

```
Graphiti.add_episode()
    │
    ├─ BedrockLLMClient.generate_response()
    │   └─ s3_logger.record(operation='llm.generate', prompt_name=..., input_tokens=..., ...)
    │
    ├─ BedrockNovaEmbedder.create()
    │   └─ s3_logger.record(operation='embedding.create', input_tokens=..., ...)
    │
    └─ Graphiti.close()
        └─ s3_logger.flush()  ← 确保残余 buffer 写出

S3InvocationLogger (内存 buffer)
    │
    ├─ buffer 达到 flush_threshold (默认 50 条) → 自动 flush
    │
    └─ flush() → gzip 压缩 → S3 PutObject
        └─ s3://{bucket}/{prefix}/year=YYYY/month=MM/day=DD/hour=HH/{timestamp}_{uuid}.jsonl.gz
```

### 7.3 核心组件

#### S3InvocationLogger

文件：`graphiti_core/logging/s3_logger.py`

- 线程安全的缓冲日志器，使用 `threading.Lock` 保护 buffer
- 每条记录为 `LogRecord` dataclass，包含 timestamp、operation、model_id、prompt_name、group_id、input_tokens、output_tokens、latency_ms、status、error_message、input_preview、metadata
- 写入 S3 时采用 gzip 压缩的 JSON Lines 格式
- S3 路径采用 Hive-style 分区：`year=YYYY/month=MM/day=DD/hour=HH/`，支持 Athena 分区裁剪
- flush 失败时将记录放回 buffer，不丢数据

#### create_s3_logger 工厂函数

文件：`graphiti_core/logging/__init__.py`

- 根据环境变量自动判断是否启用，返回 `S3InvocationLogger | None`
- `S3_LOG_BUCKET` 设置即启用，`S3_LOG_ENABLED=false` 显式关闭

### 7.4 集成点

| 组件 | 集成方式 | 记录内容 |
|------|---------|---------|
| `LLMClient` 基类 | `set_s3_logger()` 方法，子类在 `generate_response()` 中调用 `s3_logger.record()` | operation、model_id、prompt_name、tokens、latency |
| `BedrockLLMClient` | 在 `generate_response()` 成功返回后记录，包含重试累计的 token 数 | `_generate_response()` 返回 `(result, input_tokens, output_tokens)` 三元组 |
| `BedrockNovaEmbedder` | `set_s3_logger()` 方法，在 `create()` 中记录 | 从 Nova 响应中提取 `inputTextTokenCount` |
| `Graphiti.__init__()` | 接受 `s3_logger` 参数，分发给 llm_client 和 embedder | — |
| `Graphiti.close()` | 调用 `s3_logger.flush()` 确保残余数据写出 | — |

### 7.5 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `S3_LOG_BUCKET` | （无，必填） | S3 bucket 名称 |
| `S3_LOG_PREFIX` | `GraphitiLogs` | S3 key 前缀 |
| `S3_LOG_REGION` | `AWS_REGION` | S3 region |
| `S3_LOG_FLUSH_THRESHOLD` | `50` | buffer 满多少条自动 flush |
| `S3_LOG_PREVIEW_CHARS` | `200` | input_preview 截断长度，0 禁用 |
| `S3_LOG_ENABLED` | `true`（如 bucket 已设） | 显式 `false` 关闭 |

### 7.6 Athena 查询分析

日志写入 S3 后，通过 Athena 外部表进行查询分析。创建了两个表：

1. `graphiti.graphiti_llm_invocation_logs` — Graphiti 运行时日志（Hive 分区，需 `MSCK REPAIR TABLE`）
2. `graphiti.bedrock_invocation_logs` — Bedrock 原生调用日志（partition projection，自动分区）

DDL 文件：`graphiti_core/logging/athena_ddl.sql`
查询指南：`graphiti_core/logging/README.md`

关键发现：Bedrock Mantle（OpenAI 兼容端点）的调用（如 Kimi K2.5）不会出现在 `bedrock_invocation_logs` 中，因为 Mantle 不走标准 `InvokeModel` API。LLM 调用的 token 数只能从 `graphiti_llm_invocation_logs` 获取。

### 7.7 Token 记录修复

初始实现中 `BedrockLLMClient._generate_response()` 只返回 `dict`，未返回 token 数。修复后：

- `_generate_response()` 返回 `tuple[dict, int, int]`（result, input_tokens, output_tokens），从 OpenAI SDK 响应的 `response.usage.prompt_tokens` / `completion_tokens` 提取
- `generate_response()` 累计重试过程中的 token 数，传给 `token_tracker.record()` 和 `s3_logger.record()`
- `BedrockNovaEmbedder.create()` 从 Nova 响应中提取 `inputTextTokenCount` 传给 `s3_logger.record()`

### 7.8 S3 Logger 相关文件参考

| 文件 | 说明 |
|------|------|
| `graphiti_core/logging/__init__.py` | 模块入口，导出 `create_s3_logger` |
| `graphiti_core/logging/s3_logger.py` | `S3InvocationLogger` 核心实现 |
| `graphiti_core/logging/athena_ddl.sql` | Athena 建表 DDL |
| `graphiti_core/logging/README.md` | Athena 查询指南 |
| `graphiti_core/llm_client/client.py` | 基类新增 `s3_logger` 属性 |
| `graphiti_core/llm_client/bedrock_client.py` | token 三元组返回；日志记录 |
| `graphiti_core/embedder/bedrock_nova.py` | embedding 调用日志记录 |
| `graphiti_core/graphiti.py` | `s3_logger` 分发与 flush |

## 8. 测试与可观测性

### 8.1 测试环境

- EC2 instance，IAM role: `admin-role`，region: `us-east-1`
- Neo4j 5.x（docker compose 本地部署）
- LLM: Kimi K2.5（via Bedrock Mantle）
- Embedder: Amazon Nova Multimodal Embeddings v1（dim=1024）
- S3 Vectors bucket: `graphiti-vectors-poc`

### 8.2 S3 Vectors 基础功能验证（2026-03-02）

测试脚本：`examples/quickstart/quickstart_bedrock.py`

测试流程：创建 S3 Vectors bucket + 3 个 index → 写入 3 个 episodes → Edge Search → Node Search → S3 Vectors 直接查询验证

验证项：
- ✅ S3 Vectors bucket/index 创建
- ✅ Episode 写入（Neo4j + S3 Vectors 双写）
- ✅ Edge Search（via S3 Vectors）— 查询 "Who works at TechCorp?" 正确返回 5 条相关 facts
- ✅ Node Search（via S3 Vectors）— 查询 "TechCorp" 正确返回 5 个相关节点
- ✅ S3 Vectors 直接查询验证 — cosine score 分布合理（top1 0.9996，top5 0.67+）

#### 实现过程中修复的问题

1. **Pydantic forward reference**: `GraphitiClients` 使用 `TYPE_CHECKING` 延迟导入 `S3VectorsClient` 导致运行时 `model_rebuild()` 错误。修复：改为直接导入并在类定义后调用 `GraphitiClients.model_rebuild()`
2. **S3 Vectors API 名称**: boto3 的 S3 Vectors client 方法名是 `create_index`/`get_index`/`delete_index`，不是 `create_vector_index` 等
3. **`create_index` 必需参数**: 需要 `dataType='float32'` 参数
4. **异常错误码**: bucket/index 不存在时抛出 `NotFoundException`，不是 `VectorBucketNotFoundException`

### 8.3 三国演义端到端测试（2026-03-02 ~ 03-03）

#### 测试脚本体系

目录：`examples/sanguo/`

| 脚本 | 功能 | 用法 |
|------|------|------|
| `1_clear.py` | 清空 Neo4j 数据 + 重建 S3 Vectors 索引 | `./run.sh 1` |
| `2_ingest.py N` | 导入三国演义第一回前 N 段落 | `./run.sh 2 3` |
| `3_search.py` | 标准搜索 + 深度搜索测试 | `./run.sh 3` |
| `4_uncovered.py` | 查看 uncovered excerpts | `./run.sh 4` |
| `bench_llm.py` | LLM 延迟基准测试 | `uv run python examples/sanguo/bench_llm.py` |
| `run.sh` | Shell 包装器，检查进程冲突后执行 | `./run.sh all 3` |

> **⚠️ 章节12改造后更新**：脚本体系已更新。原 `4_uncovered.py` 拆分为 `4_describes.py`（展示 DescribesEdge 完整内容，通过 `DescribesEdge.get_by_entity_uuid()` 获取指定实体的描述边）和 `5_uncovered.py`（展示纯 uncovered excerpts）。`3_search.py` 改用 `COMBINED_HYBRID_SEARCH_RRF` 配置（edge + node），使 `deep_search=True` 能同时触发 Edge source_similarity 和 Node source_similarity（DescribesEdge 搜索）。`run.sh` 支持步骤 1-5。
>
> **⚠️ 术语重构**：本节（8.3）中的 `uncovered excerpts` 已在后续重构中统一更名为 `narrative excerpts`（episode narratives），脚本 `5_uncovered.py` 展示的是 `narrative_excerpts` 属性，详见 12.9.5。

`run.sh` 在执行前会检查是否有残留的 sanguo 测试进程，避免多进程同时操作 Neo4j 和 S3 导致数据竞争。

`common.py` 提供共享工具：Graphiti 客户端构建（集成 Bedrock LLM + Nova Embedding + S3 Vectors + S3 Logger）、语料加载、清库、导入、统计等。

#### 导入结果

语料：三国演义第一回前 3 段落，group_id: `sanguo-test`

| 段落 | 耗时 | 说明 |
|------|------|------|
| 段落 0 | 118.3s | 汉朝衰落、桓灵二帝、宦官弄权 |
| 段落 1 | 276.2s | 灾异频发、蔡邕上疏、十常侍 |
| 段落 2 | 194.4s | 张角得道、太平道、黄巾起义 |
| 合计 | 588.9s | 75 实体, 3 episodes, 70 边（source_excerpt 和 uncovered excerpts 数据待重新验证，早期测试中 source_excerpt 传递存在 bug，已修复） |

#### 搜索结果

| 查询 | 模式 | 结果 |
|------|------|------|
| 张角是什么人？ | 标准 | 10 条边，命中张角兄弟关系、太平道人、大贤良师、天公将军等 |
| 黄巾起义的原因 | 标准 | 10 条边，命中黄巾军、高祖起义、灾异、马元义、宦官弄权等 |
| 念咒者何人 | 标准 | 10 条边，未直接命中"念咒"相关内容 |
| 念咒者何人 | 深度 | 20 条边 + 20 uncovered excerpts，首条 uncovered: "角有徒弟五百余人，云游四方，皆能书符念咒"（score=0.787）|

深度搜索对"念咒者何人"这类用接近原文表述的查询有明显优势——标准搜索完全无法命中，深度搜索通过 uncovered excerpts 向量检索精准召回。

### 8.4 Ingest 可观测性

#### 进度日志

在 `Graphiti.add_episode()` 中添加了 6 阶段进度日志，每个阶段记录耗时和关键计数：

```
[add_episode] Phase 1/6: Extracting nodes...
[add_episode] Phase 1/6: Extracted 33 nodes (12.9s)
[add_episode] Phase 2/6: Resolving nodes...
[add_episode] Phase 2/6: Resolved to 33 nodes (34.8s)
[add_episode] Phase 3/6: Extracting & resolving edges...
[add_episode] Phase 3/6: 32 resolved, 0 invalidated, 32 new edges, 5 uncovered excerpts (300.9s)
[add_episode] Phase 4/6: Extracting node attributes...
[add_episode] Phase 4/6: Hydrated 33 nodes (14.1s)
[add_episode] Phase 5/6: Saving to graph...
[add_episode] Phase 5/6: Saved (33 episodic edges) (0.3s)
[add_episode] Phase 6/6: Syncing to S3 Vectors...
[add_episode] Phase 6/6: Synced 33 nodes, 32 edges, 5 uncovered excerpts (41.7s)
```

> **⚠️ 术语重构**：上述日志示例中的 `uncovered excerpts` 在当前代码中已更名为 `episode narratives`，详见 12.9.5。

#### 噪音日志抑制

抑制了以下第三方库的 INFO 级别日志，保持前台输出清晰：

| 库 | 抑制原因 |
|---|---------|
| `neo4j` | 大量连接和查询日志 |
| `botocore.credentials` | IAM 凭证刷新日志 |
| `httpx` | OpenAI SDK 的 HTTP 请求日志 |

### 8.5 S3 Vectors 基础改造文件参考

以下为 S3 Vectors 初始改造（写入/检索/删除路径）涉及的文件，完整变更清单待改造稳定后通过 `git diff` 生成（见 10.3）。

| 文件 | 改动类型 |
|------|---------|
| `graphiti_core/vector_store/__init__.py` | 新增 |
| `graphiti_core/vector_store/s3_vectors_client.py` | 新增 |
| `graphiti_core/graphiti_types.py` | 修改 |
| `graphiti_core/graphiti.py` | 修改 |
| `graphiti_core/search/search.py` | 修改 |
| `graphiti_core/search/search_utils.py` | 修改 |
| `examples/quickstart/quickstart_bedrock.py` | 修改 |
| `deploy/s3-vectors-policy.json` | 新增 |

### 8.6 单元测试（mock 模式，无外部依赖）

全部使用 mock 模式，不依赖任何外部系统（无 AWS、Neo4j、LLM 调用），可在无网络环境下运行。

分为两部分：本地新增（100 cases）和社区原有（166 cases），合计 266 cases。

> **⚠️ 章节12改造后更新**：章节12的 DescribesEdge 相关测试已集成到现有测试文件中（`test_deep_search.py` 和 `test_s3_vectors_data.py`），未新增独立测试文件。`test_s3_vectors_data.py` 中 `TestS3VectorsConfig` 已更新为验证 7 个索引（含 `describes-fact-embeddings` 和 `describes-excerpt-embeddings`）。当前全量单元测试（含社区新增）共 422 cases。

```bash
# 运行全部 266 个纯单元测试（~16s）
DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 \
.venv/bin/python -m pytest tests/test_s3_logger.py tests/llm_client/test_bedrock_client.py \
  tests/test_deep_search.py tests/test_s3_vectors_data.py \
  tests/utils/ tests/test_text_utils.py \
  tests/llm_client/test_token_tracker.py tests/llm_client/test_cache.py \
  tests/llm_client/test_errors.py tests/llm_client/test_client.py \
  tests/helpers_test.py -v
```

#### 8.6.1 `tests/test_s3_logger.py` — S3 调用日志系统（18 cases）

对应章节 7。外部依赖 mock：`boto3.client('s3')` → `MagicMock`。

| 测试类 | 覆盖内容 |
|--------|---------|
| TestLogRecord | 默认值、序列化往返 |
| TestS3InvocationLogger | 缺 bucket 报错、环境变量初始化、缓冲写入、preview 截断/禁用、auto-flush 阈值、gzip JSON Lines 格式 + Hive 分区 key、空 buffer noop、flush 失败回填、None 字段过滤、latency 四舍五入 |
| TestCreateS3Logger | 显式禁用、无 bucket 返回 None、有 bucket 自动启用、创建异常返回 None |

#### 8.6.2 `tests/llm_client/test_bedrock_client.py` — LLM 超时/重试（28 cases）

对应章节 7。外部依赖 mock：`_generate_bedrock_token` → `'fake-token'`，`AsyncOpenAI.chat.completions.create` → `AsyncMock`。

| 测试类 | 覆盖内容 |
|--------|---------|
| TestPromptTimeouts | 5 类超时默认值（DEFAULT/NODE_EXTRACT_TXT/MM/EDGE_EXTRACT_TXT/MM）、未知 prompt 回退、env-var override（5 个环境变量）、无效值 fallback |
| TestLLMTraceEnabled | 环境变量 true/1/false/未设置 |
| TestGenerateResponseInternal | 成功返回三元组、JSON 错误、asyncio.timeout 超时、429→RateLimitError |
| TestGenerateResponseOuter | 首次成功、JSON 错误重试（追加 error context）、超时重试（不追加）、MAX_RETRIES 耗尽、RateLimitError 不重试、token 累积行为、s3_logger 成功/失败时的调用 |

#### 8.6.3 `tests/test_deep_search.py` — 深度搜索（26 cases）

对应章节 5、6、8.10。外部依赖 mock：`boto3.client('s3vectors')` → `MagicMock`，`EntityEdge.get_by_uuids` / `EntityNode.get_by_uuids` / `CommunityNode.get_by_uuids` → `AsyncMock`。

| 测试类 | 覆盖内容 |
|--------|---------|
| TestS3VectorsClientQueryVectors | distance→similarity 转换、min_score 过滤、单/多 group_id filter、edge 复合 $and filter、top_k 上限 100 |
| TestS3VectorsEdgeSearch | S3→Neo4j 取回 + 排序保持、空结果、Neo4j 缺失记录跳过 |
| TestS3VectorsEdgeSourceSearch | source_excerpt index 路由 |
| TestS3VectorsNodeSearch | 排序保持 |
| TestS3VectorsCommunitySearch | 排序保持 |
| TestUncoveredExcerptSearch | 标点去重、空结果、返回 dict 结构 |
| TestEdgeSearchRouting | cosine_similarity/source_similarity 路由到 S3 Vectors、无 s3_vectors 时 source_similarity 无结果 |
| TestNodeSearchRouting | cosine_similarity 路由到 S3 Vectors |
| TestCommunitySearchRouting | cosine_similarity 路由到 S3 Vectors |
| TestSearchUncoveredExcerptsTrigger | 三种触发条件（有 s3_vectors + source_similarity 才触发） |
| TestExtractEdgesEmptyInput | 空边返回 uncovered、无效实体名过滤、有效边 source_excerpt 保留 |

> **⚠️ 术语重构**：上述测试类名和描述中的 `Uncovered` / `uncovered` 已在后续重构中统一更名为 `Narrative` / `narrative`，详见 12.9.5。

#### 8.6.4 `tests/test_s3_vectors_data.py` — S3 Vectors 数据结构 & Ingest 同步（37 cases）

对应章节 4、5。外部依赖 mock：`boto3.client('s3vectors')` → `MagicMock`，`Graphiti` 实例的 `s3_vectors`/`embedder` → `MagicMock`。

| 测试类 | 覆盖内容 |
|--------|---------|
| TestS3VectorsConfig | 5 个 index 默认名称、index 数量 |
| TestUpsertVectors | 单条写入、空 metadata 不传、超 MAX_BATCH_SIZE 分批、空列表 noop |
| TestUpsertEntityVector | metadata 字段验证 |
| TestUpsertEdgeVector | metadata 字段验证、fact 截断 200 |
| TestUpsertEdgeSourceVector | metadata 字段验证、excerpt 截断 200 |
| TestUpsertUncoveredExcerptVector | key 格式、metadata 字段验证 |
| TestUpsertCommunityVector | metadata 字段验证 |
| TestDeleteVectors | 按 key 删除、空列表 noop、分批删除、5 种 typed delete 路由 |
| TestLifecycle | ensure_bucket_and_indices 创建/跳过、delete_all_indices |
| TestSyncNodesToS3Vectors | 有 embedding 同步、无 embedding 跳过、s3_vectors=None noop |
| TestSyncEdgesToS3Vectors | fact+source_excerpt 双写、无 excerpt 只写 fact、无 embedding 全跳过、noop |
| TestSyncCommunitiesToS3Vectors | 有 embedding 同步、无 embedding 跳过 |
| TestSyncUncoveredExcerptsToS3Vectors | embed+upsert+key 格式、空字符串过滤、s3_vectors=None noop、空列表 noop |

> **⚠️ 术语重构**：上述测试类名中的 `Uncovered` 已在后续重构中统一更名为 `Narrative`，详见 12.9.5。

#### 8.6.5 社区原有单元测试（166 cases）

社区版本提供的纯单元测试，全部 mock 外部依赖，无需 Neo4j/LLM/AWS。

| 文件 | 测试数 | 覆盖内容 |
|------|--------|---------|
| `tests/utils/test_content_chunking.py` | 55 | 文本/JSON 分块、token 估算、密度估计、覆盖分块 |
| `tests/utils/maintenance/test_node_operations.py` | 23 | 节点提取、解析、去重 |
| `tests/utils/maintenance/test_entity_extraction.py` | 18 | 实体提取、summary 批量生成、prompt 选择 |
| `tests/llm_client/test_token_tracker.py` | 16 | token 用量追踪、prompt 级别统计 |
| `tests/utils/maintenance/test_bulk_utils.py` | 14 | 批量写入工具函数 |
| `tests/test_text_utils.py` | 11 | 文本处理工具函数 |
| `tests/llm_client/test_cache.py` | 10 | LLM 响应缓存 |
| `tests/utils/maintenance/test_edge_operations.py` | 6 | 边提取、去重、矛盾检测 |
| `tests/llm_client/test_errors.py` | 6 | LLM 错误类型（RefusalError、RateLimitError、EmptyResponseError） |
| `tests/utils/search/search_utils_test.py` | 5 | 搜索工具函数 |
| `tests/llm_client/test_client.py` | 1 | LLM client 基类 |
| `tests/helpers_test.py` | 1 | Lucene 查询转义 |

### 8.7 集成测试（真实 Neo4j + mock LLM/Embedder）

采用社区 `test_graphiti_mock.py` 的模式：连接真实 Neo4j，mock 掉 LLM/Embedder/S3 Vectors，无 AWS 调用。分为三部分：本地新增（15 cases）、社区 mock 测试（24 cases）、社区 add_triplet 测试（11 cases），合计 50 cases。

```bash
# 运行全部 50 个集成测试（需本地 Neo4j，~4s）
DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 \
.venv/bin/python -m pytest tests/test_integration.py tests/test_graphiti_mock.py tests/test_add_triplet.py -v
```

架构要点：
- Neo4j：真实连接，真实 Cypher 查询
- LLM：mock（`generate_response` 返回预设 dict）
- Embedder：mock（返回确定性向量）
- S3 Vectors：mock（`Mock(spec=S3VectorsClient)`）
- Cross Encoder：mock

#### 8.7.1 `tests/test_integration.py` — 本地集成测试（15 cases）

新编写的集成测试，覆盖 Node/Edge CRUD、搜索、S3 Vectors 同步、删除清理、多 group 隔离、source_excerpt 持久化。

| 测试函数 | 覆盖内容 |
|---------|---------|
| test_save_and_retrieve_nodes | add_nodes_and_edges_bulk → Neo4j 持久化，节点/边计数验证 |
| test_node_fulltext_search | BM25 全文检索命中验证 |
| test_edge_fulltext_search | 边 fact 全文检索命中验证 |
| test_node_similarity_search | 节点向量相似度检索（Neo4j 暴力计算） |
| test_edge_similarity_search | 边 fact 向量相似度检索 |
| test_sync_nodes_to_s3_vectors | 节点同步到 S3 Vectors（mock）验证 upsert 调用 |
| test_sync_edges_to_s3_vectors | 边 fact 同步到 S3 Vectors |
| test_sync_edges_with_source_excerpt | 边 source_excerpt 双写验证（fact + source 两次 upsert） |
| test_sync_communities_to_s3_vectors | 社区节点同步到 S3 Vectors |
| test_sync_uncovered_excerpts | uncovered excerpts embed + upsert 验证 |
| test_sync_skips_when_s3_vectors_is_none | s3_vectors=None 时同步跳过（降级验证） |

> **⚠️ 术语重构**：上述测试函数名中的 `uncovered` 已在后续重构中统一更名为 `narrative`，详见 12.9.5。
| test_get_mentioned_nodes | episode → entity 的 episodic edge 关联验证 |
| test_remove_episode_cleans_neo4j | remove_episode 清理 Neo4j 中的 episode + episodic edges |
| test_group_isolation_in_search | 不同 group_id 的数据搜索隔离 |
| test_edge_source_excerpt_persisted | source_excerpt 字段写入 Neo4j 后可读回 |

#### 8.7.2 `tests/test_graphiti_mock.py` — 社区 mock 集成测试（24 cases）

社区提供的 mock 集成测试，使用真实 Neo4j + mock LLM。覆盖 `add_episode` 端到端流程、搜索、节点/边 CRUD、社区聚类等。

注意：`test_get_community_clusters` 需要 Neo4j 中无残留 Entity 数据，否则标签传播算法可能超时。测试前建议清空测试 group 数据。

#### 8.7.3 `tests/test_add_triplet.py` — 社区 add_triplet 测试（11 cases，已修复）

社区提供的 `add_triplet` 测试，原始版本因数据结构过时全部失败。修复内容：
1. mock LLM 返回值增加 `entity_resolutions: []`（匹配当前 `NodeResolutions` schema）
2. labels 断言增加 `'Entity'`（`resolve_extracted_nodes` 会添加默认 label）
3. 3 个无效 UUID 测试从 `pytest.raises(ValueError)` 改为验证新节点创建（当前代码回退到 `resolve_extracted_nodes`）
4. `helpers_test.py` 的 `mock_embedder` fixture 增加 `create_batch` async 方法

### 8.8 不适用的社区测试

以下社区测试文件因 provider 不匹配或功能不适用，不纳入日常测试：

| 文件 | 原因 |
|------|------|
| `tests/llm_client/test_anthropic_client.py` | Anthropic provider，我们使用 Bedrock |
| `tests/llm_client/test_anthropic_client_int.py` | Anthropic provider 集成测试（需 API key） |
| `tests/llm_client/test_gemini_client.py` | Google Gemini provider |
| `tests/llm_client/test_azure_openai_client.py` | Azure OpenAI provider |
| `tests/cross_encoder/test_gemini_reranker_client.py` | Gemini reranker |
| `tests/cross_encoder/test_bge_reranker_client.py` | BGE reranker（本地模型） |
| `tests/embedder/test_voyage_embedder.py` | Voyage embedder |
| `tests/test_graphiti_int.py` | 需要 OpenAI API key（默认 embedder） |
| `tests/test_graphiti_falkordb.py` | FalkorDB driver |
| `tests/test_graphiti_kuzu.py` | Kuzu driver |
| `tests/test_graphiti_neptune.py` | Neptune driver |
| `tests/driver/test_falkordb_driver.py` | FalkorDB driver |

### 8.9 测试汇总

| 类别 | 测试数 | 外部依赖 | 耗时 | 运行命令 |
|------|--------|---------|------|---------|
| 8.6 纯单元测试（本地） | 109 | 无 | ~0.5s | 见 8.6 |
| 8.6 纯单元测试（社区） | 166 | 无 | ~16s | 见 8.6 |
| 8.7 集成测试（本地） | 15 | Neo4j | ~2s | 见 8.7 |
| 8.7 集成测试（社区 mock） | 24 | Neo4j | ~1.5s | 见 8.7 |
| 8.7 集成测试（社区 add_triplet） | 11 | Neo4j | ~0.5s | 见 8.7 |
| 合计 | 325 | — | ~32s | — |

> **⚠️ 章节12改造后更新**：章节12完成后，全量测试（排除需真实 OpenAI/Neo4j 的集成测试）共 422 cases。增量来自章节12的 DescribesEdge 测试集成到现有文件，以及社区版本后续新增的测试。
>
> **⚠️ 章节13改造后更新**：LLM 超时配置化（13.16.10）后，`TestPromptTimeouts` 从 3 cases 扩展到 12 cases（5 类默认值 + 5 个 env-var override + 1 个未知 prompt 回退 + 1 个无效值 fallback），全量测试共 429 cases, 4 skipped。

一键运行全部测试：

```bash
DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 \
.venv/bin/python -m pytest tests/test_s3_logger.py tests/llm_client/test_bedrock_client.py \
  tests/test_deep_search.py tests/test_s3_vectors_data.py \
  tests/utils/ tests/test_text_utils.py \
  tests/llm_client/test_token_tracker.py tests/llm_client/test_cache.py \
  tests/llm_client/test_errors.py tests/llm_client/test_client.py \
  tests/helpers_test.py \
  tests/test_integration.py tests/test_graphiti_mock.py tests/test_add_triplet.py -v
```

### 8.10 测试 TODO（功能验证）

部分测试项已被 8.6/8.7（单元测试和集成测试）覆盖，剩余为手动 benchmark 或待功能实现后补充。

| 类别 | 测试项 | 优先级 | 状态 | 说明 |
|------|--------|--------|------|------|
| 数据完整性 | source_excerpt 端到端验证 | 高 | ✅ 8.7 覆盖 | `test_edge_source_excerpt_persisted`、`test_sync_edges_with_source_excerpt` |
| 数据完整性 | uncovered_excerpts 持久化验证 | 高 | ✅ 8.7 覆盖 | `test_sync_uncovered_excerpts`（mock S3 Vectors 验证） |

> **⚠️ 术语重构**：上述测试名中的 `uncovered_excerpts` 已在后续重构中统一更名为 `narrative_excerpts`，详见 12.9.5。
| 检索质量 | 标准搜索 vs 深度搜索对比 | 高 | ⬜ 手动 | 需真实 LLM + S3 Vectors，适合端到端 benchmark |
| 检索质量 | S3 Vectors vs Neo4j 暴力计算结果对比 | 中 | ⬜ 手动 | 需构造双路径对比，适合手动 benchmark |
| 性能 | 大数据量导入性能 | 中 | ⬜ 手动 | 导入三国演义完整第一回，观察去重耗时变化 |
| 性能 | 搜索延迟基准 | 中 | ⬜ 手动 | 记录不同数据量下 S3 Vectors 查询延迟 |
| 删除 | remove_episode 端到端验证 | 中 | ✅ 8.7 覆盖 | `test_remove_episode_cleans_neo4j` |
| 边界情况 | 空段落 / 无实体段落 | 低 | ✅ 8.6 覆盖 | `TestExtractEdgesEmptyInput`（空边、无效实体名过滤、source_excerpt 保留） |
| 边界情况 | S3 Vectors 不可用时的降级 | 低 | ✅ 8.7 覆盖 | `test_sync_skips_when_s3_vectors_is_none` |
| 批量导入 | skip_dedup 模式 | 低 | ⬜ 待实现 | 待 skip_dedup 功能实现后补充 |

## 9. TODO

### 9.1 一次性批量导入（skip dedup）— 未实现

**问题**：当前 `add_episode()` 每导入一段文本都会对提取出的 edge 执行去重流程（`resolve_extracted_edges`），包括：对每条 edge 做 BM25 + 向量检索找候选重复项，再调用 LLM 判断是否重复。当 Neo4j 中已有数据时，去重开销极大（实测单段文本对 200+ 条 edge 去重耗时 800s+）。

**需求**：提供一次性导入模式，适用于首次灌入语料的场景。跳过去重逻辑，直接写入 Neo4j 和 S3 Vectors。

**初步方案**：
- `add_episode()` 新增 `skip_dedup: bool = False` 参数
- 当 `skip_dedup=True` 时，跳过 `resolve_extracted_edges()` 和 `resolve_extracted_nodes()`，直接将提取结果写入
- 调用方需自行保证数据不重复（如清库后首次导入）
- 批量导入脚本在调用前先清空 group_id 对应的数据

### 9.2 Episode Narratives 后续处理与质量优化

**背景**：narrative excerpts（原 uncovered excerpts）是 episode 中未能被结构化为实体关系的文本片段。这些片段本质上分为两类：

1. **纯叙事性描述**：如"话说天下大势，分久必合，合久必分"，本身与知识图谱无关，仅作为原文溯源的补充
2. **上下文缺失导致的关联丢失**：如某段文本提到了一个事件但当前 episode 上下文中缺少相关命名实体，LLM 无法建立关联。随着后续 episode 引入新实体、图谱信息逐步健全，这些片段有机会被重新关联到图谱中

**已完成的基础设施**：

- ✅ Neo4j 持久化：`EpisodicNode.narrative_excerpts` 属性，所有 driver 已适配
- ✅ S3 Vectors 存储：基于内容哈希的 key（`{episode_uuid}:narrative:{content_hash_8}`），支持关联清除
- ✅ 深度搜索检索：`episode-narrative-embeddings` 索引，向量相似度召回

**后续优化方向**（不急，待合适时机迭代）：

#### 1. 补全机制

定期或触发式扫描流程，利用已健全的图谱重新尝试关联：
1. 查询有 `narrative_excerpts` 的 episode 节点
2. 对每条片段，用当前图谱中的实体列表重新尝试 LLM 提取
3. 成功提取出新的实体关系则写入图谱，并从 episode 的 `narrative_excerpts` 和 S3 Vectors 中清除
4. 也可结合专业人员人工补充额外信息，辅助建立关联

#### 2. 独立 limit 控制

当前 narrative excerpts 搜索复用了 `SearchConfig.limit`（与 edge 搜索共享）。图谱中的 edge 是高价值结构化信息，而 narrative excerpts 仅依赖向量相似度排序，噪音概率更高。应新增独立的 `narrative_limit` 参数（建议默认 5~10），与 edge limit 解耦。

#### 3. 质量过滤

当前 LLM 输出的 narrative excerpts 中，部分片段的核心信息已被实体节点或边的 `source_excerpt` 覆盖，保留价值低。优化方向：
- Prompt 层面：指引 LLM 排除已被实体名或边 fact 覆盖的句子
- 后处理层面：写入 S3 Vectors 前，与当前 episode 产生的实体名/边 fact 做文本匹配过滤
- 长度阈值：过短片段（如 < 10 字）信息密度极低，可直接过滤

**优先级**：中。当前 POC 阶段基础设施已完备，上述优化待核心链路稳定、有实际质量需求时再迭代。

### 9.3 TokenTracker 在 JSON 解析失败时丢失 token 计数

`BedrockLLMClient._generate_response()` 中，`json.loads(result)` 抛异常发生在 `return parsed, input_tokens, output_tokens` 之前，导致该次调用的 token 用量无法传递给外层 `generate_response()` 的累加逻辑。实际 API 调用已完成、Bedrock 已计费，但 TokenTracker 不会记录这笔消耗。

**影响**：tracker 报告的 token 用量略低于实际账单。当前阶段日志系统主要用于排查问题而非精准计费，实际账单对账应使用 AWS Cost Explorer 或 Bedrock Invocation Log。

**优先级**：低。

### 9.4 提示词场景化模板（Prompt Templates）

**背景**：当前 `extract_nodes` 和 `extract_edges` 的提示词是面向对话场景设计的（如 Speaker Extraction、conversational messages），对叙事文本（小说、报告、法规等）的实体提取效果不够理想。实测三国演义语料时，段落中的事件/现象描述（如"大青蛇从梁上飞将下来"、"雌鸡化雄"、"冰雹"）未被提取为实体，导致大量有价值的内容落入 narrative excerpts 而非结构化图谱。

**观察**：12.9.5 rename 后 narrative excerpts 数量增多，说明提示词措辞的变化确实影响了 LLM 的提取行为。这进一步印证了提示词优化对图谱质量的直接影响。

**设计思路**：

将提示词按使用场景抽象为可选模板，用户根据语料类型选择最合适的模板：

| 模板 | 适用场景 | 节点提取侧重 | 边提取侧重 |
|------|---------|-------------|-----------|
| `conversation` | 对话、聊天记录 | 说话人 + 提及实体（当前默认） | 人物间关系、事件 |
| `narrative` | 小说、历史文献、新闻 | 人物 + 地点 + 事件/现象 | 因果关系、时序关系、参与关系 |
| `document` | 技术文档、法规、报告 | 概念 + 组织 + 条款 | 定义关系、约束关系、引用关系 |
| `custom` | 用户自定义 | 用户提供 entity_types + 提取指引 | 用户提供 relation_types + 提取指引 |

实现方式：
- `extract_nodes.py` / `extract_edges.py` 中的 prompt 函数接受 `template` 参数
- 每个模板对应一组 system prompt + user prompt 的变体
- 通过 `GraphitiConfig` 或 `add_episode()` 参数传入

**优先级**：低。当前 POC 阶段以功能验证为主，提示词模板化属于质量优化方向，待核心链路稳定后再迭代。

## 10. 实施概况

本项目基于 [getzep/graphiti](https://github.com/getzep/graphiti) 社区版本进行改造。改造涉及多个方面，且在持续迭代中，逐条维护实施步骤容易过时。改造稳定后，将基于社区版本做 `git diff` 生成完整的变更清单。

### 10.1 改造方向总览

| 方向 | 状态 | 核心改动 |
|------|------|---------|
| S3 Vectors 向量检索 | ✅ 已完成 | 新增 `S3VectorsClient`，向量写入/检索/删除路径改造 |
| 深度搜索（source_excerpt + narrative excerpts） | ✅ 已完成 | 边提取时标注原文片段，多路 RRF 融合，叙事片段向量检索 |
| Bedrock LLM/Embedding 集成 | ✅ 已完成 | `BedrockLLMClient`、`BedrockNovaEmbedder`，Kimi K2.5 via Mantle |
| LLM 调用超时与重试 | ✅ 已完成 | `asyncio.timeout()` 包装，按 prompt 类型配置超时 |
| S3 调用日志系统 | ✅ 已完成 | `S3InvocationLogger`，Athena 查询分析 |
| Ingest 可观测性 | ✅ 已完成 | 6 阶段进度日志，LLM trace 环境变量控制 |
| Episode Narratives 持久化 | ✅ 已完成 | `EpisodicNode.narrative_excerpts`，S3 Vectors 存储 |
| 深度搜索泛化（DescribesEdge + deep_search 参数） | ✅ 已完成 | DescribesEdge（Episode→Entity 描述边）、`_apply_deep_search()` 正交参数、Node source_similarity 搜索（见章节12） |
| 一次性批量导入（skip dedup） | ⬜ 待实现 | 跳过去重逻辑，首次灌入语料场景 |
| 多模态数据支持 | ⬜ 设计完成 | ContentBlock 有序内容块、文档解析框架、跨模态检索（见章节13） |
| 音视频数据处理 | ✅ 已验证 | ffmpeg 音频提取 + AWS Transcribe 转录 → 纯文本导入（见章节14） |

### 10.2 双写策略

POC 阶段采用双写策略（Neo4j + S3 Vectors 都存 embedding），不修改 `has_aoss` 标志。这样即使 S3 Vectors 不可用，系统仍可回退到 Neo4j 暴力计算。

### 10.3 差异对比（TODO）

改造稳定后，基于社区版本生成完整变更清单：

```bash
# 对比本地改造与社区版本的差异
git diff upstream/main -- graphiti_core/ > local_patches.diff

# 按文件统计改动行数
git diff upstream/main --stat -- graphiti_core/
```

输出将包含：新增文件、修改文件、每个文件的改动行数，作为最终的改造文件清单。

## 11. 注意事项

- S3 Vectors 写入吞吐为 1000 vectors/s（GA 版本），批量导入时需注意限流
- S3 Vectors 每个向量的 filterable metadata 有大小限制，`fact` 字段需截断
- BM25 全文检索仍走 Neo4j，不受影响
- MMR reranker 需要加载 embedding，改造后需从 S3 Vectors 获取而非 Neo4j
- 后续可平滑迁移到 OpenSearch 以获得更高性能和全文检索统一


## 12. 深度搜索泛化改造：原文溯源从 Edge 扩展到 Node

### 12.1 背景与动机

当前深度搜索（`deep_search`）仅支持 Edge 级别的原文溯源：

- `EdgeSearchMethod.source_similarity` 搜索边的 `source_excerpt` 向量
- `EDGE_DEEP_SEARCH_RRF` 作为独立配方硬编码在 `search()` 中
- `uncovered_excerpts` 作为 Episode 级别的"兜底"补充

这存在三个问题：

1. **Node 无溯源能力**：`EntityNode` 只有 `name` + `summary`（LLM 生成的摘要），没有原文关联。查询"张三是谁"时，无法回溯到描述张三的原文片段
2. **deep_search 与 SearchConfig 正交性差**：`deep_search` 被实现为一个独立配方（`EDGE_DEEP_SEARCH_RRF`），而非可叠加到任意配方上的增强开关。用户无法在 Cross-Encoder 精排 + 深度溯源的组合下使用
3. **uncovered_excerpts 归属不清**：当前所有未被 edge 覆盖的原文片段都堆在 Episode 级别，但其中很多片段实际上在描述某个特定实体（如"角有徒弟五百余人，云游四方，皆能书符念咒"描述的是张角），应该关联到对应的 EntityNode

> **⚠️ 术语重构**：章节 12.1~12.8 中的 `uncovered_excerpts` / `UncoveredExcerpt` 等术语为改造过程中的历史用词，已在 12.9.5 中统一更名为 `narrative_excerpts` / `NarrativeExcerpt`。代码、索引、日志、Neo4j 属性均已完成重命名。

### 12.2 设计目标

1. `deep_search` 作为 `search_()` 的正交参数，可叠加到任意 `SearchConfig` 上
2. EntityNode 获得原文溯源能力，通过 `source_similarity` 搜索关联的原文片段
3. uncovered_excerpts 按归属重新分类：关联到 Node 的归 Node，两头都不沾的留在 Episode
4. `search()` 简单入口透传 `deep_search` 参数
5. 废弃 `EDGE_DEEP_SEARCH_RRF` 独立配方

### 12.3 原文片段归属模型

改造后，Episode 原文被拆分为三类，通过三种方式关联：

```
Episode.content（原文全文）
  │
  ├─ EntityEdge (RELATES_TO)（已有）— 图边
  │   被识别为两个命名实体之间的关系
  │   边上存 fact（LLM 概括）+ source_excerpt（原文片段）
  │   EpisodicNode.entity_edges 记录 UUID
  │   向量存储：edge-fact-embeddings + edge-excerpt-embeddings 索引
  │
  ├─ DescribesEdge (DESCRIBES)（新增）— 图边
  │   描述某个特定实体的属性、能力、背景等，但无法表示为实体间关系
  │   从 EpisodicNode 指向 EntityNode，边上存 fact（LLM 概括）+ excerpt（原文片段）
  │   EpisodicNode.describes_edges 记录 UUID
  │   向量存储：describes-fact-embeddings + describes-excerpt-embeddings 索引（新增）
  │
  └─ 纯叙事文本（保留）— 节点属性，非图边
      与任何已识别实体都无关的纯叙事/环境描写
      保留在 Episode 级别
      EpisodicNode.uncovered_excerpts 记录文本（JSON）
      向量存储：episode-narrative-embeddings 索引
```

**图结构对比**：

```
改造前：
  (Episodic)--[MENTIONS]-->(Entity)        # 轻量关联，无内容
  (Entity)--[RELATES_TO]-->(Entity)        # 实体间关系，有 fact + source_excerpt

改造后（新增）：
  (Episodic)--[DESCRIBES {fact, excerpt}]-->(Entity)  # 单一实体描述，有 LLM 概括 + 原文片段
```

**设计决策**：采用 DescribesEdge 而非在 EntityNode 上新增 source_excerpts 字段，原因：
1. 关联关系用图的边表达，符合图数据库范式
2. EntityNode 不需要加任何字段，保持模型简洁
3. 与 EpisodicEdge（MENTIONS）平行，继承 Edge 基类即可，不需要改基类
4. EpisodicNode 的关联信息完整：entity_edges + describes_edges + uncovered_excerpts = content 全覆盖

### 12.4 分步实施计划

改造分为 6 个 Phase，每个 Phase 独立可测试，前一个 Phase 完成并验证后再进入下一个。

---

#### Phase 1：抽取层改造 — uncovered_excerpts 结构化归属

**目标**：让 LLM 在抽取边的同时，判断每条 uncovered excerpt 是否关联到某个已识别实体。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/prompts/extract_edges.py` | `UncoveredExcerpt` 模型新增 `related_entity` 字段；prompt 增加归属判断规则 |

**数据模型变更**：

```python
# 现有
class ExtractedEdges(BaseModel):
    edges: list[Edge]
    uncovered_excerpts: list[str]

# 改造后
class UncoveredExcerpt(BaseModel):
    excerpt: str = Field(
        description='The exact sentence(s) copied verbatim from the CURRENT MESSAGE.'
    )
    related_entity: str | None = Field(
        default=None,
        description='If this excerpt describes a specific entity from the ENTITIES list, '
                    'put the entity name here. None if the excerpt is pure narrative/setting.'
    )
    fact: str | None = Field(
        default=None,
        description='If related_entity is set, a concise summary of what this excerpt '
                    'describes about the entity. None if the excerpt is pure narrative/setting.'
    )

class ExtractedEdges(BaseModel):
    edges: list[Edge]
    uncovered_excerpts: list[UncoveredExcerpt]  # list[str] → list[UncoveredExcerpt]
```

**Prompt 新增规则**（`extract_edges.py` 的 UNCOVERED EXCERPTS RULES 部分）：

```
For each uncovered excerpt, determine if it primarily describes a specific entity
from the ENTITIES list (e.g., abilities, characteristics, background of that entity).
If so, set `related_entity` to that entity's name and provide a concise `fact` 
summarizing what the excerpt describes about the entity. If the excerpt is pure 
narrative, setting description, or cannot be attributed to a single entity, set 
both `related_entity` and `fact` to null.
```

**下游适配**：

`edge_operations.py` 中 `extract_edges()` 返回类型从 `tuple[list[EntityEdge], list[str]]` 改为 `tuple[list[EntityEdge], list[UncoveredExcerpt]]`。所有消费 uncovered_excerpts 的地方需要适配新结构。

**测试方法**：

| 测试类型 | 方法 | 验证点 |
|---------|------|--------|
| 单元测试 | mock LLM 返回包含 `related_entity` 的结构化 uncovered excerpts | `ExtractedEdges` 模型解析正确；`related_entity` 为 None 和非 None 的情况都能处理 |
| 单元测试 | 测试 `extract_edges()` 返回新类型 | 下游代码能正确消费 `UncoveredExcerpt` 对象 |
| 端到端验证 | 用三国演义第一回段落跑 `2_ingest.py` | 观察 LLM 输出：如"角有徒弟五百余人"应归属到"张角"并附带 fact 概括；纯叙事如"时人有桥玄者"应为 None |
| 回归测试 | 运行现有 316 个测试 | 确保不破坏现有功能 |

**风险**：LLM 归属判断的准确性。可能出现：错误归属（把无关片段归到某实体）、遗漏归属（该归的没归）。需要通过端到端测试观察质量，必要时调整 prompt。

---

#### Phase 2：新增 DescribesEdge — Episode 到 Entity 的描述边

**目标**：新增 DescribesEdge 类型，表示"Episode 描述了 Entity"，边上携带 LLM 概括和原文片段。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/edges.py` | 新增 `DescribesEdge` 类，继承 Edge |
| `graphiti_core/models/edges/edge_db_queries.py` | 新增 DESCRIBES 边的 save/return Cypher 查询 |
| `graphiti_core/nodes.py` | `EpisodicNode` 新增 `describes_edges: list[str]` 字段 |
| `graphiti_core/models/nodes/node_db_queries.py` | save/save_bulk/return 查询包含 describes_edges 字段 |

**数据模型**：

```python
class DescribesEdge(Edge):
    """Episode → Entity 的描述边，携带 LLM 概括和原文片段。
    
    source_node_uuid = episode.uuid (EpisodicNode)
    target_node_uuid = entity.uuid (EntityNode)
    
    与 EntityEdge 对齐：fact（LLM 概括）用于标准搜索和去重，
    excerpt（原文片段）用于深度搜索和原文溯源。
    """
    fact: str = Field(
        description='LLM-generated summary of what this excerpt describes about the entity'
    )
    fact_embedding: list[float] | None = Field(
        default=None, description='embedding of the fact'
    )
    excerpt: str = Field(
        description='Original text excerpt from the episode that describes the target entity'
    )
    excerpt_embedding: list[float] | None = Field(
        default=None, description='embedding of the excerpt'
    )
```

```python
class EpisodicNode(Node):
    # ... 现有字段
    entity_edges: list[str]          # EntityEdge UUID（用 | 拼接，UUID 安全）
    describes_edges: list[str] = Field(  # 新增
        default_factory=list,
        description='list of describes edges referenced in this episode'
    )
    uncovered_excerpts: list[str]    # 纯叙事（JSON 存储，Phase 0 已修复）
```

**Neo4j Cypher**：

```cypher
-- save
MATCH (ep:Episodic {uuid: $episode_uuid}), (en:Entity {uuid: $entity_uuid})
MERGE (ep)-[e:DESCRIBES {uuid: $uuid}]->(en)
SET e.group_id = $group_id,
    e.fact = $fact,
    e.excerpt = $excerpt,
    e.created_at = $created_at
RETURN e.uuid AS uuid

-- return
e.uuid AS uuid,
e.group_id AS group_id,
startNode(e).uuid AS source_node_uuid,
endNode(e).uuid AS target_node_uuid,
e.fact AS fact,
e.excerpt AS excerpt,
e.created_at AS created_at
```

**设计决策**：
- `fact_embedding` 和 `excerpt_embedding` 不存 Neo4j（与 EntityEdge 一致），只存 S3 Vectors
- `fact` 存 Neo4j，可用于 BM25 全文检索
- `describes_edges` 在 EpisodicNode 中用 `|` 拼接存储（存的是 UUID，不含 `|`，安全）
- DescribesEdge 的删除逻辑需要加入 `Node.delete()` 和 `Edge.delete_by_uuids()` 中的 Cypher（在 MENTIONS|RELATES_TO|HAS_MEMBER 后加 DESCRIBES）

**测试方法**：

| 测试类型 | 方法 | 验证点 |
|---------|------|--------|
| 单元测试 | 构造 DescribesEdge，序列化/反序列化 | 字段完整；excerpt 默认空字符串不报错 |
| 单元测试 | EpisodicNode 带 describes_edges，序列化/反序列化 | 新字段默认空列表；非空列表正确处理 |
| 集成测试 | DescribesEdge.save() → Neo4j → get_by_uuid() 读回 | excerpt 写入后完整读回 |
| 集成测试 | EpisodicNode.save() 包含 describes_edges | UUID 列表正确持久化 |
| 集成测试 | Node.delete() 清理 DESCRIBES 边 | 删除 Entity 节点时关联的 DESCRIBES 边也被清理 |
| 回归测试 | 运行全部测试 | 新增字段有默认值，不破坏现有代码 |

---

#### Phase 3：写入流程改造 — uncovered excerpts 按归属分流 + DescribesEdge 创建

**目标**：在 `add_episode()` 写入阶段，将 uncovered excerpts 按归属分流：有实体归属的创建 DescribesEdge，无归属的留在 Episode 级别。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/graphiti.py` | `add_episode()` Phase 6 分流逻辑；新增 `_create_describes_edges()` 方法 |
| `graphiti_core/vector_store/s3_vectors_client.py` | 新增 `describes-fact-embeddings` + `describes-excerpt-embeddings` 索引配置；新增 `upsert_describes_vector()` / `query_describes_vectors()` / `delete_describes_vectors()` |

**S3 Vectors 新增索引**（2 个）：

```
index: describes-fact-embeddings
dimension: 1024
distance_metric: cosine

每条向量:
  key: "{describes_edge_uuid}"
  data: float[1024]  # fact embedding（LLM 概括）
  metadata (filterable):
    uuid: string
    group_id: string
    target_node_uuid: string
    fact: string (截断到 200 字符)
    created_at: number (unix timestamp)
```

```
index: describes-excerpt-embeddings
dimension: 1024
distance_metric: cosine

每条向量:
  key: "{describes_edge_uuid}"
  data: float[1024]  # excerpt embedding（原文片段）
  metadata (filterable):
    uuid: string
    group_id: string
    target_node_uuid: string
    excerpt: string (截断到 200 字符)
    created_at: number (unix timestamp)
```

**写入流程变更**（`add_episode()` Phase 6）：

```python
# 现有逻辑
if uncovered_excerpts:
    await self._sync_uncovered_excerpts_to_s3_vectors(uncovered_excerpts, episode, group_id, now)
    episode.uncovered_excerpts = uncovered_excerpts

# 改造后
node_excerpts: dict[str, list[UncoveredExcerpt]] = {}   # entity_name → [UncoveredExcerpt, ...]
pure_uncovered: list[str] = []

for ue in uncovered_excerpts:  # UncoveredExcerpt 对象
    if ue.related_entity:
        node_excerpts.setdefault(ue.related_entity, []).append(ue)
    else:
        pure_uncovered.append(ue.excerpt)

# 1. 有归属的 → 创建 DescribesEdge + 写入 S3 Vectors
describes_edges = await self._create_describes_edges(
    node_excerpts, resolved_nodes, episode, group_id, now
)
episode.describes_edges = [e.uuid for e in describes_edges]

# 2. 真正的 uncovered → Episode 级别（逻辑不变，数据量减少）
if pure_uncovered:
    await self._sync_uncovered_excerpts_to_s3_vectors(pure_uncovered, episode, group_id, now)
episode.uncovered_excerpts = pure_uncovered
await episode.save(self.driver)
```

**`_create_describes_edges()` 实现**：

```python
async def _create_describes_edges(
    self,
    node_excerpts: dict[str, list[UncoveredExcerpt]],
    resolved_nodes: list[EntityNode],
    episode: EpisodicNode,
    group_id: str,
    now: datetime,
) -> list[DescribesEdge]:
    name_to_node = {n.name: n for n in resolved_nodes}
    edges = []
    
    for entity_name, ue_list in node_excerpts.items():
        node = name_to_node.get(entity_name)
        if node is None:
            logger.warning('DescribesEdge: entity "%s" not found in resolved_nodes, '
                          'demoting %d excerpts to uncovered', entity_name, len(ue_list))
            continue  # 降级：调用方需要把这些 excerpt 加回 pure_uncovered
        
        for ue in ue_list:
            edge = DescribesEdge(
                source_node_uuid=episode.uuid,
                target_node_uuid=node.uuid,
                group_id=group_id,
                fact=ue.fact or ue.excerpt,  # fallback: LLM 未输出 fact 时用原文
                excerpt=ue.excerpt,
                created_at=now,
            )
            await edge.save(self.driver)
            edges.append(edge)
    
    # Batch embed fact + excerpt, sync to S3 Vectors
    if edges:
        valid_edges = [e for e in edges if e.fact or e.excerpt]
        if valid_edges:
            # fact embeddings
            fact_embeddings = await self.clients.embedder.create_batch(
                [e.fact for e in valid_edges]
            )
            # excerpt embeddings
            excerpt_embeddings = await self.clients.embedder.create_batch(
                [e.excerpt for e in valid_edges]
            )
            for edge, fact_emb, excerpt_emb in zip(
                valid_edges, fact_embeddings, excerpt_embeddings, strict=True
            ):
                edge.fact_embedding = fact_emb
                edge.excerpt_embedding = excerpt_emb
                self.s3_vectors.upsert_describes_fact_vector(
                    uuid=edge.uuid,
                    embedding=fact_emb,
                    group_id=edge.group_id,
                    target_node_uuid=edge.target_node_uuid,
                    fact=edge.fact,
                    created_at_ts=edge.created_at.timestamp(),
                )
                self.s3_vectors.upsert_describes_excerpt_vector(
                    uuid=edge.uuid,
                    embedding=excerpt_emb,
                    group_id=edge.group_id,
                    target_node_uuid=edge.target_node_uuid,
                    excerpt=edge.excerpt,
                    created_at_ts=edge.created_at.timestamp(),
                )
    
    return edges
```

**实体名匹配逻辑**：

精确匹配 `node.name == related_entity`。未匹配到时记录 warning 日志，该 excerpt 降级为 pure_uncovered。

**测试方法**：

| 测试类型 | 方法 | 验证点 |
|---------|------|--------|
| 单元测试 | mock S3 Vectors，验证 `upsert_describes_fact_vector()` 和 `upsert_describes_excerpt_vector()` 调用参数 | key 为 edge UUID；metadata 包含 target_node_uuid；fact/excerpt 截断到 200 字符 |
| 单元测试 | 构造含 `related_entity` + `fact` 的 UncoveredExcerpt 列表，验证分流逻辑 | 有归属的创建 DescribesEdge（含 fact）；无归属的进 pure_uncovered；实体名不匹配的降级 |
| 单元测试 | S3VectorsConfig 验证 | index 数量从 5 变为 7；新增 `describes_fact_index_name` + `describes_excerpt_index_name` |
| 集成测试 | `add_episode()` 端到端，mock LLM 返回含归属的 uncovered excerpts | DescribesEdge 写入 Neo4j；S3 Vectors describes 索引有写入；Episode.describes_edges 记录 UUID；Episode.uncovered_excerpts 只含无归属片段 |
| 集成测试 | 通过 EntityNode 反查 DESCRIBES 边 | `MATCH (ep:Episodic)-[d:DESCRIBES]->(en:Entity {uuid: $uuid})` 能查到关联的描述边 |
| 端到端验证 | 三国演义段落导入 | 观察"角有徒弟五百余人"是否创建了指向张角的 DescribesEdge |
| 回归测试 | 运行全部测试 | 现有 uncovered excerpt 相关测试需适配新的 UncoveredExcerpt 类型 |

---

#### Phase 4：搜索层改造 — Node source_similarity 搜索实现（via DescribesEdge）

**目标**：Node 搜索支持 `source_similarity` 方法，通过 S3 Vectors 搜索 DescribesEdge 的 excerpt 向量，返回关联的 EntityNode。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/search/search_config.py` | `NodeSearchMethod` 新增 `source_similarity` 枚举值 |
| `graphiti_core/search/search_utils.py` | 新增 `s3_vectors_node_source_similarity_search()` 函数 |
| `graphiti_core/search/search.py` | `node_search()` 中处理 `NodeSearchMethod.source_similarity` 路由 |

**搜索实现**：

```python
# search_config.py
class NodeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'
    source_similarity = 'source_similarity'  # 新增

# search_utils.py
async def s3_vectors_node_source_similarity_search(
    s3_vectors: 'S3VectorsClient',
    driver: GraphDriver,
    search_vector: list[float],
    group_ids: list[str] | None,
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityNode]:
    """Search DescribesEdge fact/excerpt via S3 Vectors, return the target EntityNodes.
    
    搜索 describes-fact-embeddings（标准搜索）或 describes-excerpt-embeddings（深度搜索），
    通过 metadata 中的 target_node_uuid 关联回 EntityNode。
    """
    results = s3_vectors.query_describes_vectors(
        query_vector=search_vector,
        group_ids=group_ids,
        min_score=min_score,
    )
    if not results:
        return []
    # 从 metadata 中提取 target_node_uuid（去重，一个 node 可能有多条 describes）
    node_uuids = list(dict.fromkeys(
        r.metadata.get('target_node_uuid') for r in results
    ))
    nodes = await EntityNode.get_by_uuids(driver, node_uuids)
    # 保持 S3 Vectors 返回的排序
    uuid_to_node = {n.uuid: n for n in nodes}
    return [uuid_to_node[uid] for uid in node_uuids if uid in uuid_to_node]
```

**注意**：搜索的是 DescribesEdge 的 excerpt 向量，但返回的是 EntityNode（通过 metadata 中的 `target_node_uuid` 关联）。一个 Node 可能有多条 DescribesEdge 命中，按 target_node_uuid 去重后取最高分。

**测试方法**：

| 测试类型 | 方法 | 验证点 |
|---------|------|--------|
| 单元测试 | mock S3 Vectors 返回多条同 node_uuid 的结果 | 去重后只返回一个 EntityNode；排序按首次出现的最高分 |
| 单元测试 | mock S3 Vectors 返回空结果 | 返回空列表，不报错 |
| 单元测试 | `node_search()` 路由测试 | `NodeSearchMethod.source_similarity` 正确路由到 `s3_vectors_node_source_similarity_search`；无 s3_vectors 时跳过 |
| 单元测试 | `NodeSearchMethod` 枚举验证 | 包含 4 个值：cosine_similarity, bm25, bfs, source_similarity |
| 端到端验证 | 搜索"念咒者何人"，观察 Node 结果 | 张角节点应通过 source_similarity 被召回（因为"角有徒弟五百余人，皆能书符念咒"关联到张角） |
| 回归测试 | 运行全部测试 | 新增枚举值不影响现有配方（现有配方不包含 source_similarity） |

---

#### Phase 5：search_() 泛化 deep_search 参数

**目标**：`deep_search` 作为 `search_()` 的正交参数，自动给用户传入的任意 SearchConfig 追加 source_similarity 方法。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/graphiti.py` | `search_()` 新增 `deep_search: bool = False` 参数；新增 `_apply_deep_search()` 方法 |
| `graphiti_core/search/search_config_recipes.py` | 废弃 `EDGE_DEEP_SEARCH_RRF`（标记 deprecated，暂不删除） |

**核心实现**：

```python
async def search_(
    self,
    query: str,
    config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    group_ids: list[str] | None = None,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    search_filter: SearchFilters | None = None,
    driver: GraphDriver | None = None,
    deep_search: bool = False,  # 新增
) -> SearchResults:
    if deep_search:
        config = self._apply_deep_search(config)
    return await search(...)

@staticmethod
def _apply_deep_search(config: SearchConfig) -> SearchConfig:
    """给 config 中已启用的搜索对象追加 source_similarity 方法。
    
    只对已配置的搜索对象生效（不会凭空启用未配置的对象）。
    例如：用户只配了 edge_config，deep_search 只给 edge 加 source_similarity，
    不会自动启用 node_config。
    """
    config = config.model_copy(deep=True)
    
    if config.edge_config is not None:
        if EdgeSearchMethod.source_similarity not in config.edge_config.search_methods:
            config.edge_config.search_methods.append(EdgeSearchMethod.source_similarity)
    
    if config.node_config is not None:
        if NodeSearchMethod.source_similarity not in config.node_config.search_methods:
            config.node_config.search_methods.append(NodeSearchMethod.source_similarity)
    
    # Community 和 Episode 暂不支持 source_similarity，后续按需扩展
    
    return config
```

**设计决策**：

- `_apply_deep_search` 只对已启用的搜索对象追加，不会凭空启用。如果用户只配了 `edge_config`，deep_search 不会自动加 `node_config`
- Community 和 Episode 暂不加 source_similarity：Community 是 Node 的聚合摘要，可通过成员 Node 间接溯源；Episode 本身就是原文
- `EDGE_DEEP_SEARCH_RRF` 标记 deprecated 但暂不删除，避免破坏已有代码引用

**测试方法**：

| 测试类型 | 方法 | 验证点 |
|---------|------|--------|
| 单元测试 | `_apply_deep_search()` 对各种 SearchConfig 输入 | 只有 edge_config → 只加 edge source_similarity；只有 node_config → 只加 node source_similarity；combined → 两个都加；已有 source_similarity 不重复添加 |
| 单元测试 | `_apply_deep_search()` 不修改原始 config | 验证 deep copy，原始 config 对象不变 |
| 单元测试 | `search_()` 传 `deep_search=True` | 验证底层 `search()` 收到的 config 包含 source_similarity |
| 单元测试 | `search_()` 传 `deep_search=False`（默认） | config 不变，行为与改造前一致 |
| 端到端验证 | `search_(query, config=COMBINED_HYBRID_SEARCH_RRF, deep_search=True)` | Edge 和 Node 都通过 source_similarity 召回额外结果 |
| 端到端验证 | `search_(query, config=NODE_HYBRID_SEARCH_CROSS_ENCODER, deep_search=True)` | 只有 Node 加了 source_similarity（因为没配 edge_config） |
| 回归测试 | 运行全部测试 | `deep_search` 默认 False，不影响现有行为 |

---

#### Phase 6：search() 简单入口适配

**目标**：`search()` 简单入口透传 `deep_search` 参数，废弃内部硬编码的 `EDGE_DEEP_SEARCH_RRF` 分支。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/graphiti.py` | `search()` 内部改为调用 `search_()` + `deep_search` 参数，不再维护独立的 config 选择逻辑 |

**改造前**：

```python
async def search(self, query, center_node_uuid=None, ..., deep_search=False):
    if deep_search:
        search_config = EDGE_DEEP_SEARCH_RRF.model_copy(update={'limit': num_results * 2})
    elif center_node_uuid is not None:
        search_config = EDGE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(update={'limit': num_results})
    else:
        search_config = EDGE_HYBRID_SEARCH_RRF.model_copy(update={'limit': num_results})
    # ... 调用底层 search()
```

**改造后**：

```python
async def search(self, query, center_node_uuid=None, ..., deep_search=False):
    if center_node_uuid is not None:
        search_config = EDGE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(update={'limit': num_results})
    else:
        search_config = EDGE_HYBRID_SEARCH_RRF.model_copy(update={'limit': num_results})
    
    # deep_search 不再影响 config 选择，而是透传给 search_()
    # deep_search 时 limit 翻倍的逻辑也移到这里
    if deep_search:
        search_config = search_config.model_copy(update={'limit': num_results * 2})
    
    search_results = await self.search_(
        query=query,
        config=search_config,
        group_ids=group_ids,
        center_node_uuid=center_node_uuid,
        search_filter=search_filter,
        driver=driver,
        deep_search=deep_search,  # 透传
    )
    return search_results.edges
```

**关键变化**：

- `deep_search` 与 `center_node_uuid` 不再互斥，可以同时使用（以某个实体为中心 + 深度溯源）
- `search()` 返回值仍为 `list[EntityEdge]`（从 SearchResults 中取 edges），保持向后兼容
- uncovered_excerpts 信息在 `search()` 中丢失（只返回 edges），需要完整结果的用户应使用 `search_()`

**测试方法**：

| 测试类型 | 方法 | 验证点 |
|---------|------|--------|
| 单元测试 | `search(deep_search=True, center_node_uuid='xxx')` | 两个参数同时生效：config 用 NODE_DISTANCE 重排 + source_similarity 追加 |
| 单元测试 | `search(deep_search=True)` | limit 翻倍；底层 search_() 收到 deep_search=True |
| 单元测试 | `search(deep_search=False)`（默认） | 行为与改造前完全一致 |
| 端到端验证 | 三国演义搜索对比 | 标准搜索 vs 深度搜索结果对比，深度搜索应召回更多相关结果 |
| 回归测试 | 运行全部测试 | 默认行为不变 |

### 12.5 改动文件汇总

| Phase | 文件 | 改动类型 |
|-------|------|---------|
| 0 | `graphiti_core/models/nodes/node_db_queries.py` | 修改（JSON 存储） |
| 0 | `graphiti_core/nodes.py` | 修改（JSON 序列化/反序列化） |
| 1 | `graphiti_core/prompts/extract_edges.py` | 修改 |
| 1 | `graphiti_core/utils/maintenance/edge_operations.py` | 修改 |
| 2 | `graphiti_core/edges.py` | 新增 DescribesEdge |
| 2 | `graphiti_core/models/edges/edge_db_queries.py` | 新增 DESCRIBES 查询 |
| 2 | `graphiti_core/nodes.py` | 修改（EpisodicNode 加 describes_edges） |
| 2 | `graphiti_core/models/nodes/node_db_queries.py` | 修改 |
| 3 | `graphiti_core/graphiti.py` | 修改 |
| 3 | `graphiti_core/vector_store/s3_vectors_client.py` | 修改 |
| 4 | `graphiti_core/search/search_config.py` | 修改 |
| 4 | `graphiti_core/search/search_utils.py` | 修改 |
| 4 | `graphiti_core/search/search.py` | 修改 |
| 5 | `graphiti_core/graphiti.py` | 修改 |
| 5 | `graphiti_core/search/search_config_recipes.py` | 修改 |
| 6 | `graphiti_core/graphiti.py` | 修改 |

### 12.6 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Phase 0 旧数据兼容 | 已有 `|` 分隔的 uncovered_excerpts 需要兼容读取 | `_parse_excerpts()` fallback 到 `split('|')` |
| Phase 1 LLM 归属判断不准确 | uncovered excerpts 错误归属到 Node 或遗漏归属 | 端到端测试观察质量；prompt 迭代调优；降级策略（不确定的归为 None） |
| Phase 1 输出格式变化导致解析失败 | `ExtractedEdges` 从 `list[str]` 改为 `list[UncoveredExcerpt]`，LLM 可能不稳定输出结构化对象 | Pydantic 解析失败时 fallback 到 `list[str]`（每条 excerpt 的 related_entity 设为 None） |
| Phase 2 DESCRIBES 边的删除清理 | 删除 Entity/Episode 时需要同步清理 DESCRIBES 边 | 在 Node.delete() 和相关 Cypher 中加入 DESCRIBES 类型 |
| Phase 3 实体名匹配失败 | LLM 返回的 related_entity 与 resolved_nodes 中的 name 不完全一致 | 精确匹配 + warning 日志 + 降级为 pure_uncovered；后续可加模糊匹配 |
| Phase 4 Node 去重问题 | 一个 Node 多条 DescribesEdge 命中，搜索结果中出现重复 Node | 按 target_node_uuid 去重，取最高分 |
| Phase 5-6 向后兼容 | 现有调用方代码可能依赖 `EDGE_DEEP_SEARCH_RRF` | 标记 deprecated 但不删除；新参数都有默认值 |

### 12.7 不在本次改造范围内

| 项目 | 原因 |
|------|------|
| CommunityNode source_similarity | Community 是 Node 的聚合摘要，可通过成员 Node 间接溯源，优先级低 |
| EpisodicNode source_similarity | Episode 本身就是原文，不需要溯源 |
| SagaNode source_similarity | SagaNode 字段极少，暂无溯源需求 |
| uncovered excerpts 补全机制（9.2） | 独立功能，不依赖本次改造，按原计划后续迭代 |
| uncovered excerpts 独立 limit（9.3） | 可在本次改造中顺带实现，但非核心目标，视工期决定 |

> **⚠️ 术语重构**：上表中的 `uncovered excerpts` 相关条目已在 12.9.5 重构中统一更名为 `narrative excerpts`，对应 TODO 章节已合并为 9.2。

### 12.8 前置修复：uncovered_excerpts Neo4j 存储格式从 `|` 分隔改为 JSON

> **⚠️ 术语重构**：本节中的 `uncovered_excerpts` 已在 12.9.5 中统一更名为 `narrative_excerpts`，Neo4j 属性名已完成迁移。

**问题**：当前 `EpisodicNode.uncovered_excerpts` 在 Neo4j 中用 `|` 拼接存储，读取时用 `split('|')` 拆分。但 uncovered_excerpts 存的是原文片段，内容中完全可能包含 `|` 字符，导致读回时被错误拆分。`entity_edges` 存的是 UUID 不受影响。

**现有代码**：

```cypher
-- 写入（join）
uncovered_excerpts: join([x IN coalesce($uncovered_excerpts, []) | toString(x) ], '|')

-- 读取（split）
split(e.uncovered_excerpts, "|") AS uncovered_excerpts
```

**修复方案**：改为 JSON 字符串存储，Python 侧序列化/反序列化。Neo4j 中该字段无任何 Cypher 层面的查询或过滤操作，纯存取，JSON 完全适用。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/models/nodes/node_db_queries.py` | 写入时去掉 `join(..., '|')`，改为 Python 侧 `json.dumps()` 后传入字符串；读取时去掉 `split(..., '|')`，直接返回原始字符串 |
| `graphiti_core/nodes.py` | `save()` 传参前 `json.dumps(self.uncovered_excerpts)`；`get_episodic_node_from_record()` 读取时 `json.loads()` |

**写入改造**：

```python
# EpisodicNode.save() 中
episode_args = {
    ...
    'uncovered_excerpts': json.dumps(self.uncovered_excerpts, ensure_ascii=False),
    ...
}
```

```cypher
-- Cypher 中直接存字符串，不再 join
uncovered_excerpts: $uncovered_excerpts
```

**读取改造**：

```cypher
-- Cypher 中直接返回字符串，不再 split
e.uncovered_excerpts AS uncovered_excerpts
```

```python
# get_episodic_node_from_record() 中
raw = record.get('uncovered_excerpts', '[]')
uncovered_excerpts = json.loads(raw) if isinstance(raw, str) else (raw or [])
```

**兼容性**：已有数据中 uncovered_excerpts 是 `|` 分隔的字符串。读取时需要兼容：如果 `json.loads()` 失败（不是合法 JSON），fallback 到 `split('|')`。新写入的数据统一用 JSON。

```python
def _parse_excerpts(raw: str | list | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else [raw]
    except (json.JSONDecodeError, TypeError):
        return raw.split('|') if raw else []  # 兼容旧数据
```

**同步应用到 DescribesEdge**：Phase 2 新增的 `DescribesEdge.excerpt` 是单个字符串字段，不涉及列表序列化问题。`EpisodicNode.describes_edges` 存的是 UUID 列表，用 `|` 拼接安全。

**测试方法**：

| 测试类型 | 方法 | 验证点 |
|---------|------|--------|
| 单元测试 | `_parse_excerpts()` 输入 JSON 字符串 | 正确解析为 list |
| 单元测试 | `_parse_excerpts()` 输入旧格式 `|` 分隔字符串 | fallback 正确拆分 |
| 单元测试 | `_parse_excerpts()` 输入 None / 空字符串 / 已是 list | 边界情况处理 |
| 单元测试 | 原文包含 `|` 字符的 excerpt | JSON 存储后读回内容完整，不被错误拆分 |
| 集成测试 | EpisodicNode save → get_by_uuid 往返 | uncovered_excerpts 写入 JSON 后完整读回 |
| 回归测试 | 运行全部测试 | 现有测试中 mock 的 uncovered_excerpts 数据能被正确解析 |

**实施时机**：在 Phase 1 之前作为前置修复完成，后续所有 Phase 基于 JSON 存储格式开发。

### 12.9 补充记录

#### 12.9.1 S3 Vectors 索引清单更新

改造完成后索引从 5 个变为 8 个：

| # | 索引名 | 来源 | 用途 |
|---|--------|------|------|
| 1 | `entity-name-embeddings` | 4.2 | EntityNode name 向量 |
| 2 | `edge-fact-embeddings` | 4.3 | EntityEdge fact 向量（LLM 概括） |
| 3 | `edge-source-embeddings` | 6.5 | EntityEdge source_excerpt 向量（原文片段） |
| 4 | `community-name-embeddings` | 4.4 | CommunityNode name 向量 |
| 5 | `episode-narrative-embeddings` | 6.8 | 纯叙事文本向量（无实体归属） |
| 6 | `describes-fact-embeddings` | 12.4 Phase 3 | DescribesEdge fact 向量（LLM 概括） |
| 7 | `describes-excerpt-embeddings` | 12.4 Phase 3 | DescribesEdge excerpt 向量（原文片段） |
| 8 | `episode-content-embeddings` | 12.9.4 | Episode 内容向量（重复导入检测） |

`S3VectorsConfig` 和 `ensure_bucket_and_indices()` 需同步更新。

#### 12.9.2 DescribesEdge 删除联动

DescribesEdge 涉及两个删除场景，都需要同步清理 Neo4j 边 + S3 Vectors 向量：

**场景 1：删除 EntityNode**

`Node.delete()` 中的 Cypher 已有 `OPTIONAL MATCH (n)-[r]-() DETACH DELETE`，会自动删除所有关联边（包括 DESCRIBES）。但 S3 Vectors 中的 describes 向量需要额外清理：

```python
# 删除 Entity 前，先查出关联的 DESCRIBES 边 UUID
MATCH (ep:Episodic)-[d:DESCRIBES]->(en:Entity {uuid: $uuid})
RETURN collect(d.uuid) AS describes_uuids
# 然后 s3_vectors.delete_describes_vectors(describes_uuids)
```

**场景 2：删除 EpisodicNode（remove_episode）**

当前 `remove_episode()` 清理 Episode 及其 MENTIONS 边。改造后需要同时清理：
- DESCRIBES 边（Neo4j 的 DETACH DELETE 会处理）
- S3 Vectors 中对应的 describes 向量（需要先查 UUID 再删）
- EpisodicNode.describes_edges 中记录的 UUID 可直接用于 S3 Vectors 删除，无需额外查询

#### 12.9.3 改造完成后需同步更新的文档章节

本文档中以下章节的内容基于改造前的状态，Phase 全部完成后需要回顾更新：

| 章节 | 需更新内容 | 状态 |
|------|-----------|------|
| 2.1 不需要修改的部分 | `EntityEdge` 新增 `source_excerpt` 的说明已过时，需补充 DescribesEdge | ✅ 已标注 |
| 2.2 需要修改的部分 | 补充 DescribesEdge 相关文件 | ✅ 已标注（2.2A 索引数量注释） |
| 4. S3 Vectors 数据结构设计 | 补充第 6、7 个索引 `describes-fact-embeddings` + `describes-excerpt-embeddings` | ✅ 已标注（4.4 后注释） |
| 5.2 检索流程 | 补充 deep_search 扩展的搜索路径 | ✅ 已标注 |
| 6.3 深度搜索流程图 | 补充 Node source_similarity 和正交参数说明 | ✅ 已标注 |
| 6.6 数据模型变更 | 补充 UncoveredExcerpt 结构化模型变更 | ✅ 已标注 |
| 6.7 改造文件清单 | 补充 Phase 2-4 涉及的文件 | ✅ 已标注 |
| 6.8 Uncovered Excerpts | 补充类型变更和实体归属提升机制 | ✅ 已标注 |
| 8.3 三国演义端到端测试 | 补充脚本拆分和搜索配置变更 | ✅ 已标注 |
| 8.6 单元测试 | 补充 DescribesEdge 相关测试用例数 | ✅ 已标注 |
| 8.7 集成测试 | 补充 DescribesEdge 写入/删除/搜索测试 | ✅ 已验证（49 passed, 1 skipped） |
| 8.9 测试汇总 | 更新总测试数 | ✅ 已标注（316→422） |
| 10.1 改造方向总览 | 新增"深度搜索泛化（DescribesEdge + deep_search 参数）"行 | ✅ 已新增 |

#### 12.9.4 Episode 级别重复导入检测

**背景**：端到端测试中反复遇到重复导入问题，之前通过清库清索引规避，但真实环境中重复导入不可避免。

**方案**：在 `add_episode` 入口处通过内容向量相似度检测重复：

1. 新增第 8 个 S3 Vectors 索引 `episode-content-embeddings`
2. `add_episode` 入口：embed content → query S3 Vectors → 若 score ≥ 阈值则 warning + skip
3. Phase 6：upsert episode content embedding（复用入口处的 embedding，不重复计算）
4. `remove_episode`：同步删除 episode content vector
5. 阈值通过 `EPISODE_DEDUP_MIN_SCORE` 环境变量配置，默认 0.95
6. 无 `s3_vectors` 时静默跳过检测

**涉及文件**：
- `graphiti_core/graphiti.py`：dedup guard、Phase 6 upsert、remove_episode delete
- `graphiti_core/vector_store/s3_vectors_client.py`：新索引配置、upsert/query/delete 方法
- `graphiti_core/nodes.py`：`EpisodicNode.get_by_name_and_group()` 辅助方法（保留）

#### 12.9.5 术语统一：uncovered_excerpts → narrative_excerpts

**背景**：改造过程中 S3 Vectors 索引名已从 `uncovered-excerpt-embeddings` 改为 `episode-narrative-embeddings`，日志和注释中的描述性文字也已统一为 "episode narratives"。但代码中的变量名、方法名、类名、Pydantic 模型字段名、Neo4j 属性名仍保留旧的 `uncovered_excerpt` 命名，与索引名和日志不一致。

**需要重构的符号**：

| 类别 | 当前命名 | 目标命名 | 位置 |
|------|---------|---------|------|
| Pydantic 模型类 | `UncoveredExcerpt` | `NarrativeExcerpt` | `extract_edges.py` |
| LLM JSON key | `uncovered_excerpts` | `narrative_excerpts` | `extract_edges.py` (ExtractedEdges) |
| LLM prompt 标题 | `# UNCOVERED EXCERPTS RULES` | `# NARRATIVE EXCERPTS RULES` | `extract_edges.py` (edge prompt) |
| Pydantic validator | `_coerce_uncovered_excerpts` | `_coerce_narrative_excerpts` | `extract_edges.py` |
| S3V 方法名 | `upsert_uncovered_excerpt_vector` | `upsert_narrative_vector` | `s3_vectors_client.py` |
| S3V 方法名 | `query_uncovered_excerpt_vectors` | `query_narrative_vectors` | `s3_vectors_client.py` |
| S3V 方法名 | `delete_uncovered_excerpt_vectors` | `delete_narrative_vectors` | `s3_vectors_client.py` |
| 搜索函数 | `s3_vectors_uncovered_excerpt_search` | `s3_vectors_narrative_search` | `search_utils.py` |
| 内部方法 | `_sync_uncovered_excerpts_to_s3_vectors` | `_sync_narratives_to_s3_vectors` | `graphiti.py` |
| SearchResults 字段 | `uncovered_excerpts` | `narrative_excerpts` | `search_config.py` |
| EpisodicNode 字段 | `uncovered_excerpts` | `narrative_excerpts` | `nodes.py` |
| Neo4j 属性名 | `uncovered_excerpts` | `narrative_excerpts` | `node_db_queries.py` (Cypher) |
| 局部变量 | `pure_uncovered` | `pure_narratives` | `graphiti.py` |
| 向量 key 前缀 | `{uuid}:uncovered:{hash}` | `{uuid}:narrative:{hash}` | `graphiti.py` |

**LLM prompt 改动详情**：

仅 `extract_edges.py` 的 `edge()` prompt 需要修改，其他 prompt（extract_nodes、dedupe_nodes、dedupe_edges、summarize_nodes、eval）均不涉及。

改动点：
1. `# UNCOVERED EXCERPTS RULES` → `# NARRATIVE EXCERPTS RULES`
2. prompt 描述从"未被覆盖的残余文本"调整为"无法用双实体关系表达的叙事性描述"
3. JSON schema key `uncovered_excerpts` → `narrative_excerpts`
4. backward compat：validator 同时接受旧 key `uncovered_excerpts`（LLM 可能偶尔返回旧格式）

**Neo4j 数据迁移**：

属性名从 `uncovered_excerpts` 改为 `narrative_excerpts` 后，需要对已有数据执行一次性迁移：

```cypher
MATCH (e:Episodic) WHERE e.uncovered_excerpts IS NOT NULL
SET e.narrative_excerpts = e.uncovered_excerpts
REMOVE e.uncovered_excerpts
```

**影响范围**：

| 层级 | 涉及文件数 | 改动点 | 风险 |
|------|-----------|--------|------|
| Python 符号（变量/方法/类） | ~8 源文件 + ~6 测试文件 | ~60 处 | 低（semanticRename） |
| Neo4j 属性 | node_db_queries.py, nodes.py | ~15 处 + 迁移 | 中（需迁移脚本） |
| LLM prompt / JSON schema | extract_edges.py | ~8 处 | 中（需 backward compat） |

**预期收益**：
1. 代码、索引、日志、文档术语完全统一，消除认知负担
2. prompt 描述更准确地反映当前逻辑（这些不是"残余"，而是"叙事性描述"），有助于 LLM 更好地理解任务意图
3. 为后续可能的 prompt 优化（如调整 narrative 提取粒度、增加归属准确率）奠定清晰的术语基础


## 13. 多模态数据支持

### 13.1 背景与动机

当前系统的数据流完全基于纯文本：`add_episode()` 接收 `episode_body: str`，LLM 从文本中提取实体和关系，embedding 模型对文本生成向量。这在处理对话记录、小说、技术文档等纯文本语料时运作良好。

但真实世界的文档往往包含多模态内容：Word 文档中嵌入的图片、PDF 中的图表、技术报告中的架构图等。这些非文本信息同样承载着重要的知识，当前系统无法处理。

**核心诉求**：

1. 支持图文混排文档（如 Word 文档内嵌图片）的导入，能够还原原始文件的完整内容和顺序
2. 利用 Nova MME 的多模态 embedding 能力，实现跨模态语义检索（如以图搜文、以文搜图）
3. 利用 Kimi K2.5 的视觉理解能力，从图片中提取实体和关系
4. 预处理模块化，后续可逐步扩展音频、视频等模态

**前期聚焦**：图片 + 文本（Word 文档内嵌图片场景），其他模态预留接口但不实现。

### 13.2 设计目标

1. **内容还原**：EpisodicNode 能完整还原原始文档的内容和顺序（文本块→图片→文本块的序列）
2. **跨模态检索**：文本和图片在同一 1024 维向量空间中，支持以图搜文、以文搜图
3. **多模态 LLM 提取**：图片直接发送给 Kimi K2.5 进行实体/关系提取，不依赖 OCR
4. **向后兼容**：纯文本 episode 的行为完全不变，`content: str` 仍然可用
5. **模块化预处理**：文档解析（Word→ContentBlock 序列）独立于核心图谱逻辑，可插拔扩展
6. **多模态资产外置**：图片等二进制数据存储在独立 S3 桶中，EpisodicNode 只存引用

### 13.3 数据模型设计

#### 13.3.0 开源项目参考

在设计 ContentBlock 模型前，调研了三个主流开源文档处理项目的数据模型：

| 项目 | 核心模型 | 内容类型 | 层次结构 | 特点 |
|------|---------|---------|---------|------|
| [Docling](https://github.com/docling-project/docling)（IBM） | `DoclingDocument` + typed items | Text, Table, Picture, Caption, List, Formula 等 | 树形（body/furniture 分层，section 嵌套） | Pydantic 模型，统一表示 PDF/DOCX/PPTX/HTML/图片/音频，支持 bounding box 和 provenance |
| [EdgeQuake](edgequake/docs/deep-dives/pdf-processing.md) | `Block` + `BlockType` enum | Text, Paragraph, SectionHeader, Title, Table, TableCell, Figure, Caption, List, ListItem, Code, Equation | 树形（Block.children 递归） | Rust 实现，Processor Chain 模式，空间分析（bounding box + XY-Cut 列检测），confidence 评分 |
| [Unstructured.io](https://unstructured.io/) | `Element` 类型体系 | NarrativeText, Image, Table, FigureCaption, ListItem, Title 等 | 扁平（Element 列表 + metadata） | Python ETL 平台，partition 函数按格式路由，metadata 携带坐标和来源信息 |

**共性设计模式**：

1. **类型化内容块**：都用枚举/类型区分不同内容（文本、图片、表格等），而非统一的 `str`
2. **顺序保持**：都维护文档中的原始阅读顺序
3. **元数据丰富**：bounding box、页码、来源信息、置信度等
4. **层次结构**：Docling 和 EdgeQuake 都支持嵌套（表格包含单元格，列表包含列表项）

**我们的取舍**：

- Graphiti 的 episode 粒度比上述项目更细（一个 episode 通常是一个段落或一段对话，不是整个文档），不需要完整的文档层次结构
- 但需要为后续扩展（表格、音频、视频、PPT）预留足够的类型空间和元数据字段
- 参考 Docling 的 Pydantic 模型设计和 EdgeQuake 的 Block 类型体系，设计一个适度丰富但不过度复杂的 ContentBlock 模型

#### 13.3.1 ContentBlock：有序内容块模型

引入 `ContentBlock` 作为多模态 episode 的基本内容单元。参考 Docling 的 typed items 和 EdgeQuake 的 Block 模型，设计一个面向未来扩展的类型体系：

```python
from enum import Enum
from pydantic import BaseModel, Field

class ContentBlockType(Enum):
    """内容块类型。
    
    设计原则：按内容的模态（modality）分类，而非按文档元素的语义角色分类。
    语义角色（标题、段落、脚注等）通过 semantic_role 字段表达，与模态正交。
    
    参考：
    - Docling: Text, Table, Picture, Caption, List, Formula
    - EdgeQuake: Text, Paragraph, SectionHeader, Table, Figure, Code, Equation
    - Unstructured: NarrativeText, Image, Table, FigureCaption, ListItem
    """
    # 当前实现
    text = 'text'           # 纯文本内容
    image = 'image'         # 图片（JPEG, PNG, SVG 等）
    table = 'table'         # 表格（结构化数据）
    
    # 预留扩展
    audio = 'audio'         # 音频文件
    video = 'video'         # 视频文件
    code = 'code'           # 代码块
    formula = 'formula'     # 数学公式
    chart = 'chart'         # 图表（柱状图、饼图等，区别于普通图片）
    slide = 'slide'         # PPT 幻灯片（整页作为一个块）
    
    @staticmethod
    def from_str(block_type: str) -> 'ContentBlockType':
        try:
            return ContentBlockType(block_type)
        except ValueError:
            return ContentBlockType.text  # 未知类型降级为 text


class SemanticRole(Enum):
    """内容块的语义角色（可选），与 block_type 正交。
    
    参考 Docling 的 body/furniture 分层和 EdgeQuake 的 BlockType 语义标签。
    """
    body = 'body'               # 正文内容（默认）
    title = 'title'             # 文档标题
    section_header = 'section_header'  # 章节标题
    caption = 'caption'         # 图表标题/说明
    footnote = 'footnote'       # 脚注
    header = 'header'           # 页眉
    footer = 'footer'           # 页脚
    list_item = 'list_item'     # 列表项
    abstract = 'abstract'       # 摘要
```

```python
class ContentBlock(BaseModel):
    """文档中一个有序内容块。
    
    设计目标：
    1. 能还原原始文档的内容和顺序
    2. 为后续扩展（表格、音频、视频、PPT）预留字段
    3. 每种模态的特有属性通过 metadata dict 承载，避免模型膨胀
    """
    index: int = Field(description='块在文档中的顺序位置，从 0 开始')
    block_type: ContentBlockType = Field(description='内容块的模态类型')
    
    # --- 核心内容字段（按 block_type 选择性填充） ---
    text: str | None = Field(
        default=None,
        description='文本内容。text/table/code/formula 类型必填；'
                    '其他类型可选（如图片的 OCR 文本）',
    )
    s3_uri: str | None = Field(
        default=None,
        description='S3 对象 URI，格式 s3://{bucket}/{key}。'
                    'image/audio/video/chart/slide 类型必填',
    )
    
    # --- 通用元数据 ---
    mime_type: str | None = Field(
        default=None,
        description='MIME 类型（如 image/jpeg, audio/wav, video/mp4）',
    )
    description: str | None = Field(
        default=None,
        description='LLM 生成的内容描述，用于：'
                    '1. 纯文本降级（不支持多模态的 LLM）'
                    '2. 搜索结果展示 3. content 字段的文本化表示',
    )
    semantic_role: SemanticRole = Field(
        default=SemanticRole.body,
        description='语义角色，与 block_type 正交',
    )
    language: str | None = Field(
        default=None,
        description='内容语言（ISO 639-1），如 zh, en, ja',
    )
    
    # --- 来源追溯 ---
    source_page: int | None = Field(
        default=None,
        description='来源页码（PDF/PPT 等分页文档），从 1 开始',
    )
    source_location: dict | None = Field(
        default=None,
        description='来源位置信息（可选），如 bounding box：'
                    '{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}',
    )
    
    # --- 层次结构 ---
    parent_index: int | None = Field(
        default=None,
        description='父块的 index（用于表格单元格→表格、列表项→列表等层次关系）。'
                    '顶层块此字段为 None',
    )
    
    # --- 扩展元数据 ---
    metadata: dict | None = Field(
        default=None,
        description='类型特有的扩展元数据，避免模型字段膨胀。示例：'
                    'image: {"width": 800, "height": 600, "format": "jpeg"}'
                    'audio: {"duration_sec": 120, "sample_rate": 16000, "channels": 1}'
                    'video: {"duration_sec": 300, "fps": 30, "resolution": "1920x1080"}'
                    'table: {"rows": 5, "cols": 3, "has_header": true}'
                    'code: {"language": "python"}'
                    'slide: {"slide_number": 3, "layout": "title_and_content"}',
    )
    
    # --- 处理状态 ---
    embedding_generated: bool = Field(
        default=False,
        description='该块是否已生成 embedding（用于增量处理）',
    )
    
    # --- 临时字段（不序列化，不存入 Neo4j） ---
    _raw_bytes: bytes | None = None
    """二进制原始数据（图片、音频等）。
    
    仅在预处理管道中使用：DocumentParser.parse() 填充 → 上传 S3 后清除。
    使用 Python 私有属性（非 Pydantic Field），不参与 model_dump() 序列化，
    不存入 Neo4j，不出现在 JSON 中。
    
    注意：Pydantic v2 中需要在 model_config 中设置：
        model_config = ConfigDict(arbitrary_types_allowed=True)
    """
    # 预计算的 embedding（不序列化）— 图片等二进制块的向量，
    # 在 add_document_episode 中生成，供后续 source_excerpt embedding 使用
    _embedding: list[float] | None = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def is_binary(self) -> bool:
        """是否为二进制内容（需要 S3 存储）"""
        return self.block_type in (
            ContentBlockType.image,
            ContentBlockType.audio,
            ContentBlockType.video,
            ContentBlockType.chart,
            ContentBlockType.slide,
        )
    
    @property
    def text_representation(self) -> str:
        """获取该块的纯文本表示"""
        if self.text:
            return self.text
        if self.description:
            return f'[{self.block_type.value}: {self.description}]'
        if self.s3_uri:
            return f'[{self.block_type.value}: {self.s3_uri}]'
        return f'[{self.block_type.value}]'
```

**设计决策详解**：

| 决策 | 理由 | 参考 |
|------|------|------|
| `block_type` 按模态分类，`semantic_role` 按语义分类 | 两个维度正交：一张图片可以是正文插图（body）也可以是页眉 logo（header）；一段文本可以是标题也可以是脚注 | Docling 的 body/furniture 分层 |
| `parent_index` 支持层次结构 | 表格包含单元格、列表包含列表项，需要父子关系。用 index 引用而非嵌套对象，保持 JSON 序列化扁平 | EdgeQuake 的 Block.children |
| `metadata: dict` 承载类型特有属性 | 不同模态的元数据差异极大（图片有宽高，音频有时长采样率，视频有帧率分辨率），用 dict 避免模型字段爆炸 | Unstructured 的 Element.metadata |
| `source_page` + `source_location` | 支持 PDF/PPT 等分页文档的来源追溯，bounding box 用于精确定位 | Docling 的 provenance + EdgeQuake 的 bbox |
| `text` 字段对所有类型可选 | 图片可能有 OCR 文本，音频可能有转录文本，表格有 Markdown 表示 — 统一用 `text` 承载 | Docling 的 text 统一表示 |
| `embedding_generated` 状态标记 | 增量处理时避免重复 embedding 计算 | 实际需求 |
| `text_representation` 属性 | 统一的纯文本降级逻辑，`build_content_from_blocks()` 直接调用 | 简化调用方代码 |
| `_raw_bytes` 私有属性 | 预处理管道中暂存二进制数据，不参与序列化/存储 | 避免 Neo4j 存储二进制数据；上传 S3 后清除 |
| `_embedding` 私有属性 | 预计算的图片 embedding，在 `add_document_episode` 中生成（必须在 S3 上传清除 `_raw_bytes` 之前），供后续 `source_excerpt` embedding 使用 | 通过 `build_image_embedding_map()` 建立 `s3_uri → embedding` 映射 |

**各模态的字段使用约定**：

| block_type | text | s3_uri | mime_type | description | metadata 示例 |
|-----------|------|--------|-----------|-------------|--------------|
| text | ✅ 必填（原文） | ❌ | ❌ | ❌ | `None` |
| image | ⚪ 可选（OCR） | ✅ 必填 | ✅ image/jpeg | ✅ LLM 描述 | `{"width": 800, "height": 600}` |
| table | ✅ 必填（Markdown 表示） | ⚪ 可选（表格截图） | ⚪ | ✅ LLM 摘要 | `{"rows": 5, "cols": 3, "has_header": true}` |
| audio | ⚪ 可选（转录文本） | ✅ 必填 | ✅ audio/wav | ✅ LLM 摘要 | `{"duration_sec": 120, "sample_rate": 16000}` |
| video | ⚪ 可选（转录文本） | ✅ 必填 | ✅ video/mp4 | ✅ LLM 摘要 | `{"duration_sec": 300, "fps": 30}` |
| code | ✅ 必填（代码文本） | ❌ | ❌ | ❌ | `{"language": "python"}` |
| formula | ✅ 必填（LaTeX） | ⚪ 可选（公式图片） | ⚪ | ✅ 文字描述 | `{"format": "latex"}` |
| chart | ⚪ 可选（数据表） | ✅ 必填 | ✅ image/png | ✅ LLM 描述 | `{"chart_type": "bar", "data_points": 12}` |
| slide | ⚪ 可选（幻灯片文本） | ✅ 必填 | ✅ image/png | ✅ LLM 描述 | `{"slide_number": 3, "layout": "title_and_content"}` |

✅ = 必填，⚪ = 可选，❌ = 不适用

#### 13.3.2 EpisodicNode 扩展

```python
class EpisodeType(Enum):
    message = 'message'
    json = 'json'
    text = 'text'
    # 新增：按文档格式扩展，而非按模态
    document = 'document'  # 结构化文档（Word, PDF, PPT 等，可能包含多模态内容）

    @staticmethod
    def from_str(episode_type: str):
        ...
        if episode_type == 'document':
            return EpisodeType.document
        ...
```

> **设计决策**：用 `document` 而非 `multimodal` 作为枚举值。原因：一个 Word 文档即使只包含纯文本，也应该用 `document` 类型（因为它经过了文档解析流程）。`multimodal` 暗示"一定包含多种模态"，但实际上文档类型的 episode 可能只有文本块。`document` 更准确地描述了数据来源。

```python
class EpisodicNode(Node):
    source: EpisodeType = Field(description='source type')
    source_description: str = Field(description='description of the data source')
    content: str = Field(description='raw episode data (text) or text representation of document content')
    content_blocks: list[ContentBlock] = Field(
        default_factory=list,
        description='有序内容块列表，document 类型 episode 的完整内容表示。'
                    '纯文本 episode 此字段为空列表（向后兼容）。',
    )
    valid_at: datetime = Field(...)
    entity_edges: list[str] = Field(...)
    narrative_excerpts: list[str] = Field(...)
    describes_edges: list[str] = Field(...)
```

**`content` 与 `content_blocks` 的关系**：

| Episode 类型 | `content` | `content_blocks` |
|-------------|-----------|-------------------|
| 纯文本（text/message/json） | 原文全文 | `[]`（空列表，向后兼容） |
| 文档（document） | 文本化表示（见下方） | 完整的有序内容块列表 |

文档类型 episode 的 `content` 字段生成规则：

```python
def build_content_from_blocks(blocks: list[ContentBlock]) -> str:
    """将 content_blocks 转换为纯文本表示，用于：
    1. LLM 提取（不支持视觉的 LLM 降级）
    2. 文本 embedding（episode-content-embeddings 索引）
    3. 日志和调试
    """
    parts = []
    for block in sorted(blocks, key=lambda b: b.index):
        # 跳过子块（由父块的 text_representation 处理）
        if block.parent_index is not None:
            continue
        parts.append(block.text_representation)
    return '\n'.join(parts)
```

这样 `content` 始终是一个可读的文本字符串，所有依赖 `content` 的现有逻辑（embedding、dedup、日志）无需修改。

#### 13.3.3 Neo4j 存储

`content_blocks` 在 Neo4j 中以 JSON 字符串存储（与 `narrative_excerpts` 相同策略）：

```python
# EpisodicNode.save() 中
episode_args = {
    ...
    'content_blocks': json.dumps(
        [block.model_dump() for block in self.content_blocks],
        ensure_ascii=False,
    ) if self.content_blocks else '[]',
    ...
}
```

```cypher
-- Cypher 写入
SET e.content_blocks = $content_blocks

-- Cypher 读取
e.content_blocks AS content_blocks
```

```python
# get_episodic_node_from_record() 中
raw_blocks = record.get('content_blocks', '[]')
content_blocks = []
if raw_blocks and isinstance(raw_blocks, str):
    try:
        content_blocks = [ContentBlock(**b) for b in json.loads(raw_blocks)]
    except (json.JSONDecodeError, TypeError):
        content_blocks = []
```

**向后兼容**：已有的纯文本 episode 没有 `content_blocks` 属性，读取时 `record.get('content_blocks', '[]')` 返回默认空列表。

#### 13.3.4 S3 多模态资产存储

多模态资产（图片等）存储在独立的 S3 桶中，与 S3 Vectors 桶分离：

```
S3 桶结构：
  s3://{MULTIMODAL_ASSET_BUCKET}/
    └── {group_id}/
        └── {episode_uuid}/
            ├── block_001.jpeg
            ├── block_003.png
            └── ...
```

**Key 命名规则**：`{group_id}/{episode_uuid}/block_{index:03d}.{ext}`

- `group_id` 作为一级前缀，便于按组清理
- `episode_uuid` 作为二级前缀，便于按 episode 清理（`remove_episode` 时删除整个前缀）
- `block_{index}` 保持与 ContentBlock.index 对应
- 扩展名从 `mime_type` 推导

**配置**：

```python
class MultimodalStorageConfig(BaseModel):
    bucket: str = Field(description='S3 桶名')
    region: str = Field(default='us-east-1')
    # 后续可扩展：presigned URL 过期时间、加密配置等
```

通过环境变量 `MULTIMODAL_ASSET_BUCKET` 配置，未配置时多模态功能不可用。

### 13.4 预处理模块设计

#### 13.4.1 文档解析器接口

```python
from abc import ABC, abstractmethod

class DocumentParser(ABC):
    """文档解析器基类，将文档转换为有序 ContentBlock 列表"""

    @abstractmethod
    async def parse(self, file_path: str) -> list[ContentBlock]:
        """解析文档，返回有序内容块列表。
        
        图片等二进制内容暂存在 ContentBlock._raw_bytes 私有属性中，
        由调用方负责上传到 S3 并填充 s3_uri，然后清除 _raw_bytes。
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """返回支持的文件扩展名列表"""
        ...
```

#### 13.4.2 Word 文档解析器（首个实现）

```python
class WordDocumentParser(DocumentParser):
    """解析 .docx 文件，提取文本段落和嵌入图片"""

    def supported_extensions(self) -> list[str]:
        return ['.docx']

    async def parse(self, file_path: str) -> list[ContentBlock]:
        """
        遍历 docx 的 XML 结构，按顺序提取：
        - 文本段落 → ContentBlock(block_type=text, text=...)
        - 嵌入图片 → ContentBlock(block_type=image, _raw_bytes=..., mime_type=...)
        
        连续的文本段落合并为一个 ContentBlock（减少块数量）。
        图片的 _raw_bytes 是私有属性（不参与序列化），上传 S3 后由调用方
        填充 s3_uri 并清除 _raw_bytes。
        """
        ...
```

**依赖**：`python-docx` 库（已在 requirements.txt 中或按需添加）。

#### 13.4.3 解析器注册与路由

```python
class DocumentParserRegistry:
    """解析器注册表，根据文件扩展名路由到对应解析器"""

    def __init__(self):
        self._parsers: dict[str, DocumentParser] = {}

    def register(self, parser: DocumentParser):
        for ext in parser.supported_extensions():
            self._parsers[ext.lower()] = parser

    def get_parser(self, file_path: str) -> DocumentParser | None:
        ext = Path(file_path).suffix.lower()
        return self._parsers.get(ext)

# 默认注册
default_registry = DocumentParserRegistry()
default_registry.register(WordDocumentParser())
# 后续扩展：
# default_registry.register(PDFDocumentParser())
# default_registry.register(MarkdownDocumentParser())
```

**模块化设计**：新增文档类型只需实现 `DocumentParser` 并注册，不影响核心逻辑。

### 13.5 Embedding 扩展

#### 13.5.1 Nova MME 多模态 Embedding

当前 `BedrockNovaEmbedder` 只支持文本输入。Nova MME（`amazon.nova-2-multimodal-embeddings-v1:0`）原生支持图片 embedding，且文本和图片在同一 1024 维语义空间中。

**Nova MME 图片 embedding 请求格式**：

```json
{
    "schemaVersion": "nova-multimodal-embed-v1",
    "taskType": "SINGLE_EMBEDDING",
    "singleEmbeddingParams": {
        "embeddingPurpose": "GENERIC_INDEX",
        "embeddingDimension": 1024,
        "image": {
            "format": "jpeg",
            "source": {
                "bytes": "<base64_encoded_image_data>"
            }
        }
    }
}
```

**文本+图片混合 embedding 请求格式**：

```json
{
    "schemaVersion": "nova-multimodal-embed-v1",
    "taskType": "SINGLE_EMBEDDING",
    "singleEmbeddingParams": {
        "embeddingPurpose": "GENERIC_INDEX",
        "embeddingDimension": 1024,
        "text": {
            "truncationMode": "END",
            "value": "描述文本"
        },
        "image": {
            "format": "jpeg",
            "source": {
                "bytes": "<base64_encoded_image_data>"
            }
        }
    }
}
```

> **实现说明**：经实际验证，Nova MME API 每次请求只支持**单一模态**（text 或 image 或 audio 或 video），不支持 image+text 混合 embedding。上述混合格式虽然在文档中有描述，但实际调用会报错。因此实现中 `create_image()` 仅使用纯图片 embedding，`text` 参数保留在接口中以备后续 API 更新，但当前被忽略。文本和图片的 embedding 虽然分别生成，但处于同一 1024 维语义空间，跨模态检索仍然有效。

#### 13.5.2 EmbedderClient 接口扩展

```python
class EmbedderClient(ABC):
    @abstractmethod
    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """文本 embedding（现有接口，不变）"""
        pass

    async def create_image(
        self, image_bytes: bytes, image_format: str = 'jpeg',
        text: str | None = None,
    ) -> list[float]:
        """图片 embedding（新增，可选附带文本上下文）
        
        默认抛出 NotImplementedError，只有支持多模态的 embedder 实现此方法。
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not support image embedding'
        )

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        raise NotImplementedError()
```

#### 13.5.3 BedrockNovaEmbedder 扩展

```python
class BedrockNovaEmbedder(EmbedderClient):
    # ... 现有 create() 方法不变 ...

    def _build_image_request_body(
        self, image_bytes: bytes, image_format: str, text: str | None = None,
    ) -> dict:
        """Build Nova MME request body for image embedding.
        
        Note: Nova MME requires exactly one modality per request
        (text, image, audio, or video). Combined image+text in a single
        request is NOT supported. The `text` parameter is accepted for
        interface compatibility but ignored — image-only embedding is used.
        """
        params: dict = {
            'embeddingPurpose': 'GENERIC_INDEX',
            'embeddingDimension': self.config.embedding_dim,
            'image': {
                'format': image_format,
                'source': {
                    'bytes': base64.b64encode(image_bytes).decode('utf-8'),
                },
            },
        }
        # Nova MME: exactly one of text/image/audio/video per request.
        # text parameter is intentionally NOT added here.
        return {
            'schemaVersion': 'nova-multimodal-embed-v1',
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': params,
        }

    async def create_image(
        self, image_bytes: bytes, image_format: str = 'jpeg',
        text: str | None = None,
    ) -> list[float]:
        body = json.dumps(self._build_image_request_body(
            image_bytes, image_format, text,
        ))
        # 调用逻辑与 create() 相同，复用 invoke_model
        response = self.bedrock.invoke_model(
            modelId=self.config.model_id,
            body=body,
            contentType='application/json',
            accept='application/json',
        )
        result = json.loads(response['body'].read())
        embeddings = result.get('embeddings', [])
        if embeddings:
            return embeddings[0]['embedding'][:self.config.embedding_dim]
        raise ValueError(f'No embeddings in Nova response: {result}')
```

#### 13.5.4 跨模态检索的向量空间一致性

Nova MME 的关键特性：文本和图片 embedding 在同一语义空间中。这意味着：

- 文本 query "一个穿黄袍的人" 可以匹配到黄袍人物的图片 embedding
- 图片 query（某人照片）可以匹配到描述该人的文本 embedding
- 现有的 S3 Vectors 索引无需拆分，文本向量和图片向量可以共存于同一索引

**维度选择**：维持 1024 维。Nova MME 支持 256/384/1024/3072，1024 是精度和性能的平衡点。所有索引统一 1024 维，确保跨模态检索的向量空间一致。

### 13.6 LLM 提取链路扩展

#### 13.6.1 视觉 LLM 提取策略

当前 LLM 提取链路（extract_nodes → extract_edges）接收纯文本 `episode_content`。多模态 episode 需要将图片信息传递给 LLM。

**策略**：Kimi K2.5 支持视觉理解，图片直接以 base64 编码发送给 LLM。

**两种提取模式**：

| 模式 | 适用场景 | 实现方式 |
|------|---------|---------|
| 视觉模式 | LLM 支持视觉（Kimi K2.5） | 图片以 base64 嵌入 prompt，LLM 直接理解图片内容 |
| 降级模式 | LLM 不支持视觉 | 图片替换为 `description` 文本，退化为纯文本提取 |

#### 13.6.2 Prompt 改造

`extract_nodes.py` 和 `extract_edges.py` 新增独立的多模态 prompt 函数，而非修改现有函数。通过 `_call_extraction_llm` 和 `extract_edges` 中的 `EpisodeType.document` 分支进行调度。

**实体提取**：新增 `extract_document()` 函数（`extract_nodes.py`），加入 `Versions` TypedDict。`_call_extraction_llm`（`node_operations.py`）新增 `EpisodeType.document` 分支调用 `extract_document()`。`extract_nodes()` 在构建 context 时，对 document 类型 episode 传入 `content_blocks`。

**关系提取**：新增 `edge_document()` 函数（`extract_edges.py`），加入 `Prompt` Protocol 和 `Versions` TypedDict。`extract_edges()`（`edge_operations.py`）检测 `episode.source == EpisodeType.document` 时调用 `edge_document()` 而非 `edge()`。

**多模态 prompt 构建逻辑**（`extract_document` 和 `edge_document` 共用模式）：

```python
def extract_document(context: dict[str, Any]) -> list[Message]:
    """构建包含图片的多模态提取 prompt。
    
    图片以 content 数组形式传递（OpenAI multimodal format）：
    [
        {"type": "text", "text": "文本内容..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "更多文本..."},
    ]
    """
    content_blocks = context.get('content_blocks', [])
    user_parts = []

    # 前置指令
    user_parts.append({
        'type': 'text',
        'text': f'<ENTITY TYPES>\n{context["entity_types"]}\n</ENTITY TYPES>\n\n'
                '<DOCUMENT CONTENT>\n'
    })

    # 按顺序插入内容块
    for block in sorted(content_blocks, key=lambda b: b.index):
        if block.parent_index is not None:
            continue
        if block.block_type == ContentBlockType.text:
            if block.text:
                user_parts.append({'type': 'text', 'text': block.text})
        elif block.block_type == ContentBlockType.image:
            if block._raw_bytes:
                # 视觉模式：直接发送图片 base64
                b64 = base64.b64encode(block._raw_bytes).decode('utf-8')
                mime = block.mime_type or 'image/jpeg'
                user_parts.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:{mime};base64,{b64}'},
                })
            elif block.description:
                # 降级模式：用 text_representation 替代
                user_parts.append({'type': 'text', 'text': block.text_representation})
        else:
            user_parts.append({'type': 'text', 'text': block.text_representation})

    user_parts.append({'type': 'text', 'text': '\n</DOCUMENT CONTENT>\n\n' + INSTRUCTIONS})

    return [
        Message(role='system', content=SYS_PROMPT),
        Message(role='user', content=user_parts),
    ]
```

`edge_document` 的额外处理：对于已上传到 S3 的图片（有 `s3_uri` 但无 `_raw_bytes`），在 prompt 中以 `[image:s3://...]` 格式引用，并指导 LLM 在 `source_excerpt` 中使用相同格式引用图片来源。

**注意**：`Message.content` 已扩展为 `str | list[dict[str, Any]]`。`LLMClient._clean_input()` 和 `generate_response()` 已支持多模态 content 数组。

#### 13.6.3 图片描述生成

在预处理阶段，对每张图片调用视觉 LLM 生成文字描述，存入 `ContentBlock.description`：

```python
async def generate_image_descriptions(
    llm_client: LLMClient,
    blocks: list[ContentBlock],
    group_id: str | None = None,
) -> list[ContentBlock]:
    """为图片类型的 ContentBlock 生成文字描述"""
    for block in blocks:
        if block.block_type == ContentBlockType.image and block._raw_bytes:
            if block.description:
                continue  # 已有描述则跳过
            description = await llm_client.generate_response(
                messages=[
                    Message(role='system', content='你是一个图片描述助手。'),
                    Message(
                        role='user',
                        content=[
                            {'type': 'image_url', 'image_url': {
                                'url': f'data:{block.mime_type};base64,'
                                       f'{base64.b64encode(block._raw_bytes).decode()}',
                            }},
                            {'type': 'text', 'text': '请用一段简洁的中文描述这张图片的内容。'
                                                      '回复格式为纯 JSON：{"description": "..."}'},
                        ],
                    ),
                ],
                max_tokens=512,
                model_size=ModelSize.medium,
                group_id=group_id,
                prompt_name='generate_image_description',
            )
            block.description = description.get('description', '') if isinstance(description, dict) else str(description)
    return blocks
```

**用途**：
1. 构建 `content` 字段的文本化表示
2. 不支持视觉的 LLM 降级提取
3. 搜索结果展示

#### 13.6.4 source_excerpt 与 narrative_excerpts 的多模态引用

当 LLM 从图片中提取出实体/关系时，`source_excerpt` 和 `narrative_excerpts` 需要引用图片来源：

**规则**：
- 如果 excerpt 来自文本块：保持原样（逐字引用原文）
- 如果 excerpt 来自二进制块（image/audio/video/chart/slide）：使用 S3 URI 作为引用，格式 `[{type}:{s3_uri}] {description}`
- 如果 excerpt 来自其他文本类型块（table/code/formula）：使用 `text_representation`

```python
def build_excerpt_reference(block: ContentBlock) -> str:
    """构建 excerpt 引用"""
    if block.block_type == ContentBlockType.text:
        return block.text  # 原文逐字引用（由 LLM 在 prompt 中完成）
    elif block.is_binary and block.s3_uri:
        # 二进制类型（image/audio/video/chart/slide）：用 S3 URI + 描述
        desc = block.description or ''
        return f'[{block.block_type.value}:{block.s3_uri}] {desc}'
    else:
        # 其他文本类型（table/code/formula）：使用 text_representation
        return block.text_representation
```

这样 `source_excerpt` 和 `narrative_excerpts` 的类型仍然是 `str`，不需要改变数据模型，但内容中可以包含多模态引用标记（如 `[image:s3://...]`、`[audio:s3://...]` 等）。检索结果展示时，前端可以解析 `[{type}:{s3_uri}]` 标记并渲染对应的多媒体内容。

### 13.7 add_episode 流程扩展

#### 13.7.1 新增 add_document_episode 入口

不修改现有 `add_episode()` 签名，新增独立入口处理文档类型 episode：

```python
async def add_document_episode(
    self,
    name: str,
    file_path: str | None = None,
    content_blocks: list[ContentBlock] | None = None,
    source_description: str = '',
    reference_time: datetime | None = None,
    group_id: str | None = None,
    uuid: str | None = None,
    update_communities: bool = False,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    custom_extraction_instructions: str | None = None,
    saga: str | SagaNode | None = None,
    saga_previous_episode_uuid: str | None = None,
) -> AddEpisodeResults:
    """处理文档类型 episode（可能包含多模态内容）并更新图谱。
    
    两种输入方式（二选一）：
    1. file_path: 文档文件路径，自动解析为 content_blocks
    2. content_blocks: 预解析的内容块列表（跳过解析步骤）
    """
```

#### 13.7.2 处理流程

```
add_document_episode()
  │
  ├─ Step 1: 文档解析（如果传入 file_path）
  │   DocumentParserRegistry.get_parser(file_path).parse()
  │   → list[ContentBlock]（含 _raw_bytes）
  │
  ├─ Step 2: 生成图片 embedding（必须在 S3 上传前，因为上传会清除 _raw_bytes）
  │   对每个 image 类型的 block（有 _raw_bytes）：
  │   embedder.create_image(block._raw_bytes) → 存入 block._embedding
  │
  ├─ Step 3: 上传多模态资产到 S3（填充 s3_uri，保留 _raw_bytes）
  │   MultimodalAssetStorage.upload_blocks(blocks, group_id, episode_uuid,
  │                                        clear_raw_bytes=False)
  │
  ├─ Step 4: 生成图片描述（通过视觉 LLM，使用 _raw_bytes）
  │   generate_image_descriptions(llm_client, blocks)
  │   → 填充 block.description
  │
  ├─ Step 5: 构建 content 文本表示
  │   content = build_content_from_blocks(blocks)
  │
  ├─ Step 6: 设置 image_embedding_map（s3_uri → embedding 映射）
  │   build_image_embedding_map(blocks) → clients.image_embedding_map
  │   供后续 edge/describes/narrative embedding 使用图片向量
  │
  ├─ Step 7: 委托给 add_episode()
  │   content_blocks 中 _raw_bytes 仍在
  │   → extract_document() 构建 base64 image_url（视觉模式）
  │   → edge_document() 构建 base64 image_url（视觉模式）
  │
  └─ finally:
      ├─ 清除所有 block._raw_bytes（释放内存）
      └─ 清除 clients.image_embedding_map（瞬态数据）
```

**关键设计决策**：

1. **委托而非重写**：`add_document_episode` 在预处理后委托给 `add_episode()`，复用全部现有逻辑（节点提取、边提取、去重、保存、S3 Vectors 同步）
2. **content 作为桥梁**：多模态内容通过 `build_content_from_blocks()` 转换为文本，现有的 `add_episode()` 无需感知多模态
3. **embedding 先于上传**：图片 embedding 必须在 S3 上传前生成，因为 `upload_blocks()` 默认会清除 `_raw_bytes` 以释放内存。embedding 存入 `_embedding` 临时字段，上传后通过 `s3_uri` 建立映射
4. **image_embedding_map 瞬态**：设置在 `GraphitiClients` 上，供 `add_episode()` 内部的 edge embedding 函数使用，在 `finally` 块中清除
5. **延迟清除 _raw_bytes**：S3 上传时传 `clear_raw_bytes=False` 保留原始图片数据，使 Step 4（图片描述）和 Step 7（LLM 提取）能直接访问原始图片。`_raw_bytes` 在 `finally` 块中统一清除（见 13.16）

#### 13.7.3 add_episode() UUID 兼容性修复

**问题**：`add_document_episode()` 在委托给 `add_episode()` 时会预先生成一个新的 UUID 并传入。但 `add_episode()` 原有逻辑在收到 `uuid` 参数时，会尝试通过 `EpisodicNode.get_by_uuid()` 从数据库获取已有节点。由于这是一个全新的 UUID（数据库中不存在），会抛出 `NodeNotFoundError`，导致整个导入流程失败。

**修复**（`graphiti_core/graphiti.py`）：

```python
# 修复前：
if uuid is not None:
    episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
else:
    episode = EpisodicNode(...)

# 修复后：
if uuid is not None:
    try:
        episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
    except NodeNotFoundError:
        # UUID 由调用方（如 add_document_episode）预先生成，
        # 数据库中尚不存在，创建新节点并使用该 UUID
        episode = EpisodicNode(uuid=uuid, ...)
else:
    episode = EpisodicNode(...)
```

**影响范围**：仅影响传入 `uuid` 参数但数据库中不存在该 UUID 的场景。现有的纯文本 `add_episode()` 调用（不传 uuid）行为完全不变。

### 13.8 检索扩展

#### 13.8.1 多模态搜索输入

当前 `search()` 和 `search_()` 接收 `query: str`。跨模态检索需要支持图片作为搜索条件：

> **注意**：首期仅支持图片作为多模态搜索条件。音频/视频作为 query 输入需要额外的预处理（如关键帧提取），属于后续扩展范围（见 13.12）。

```python
async def search_(
    self,
    query: str,
    query_image: bytes | None = None,  # 新增：图片搜索条件
    query_image_format: str = 'jpeg',  # 新增：图片格式
    config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    ...
) -> SearchResults:
```

**搜索向量生成**：

```python
if query_image is not None:
    # 图片+文本混合 embedding（如果同时有 query 文本）
    search_vector = await self.embedder.create_image(
        query_image, query_image_format, text=query if query else None,
    )
elif query:
    search_vector = await self.embedder.create(query)
```

由于 Nova MME 的文本和图片 embedding 在同一语义空间，图片 query 的向量可以直接在现有的文本向量索引中搜索，无需额外索引。

#### 13.8.2 搜索结果中的多模态引用

搜索结果中的 `source_excerpt` 可能包含多模态引用标记 `[{type}:s3://...]`。前端需要：

1. 解析 `[{type}:{s3_uri}]` 标记（type 可以是 image、audio、video、chart、slide）
2. 生成 presigned URL 用于多媒体内容展示
3. 同时展示 description 文本作为 fallback

**辅助函数**：

```python
def parse_excerpt_references(excerpt: str) -> list[dict]:
    """解析 excerpt 中的多模态引用"""
    import re
    refs = []
    # 匹配所有多模态引用标记：[type:s3://...] description
    pattern = r'\[(image|audio|video|chart|slide):(s3://[^\]]+)\]\s*(.*?)(?=\[(?:image|audio|video|chart|slide):|$)'
    for match in re.finditer(pattern, excerpt):
        refs.append({
            'type': match.group(1),
            's3_uri': match.group(2),
            'description': match.group(3).strip(),
        })
    # 纯文本部分
    text_parts = re.sub(pattern, '', excerpt).strip()
    if text_parts:
        refs.insert(0, {'type': 'text', 'content': text_parts})
    return refs
```

#### 13.8.3 深度搜索的多模态扩展

`deep_search` 的 `source_similarity` 搜索方法天然支持多模态：

- Edge 的 `source_excerpt` embedding 可能来自多模态内容描述 → 图片 query 可以匹配
- DescribesEdge 的 `excerpt` embedding 同理
- Episode narrative embedding 同理

无需额外改动，跨模态检索通过向量空间的统一性自动实现。

#### 13.8.4 空查询兼容性修复（图片搜索支持）

**问题**：`search_()` 函数在入口处有一个空查询短路逻辑：

```python
# 修复前：
if query.strip() == '':
    return SearchResults()
```

当用户进行纯图片搜索时（`query=''`，`query_image=<bytes>`），`search_()` 会先通过 `embedder.create_image()` 生成 `query_vector`，然后调用底层 `search()` 函数。但 `search()` 中的上述短路逻辑会在 `query` 为空时直接返回空结果，即使 `query_vector` 已经有效。

**修复**（`graphiti_core/search/search.py`）：

```python
# 修复后：
if query.strip() == '' and query_vector is None:
    return SearchResults()
```

只有当文本查询为空**且**没有预计算的查询向量时才短路返回。这样图片搜索（有 `query_vector` 但无文本 `query`）可以正常执行。

**注意**：图片搜索时 BM25 全文检索路径自然不会触发（空文本无法驱动 Lucene fulltext），只有向量检索路径生效。因此图片搜索应使用 vector-only 的 SearchConfig（不包含 BM25/fulltext 方法），避免无意义的全文检索调用。

### 13.9 分步实施计划

改造分为 5 个 Phase，每个 Phase 独立可测试。

---

#### Phase 1：数据模型层

**目标**：引入 `ContentBlock`、`ContentBlockType`、扩展 `EpisodeType`、扩展 `EpisodicNode`。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/nodes.py` | 新增 `ContentBlockType`、`SemanticRole`、`ContentBlock` 模型；`EpisodeType` 新增 `document`；`EpisodicNode` 新增 `content_blocks` 字段 |
| `graphiti_core/models/nodes/node_db_queries.py` | Cypher 写入/读取增加 `content_blocks` 字段 |

**验证**：
- 纯文本 episode 的 `content_blocks` 为空列表，行为不变
- 多模态 episode 的 `content_blocks` 能正确序列化/反序列化
- 已有数据读取时 `content_blocks` 默认为空列表（向后兼容）
- 全部现有测试通过

---

#### Phase 2：预处理模块

**目标**：实现文档解析器框架和 Word 解析器。

**新增文件**：

| 文件 | 内容 |
|------|------|
| `graphiti_core/preprocessing/__init__.py` | 模块入口 |
| `graphiti_core/preprocessing/parser.py` | `DocumentParser` 基类、`DocumentParserRegistry` |
| `graphiti_core/preprocessing/word_parser.py` | `WordDocumentParser` 实现 |
| `graphiti_core/preprocessing/asset_storage.py` | S3 多模态资产上传/删除 |

**验证**：
- Word 文档解析：文本段落和嵌入图片按顺序提取
- 连续文本段落合并
- 图片上传到 S3 并生成正确的 s3_uri
- 解析器注册和路由正确

---

#### Phase 3：Embedding 扩展

**目标**：`BedrockNovaEmbedder` 支持图片 embedding。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/embedder/client.py` | `EmbedderClient` 新增 `create_image()` 方法（默认 NotImplementedError） |
| `graphiti_core/embedder/bedrock_nova.py` | 实现 `create_image()`，支持纯图片和图片+文本混合 embedding |

**验证**：
- 文本 embedding 行为不变
- 图片 embedding 返回 1024 维向量
- 图片+文本混合 embedding 返回 1024 维向量
- 不支持图片的 embedder（如 OpenAI）调用 `create_image()` 抛出 NotImplementedError

---

#### Phase 4：add_document_episode 入口

**目标**：实现文档类型 episode 导入的完整流程。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/graphiti.py` | 新增 `add_document_episode()` 方法 |
| `graphiti_core/preprocessing/description.py` | 新增 `generate_image_descriptions()` |
| `graphiti_core/prompts/models.py` | `Message.content` 类型扩展为 `str \| list[dict[str, Any]]` |
| `graphiti_core/llm_client/client.py` | `_clean_input()` 和 `generate_response()` 支持多模态 content |
| `graphiti_core/llm_client/bedrock_client.py` | 多模态消息格式支持 |
| `graphiti_core/prompts/extract_nodes.py` | 新增 `extract_document()` 函数 |
| `graphiti_core/prompts/extract_edges.py` | 新增 `edge_document()` 函数 |
| `graphiti_core/utils/maintenance/node_operations.py` | `_call_extraction_llm` 新增 `EpisodeType.document` 分支 |
| `graphiti_core/utils/maintenance/edge_operations.py` | `extract_edges` 新增 `EpisodeType.document` 分支 |
| `graphiti_core/graphiti_types.py` | `GraphitiClients` 新增 `image_embedding_map` 字段 |
| `graphiti_core/edges.py` | `create_entity_edge_embeddings` 支持 `image_embedding_map` |
| `graphiti_core/nodes.py` | 新增 `build_image_embedding_map()`、`extract_s3_uri_from_excerpt()`、`parse_excerpt_references()` |

**流程**（与 13.7.2 一致的 7 步）：
1. 解析文档 → content_blocks
2. 生成图片 embedding（必须在 S3 上传前，因为上传会清除 `_raw_bytes`）
3. 上传图片到 S3 → 填充 s3_uri，清除 `_raw_bytes`
4. 生成图片描述 → 填充 description
5. 构建 content 文本表示
6. 设置 image_embedding_map（s3_uri → embedding 映射）
7. 委托 add_episode()（finally 中清除 image_embedding_map）

> **⚠️ 已更新**：上述为两阶段基线流程。单阶段改造后，Step 3 传 `clear_raw_bytes=False` 保留 `_raw_bytes`，Step 7 中 LLM 直接看到原始图片，finally 中统一清除 `_raw_bytes`。详见 13.16 和更新后的 13.7.2 流程图。

**验证**：
- Word 文档端到端导入：解析 → 上传 → 描述 → 提取 → 保存
- content_blocks 正确存储在 Neo4j
- content 字段包含图片描述的文本化表示
- 纯文本 add_episode() 行为完全不变

---

#### Phase 5：搜索扩展

**目标**：支持图片作为搜索条件。

**改动文件**：

| 文件 | 改动 |
|------|------|
| `graphiti_core/graphiti.py` | `search_()` 新增 `query_image`、`query_image_format` 参数，图片通过 `embedder.create_image()` 生成搜索向量，传入 `search()` 的 `query_vector` 参数 |

**验证**：
- 文本搜索行为不变
- 图片搜索返回语义相关的文本结果
- 文本搜索返回语义相关的图片引用结果
- deep_search + 图片搜索组合正常工作

### 13.10 改动文件汇总

| Phase | 文件 | 改动类型 |
|-------|------|---------|
| 1 | `graphiti_core/nodes.py` | 修改（ContentBlock、SemanticRole、EpisodeType、EpisodicNode、build_image_embedding_map、extract_s3_uri_from_excerpt、parse_excerpt_references） |
| 1 | `graphiti_core/models/nodes/node_db_queries.py` | 修改（Cypher 增加 content_blocks） |
| 1 | `graphiti_core/driver/record_parsers.py` | 修改（解析 content_blocks） |
| 1 | `graphiti_core/driver/neo4j/operations/episode_node_ops.py` | 修改（content_blocks 读写） |
| 1 | `graphiti_core/driver/falkordb/operations/episode_node_ops.py` | 修改（content_blocks 读写） |
| 1 | `graphiti_core/driver/neptune/operations/episode_node_ops.py` | 修改（content_blocks 读写） |
| 1 | `graphiti_core/utils/bulk_utils.py` | 修改（bulk 操作支持 content_blocks） |
| 2 | `graphiti_core/preprocessing/__init__.py` | 新增 |
| 2 | `graphiti_core/preprocessing/parser.py` | 新增（DocumentParser、Registry） |
| 2 | `graphiti_core/preprocessing/word_parser.py` | 新增（WordDocumentParser） |
| 2 | `graphiti_core/preprocessing/asset_storage.py` | 新增（S3 资产上传/删除） |
| 3 | `graphiti_core/embedder/client.py` | 修改（新增 create_image） |
| 3 | `graphiti_core/embedder/bedrock_nova.py` | 修改（实现 create_image） |
| 4 | `graphiti_core/graphiti.py` | 修改（新增 add_document_episode） |
| 4 | `graphiti_core/preprocessing/description.py` | 新增（图片描述生成） |
| 4 | `graphiti_core/prompts/models.py` | 修改（Message.content 类型扩展为 `str \| list[dict[str, Any]]`） |
| 4 | `graphiti_core/llm_client/client.py` | 修改（_clean_input 和 generate_response 支持多模态） |
| 4 | `graphiti_core/llm_client/bedrock_client.py` | 修改（多模态消息格式支持；LLM 超时配置化，见 13.16.10） |
| 4 | `graphiti_core/prompts/extract_nodes.py` | 修改（新增 extract_document 函数） |
| 4 | `graphiti_core/prompts/extract_edges.py` | 修改（新增 edge_document 函数） |
| 4 | `graphiti_core/utils/maintenance/node_operations.py` | 修改（_call_extraction_llm 新增 EpisodeType.document 分支） |
| 4 | `graphiti_core/utils/maintenance/edge_operations.py` | 修改（extract_edges 新增 EpisodeType.document 分支） |
| 4 | `graphiti_core/graphiti_types.py` | 修改（GraphitiClients 新增 image_embedding_map） |
| 4 | `graphiti_core/edges.py` | 修改（create_entity_edge_embeddings 支持 image_embedding_map） |
| 5 | `graphiti_core/graphiti.py` | 修改（search_ 新增 query_image） |
| 5 | `graphiti_core/search/search.py` | 修改（空查询短路逻辑兼容 query_vector，支持图片搜索） |

### 13.11 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Nova MME 图片 embedding 质量 | 跨模态检索精度不确定 | 先用小规模测试集验证；1024 维是官方推荐的平衡点 |
| 图片 base64 编码导致 prompt 过大 | LLM token 消耗增加，可能超出上下文窗口 | 图片压缩/缩放预处理；限制单个 episode 的图片数量 |
| Word 文档解析复杂性 | 嵌套表格、SmartArt 等复杂元素可能解析失败 | 首期只处理段落文本和嵌入图片；复杂元素降级为文本描述或跳过 |
| S3 多模态资产桶的生命周期管理 | `remove_episode` 需要同步清理 S3 资产 | 按 `{group_id}/{episode_uuid}/` 前缀批量删除 |
| content_blocks JSON 体积 | 大文档的 content_blocks 可能很大 | Neo4j 属性值有大小限制（默认 ~2MB）；超大文档需要拆分为多个 episode |
| LLM 视觉理解质量 | 图片中的实体/关系提取准确率不确定 | 先用 description 降级模式验证流程；视觉模式作为增强 |
| Message.content 类型扩展 | 从 `str` 扩展为 `str | list[dict]` 影响所有 LLM client | 渐进式改造：先在 BedrockLLMClient 中支持，其他 client 保持 str |
| 向后兼容 | 已有纯文本 episode 不受影响 | content_blocks 默认空列表；EpisodeType.document 是新增值 |

### 13.12 不在本次改造范围内

| 项目 | 原因 |
|------|------|
| 音频/视频处理 | 已在章节 14 中实现基础流程（ffmpeg + Transcribe → 文本导入）。完整多模态流程（ContentBlock + S3 资产）属于后续扩展 |
| PDF 文档解析 | 已有 PageIndex 项目处理 PDF，可后续集成 |
| 表格/图表结构化提取 | 需要专门的表格识别模型，非首期目标 |
| 多模态 BM25 全文检索 | BM25 仅适用于文本，图片内容通过 description 间接支持 |
| 图片 OCR | Kimi K2.5 视觉理解已覆盖 OCR 场景，无需独立 OCR 模块 |
| content_blocks 的增量更新 | 首期只支持整体替换，不支持单个 block 的增删改 |
| 多模态 episode 的 bulk 导入 | `add_episode_bulk()` 暂不支持多模态，需要时再扩展 |

> **⚠️ 更新**：原本列在此处的"单阶段多模态 LLM 提取"已纳入改造范围，见 13.16。

### 13.13 S3 Vectors 索引影响

多模态改造不新增 S3 Vectors 索引。现有 8 个索引保持不变：

| # | 索引名 | 多模态影响 |
|---|--------|-----------|
| 1 | `entity-name-embeddings` | 无影响（name 始终是文本） |
| 2 | `edge-fact-embeddings` | 无影响（fact 始终是 LLM 生成的文本） |
| 3 | `edge-source-embeddings` | 当 `source_excerpt` 引用图片（`[image:s3://...]`）时，存储的是图片的 Nova MME embedding 而非文本 embedding。文本和图片向量共存于同一索引，支持跨模态检索 |
| 4 | `community-name-embeddings` | 无影响 |
| 5 | `episode-narrative-embeddings` | 当 narrative excerpt 引用图片（`[image:s3://...]`）时，存储的是图片的 Nova MME embedding 而非文本 embedding。文本和图片向量共存于同一索引 |
| 6 | `describes-fact-embeddings` | 无影响 |
| 7 | `describes-excerpt-embeddings` | 当 `excerpt` 引用图片（`[image:s3://...]`）时，存储的是图片的 Nova MME embedding 而非文本 embedding。文本和图片向量共存于同一索引 |
| 8 | `episode-content-embeddings` | 多模态 episode 的 content 是文本化表示 |

**关键点**：所有向量索引中存储的仍然是 1024 维向量，无论来源是文本还是图片。Nova MME 保证了跨模态向量的语义空间一致性。

### 13.14 与现有改造的关系

| 现有改造 | 与多模态的关系 |
|---------|--------------|
| 章节 12（DescribesEdge + deep_search） | 多模态 episode 的 DescribesEdge 和 narrative_excerpts 机制完全复用，excerpt 内容可能包含多模态引用标记 |
| 12.9.4（Episode 重复导入检测） | 多模态 episode 的 content 是文本化表示，dedup 机制直接复用 |
| 12.9.5（术语统一） | 多模态改造基于重构后的 narrative_excerpts 术语 |
| 9.2（Episode Narratives 后续处理） | 多模态 narrative 可能包含多模态引用标记，后续处理需要感知 |
| 9.4（Prompt 模板） | 多模态场景可能需要专门的提取模板（如"图文文档"模板） |

### 13.15 多模态端到端测试（2026-03-05）

#### 13.15.1 测试环境与数据

- 测试方案：`examples/docx-manual/`（7 个脚本，见 README.md）
- 测试文档：`Global MAP 2.0 CCS Tagging操作指南.docx`（AWS MAP 2.0 CCS 打标签操作手册）
  - 6 个章节（Heading 1/2），9 张嵌入图片（操作截图），2 个表格
  - 图文混排：文字说明 + 操作截图交替出现
- 导入方式：`add_document_episode()`（完整多模态流程，按章节导入，每章节一个 episode）
- group_id: `docx-manual-test`

#### 13.15.2 导入结果

> **注**：以下为两阶段提取（图片→描述→文本提取）的基线数据。单阶段多模态提取的改进结果见 13.16.9。

| 指标 | 数值 |
|------|------|
| 总耗时 | 353s（8 sections） |
| Entities | 31 |
| Episodes | 8 |
| Edges | 33（33/33 有 source_excerpt） |
| DescribesEdges | 14 |
| Narrative Excerpts | 8 |
| ContentBlocks | 30（19 text + 9 image + 2 table） |
| 图片 S3 上传 | 9/9 均有 s3_uri（`graphiti-multimodal-assets-poc` bucket） |

**导入流程验证**：
- ✅ Word 文档解析：文本段落、嵌入图片、表格按顺序提取为 ContentBlock
- ✅ 图片 embedding 在 S3 上传前生成（`_raw_bytes` 上传后清除）
- ✅ 图片上传到 S3 多模态资产桶，s3_uri 正确填充
- ✅ Vision LLM 生成图片描述（中英文混合）
- ✅ content 文本表示包含 `[image:s3://...]` 引用
- ✅ image_embedding_map 正确传递，finally 中清除
- ✅ add_episode() UUID 兼容性修复生效（见 13.7.3）

#### 13.15.3 文本搜索结果

| 查询 | 模式 | 结果 |
|------|------|------|
| CCS 打标签 | 标准 | 返回高质量相关边 |
| Cost Explorer | 标准 | 命中 Cost Explorer 相关实体和操作步骤 |
| 如何查看标签覆盖率 | 深度 | 20 edges + 20 nodes + 5 narratives |
| MAP 2.0 | 标准 | 命中 MAP 2.0 项目相关实体 |

- 中文内容提取准确
- DescribesEdge 的 excerpt 正确包含 `[image:s3://...]` 引用
- fact 由 vision LLM 生成（如 "Screenshot shows clicking Cost Explorer, grouping by tags"）

#### 13.15.4 图片搜索结果与跨模态检索分析

**搜索目标**：Cost Explorer "按标签分组" 页面的局部截图（`search-target.png`）

**搜索方法**：`search_()` 传入 `query_image`，通过 `embedder.create_image()` 生成搜索向量，使用 vector-only SearchConfig（不含 BM25/fulltext）。

**各索引命中情况**：

| 索引 | Top-1 Score | 命中内容 | 分析 |
|------|-------------|---------|------|
| `describes-excerpt-embeddings` | 0.50 | 原图（section 5 Cost Explorer 截图） | ✅ 最高分，跨模态检索有效 |
| `episode-narrative-embeddings` | 0.45 | 图片类 narrative（含 `[image:s3://...]`） | ✅ 图片 narrative 明显高于文本 narrative（0.15） |
| `edge-source-embeddings` | 0.43 | Cost Explorer 相关边的 source_excerpt | ✅ 图片引用的边 |
| `edge-fact-embeddings` | 0.25 | Cost Explorer 相关 fact | 较低，fact 是文本，跨模态距离大 |
| `entity-name-embeddings` | 0.18 | Cost Explorer 实体 | 较低，name 是纯文本 |

**Deep Search 结果**：通过 `describes-excerpt-embeddings` 路径，返回 Cost Explorer 实体节点。这验证了 DescribesEdge 作为图片→实体桥梁的设计有效。

**跨模态 Cosine Similarity 特征**：

| 对比维度 | Score 范围 | 说明 |
|---------|-----------|------|
| 同模态（文本↔文本） | 0.6 ~ 0.9+ | 标准文本搜索的正常范围 |
| 跨模态（图片↔图片） | 0.4 ~ 0.5 | 局部截图 vs 完整截图，相似但非完全匹配 |
| 跨模态（图片↔文本） | 0.15 ~ 0.30 | 图片 query vs 文本 embedding，语义距离较大 |

**结论**：
1. 跨模态 cosine similarity 整体低于同模态，这是 Nova MME 模型的固有特性，不是 bug
2. 图片搜索时 `sim_min_score` 需要设为 0.0（或很低的值），否则会过滤掉有效结果
3. `describes-excerpt-embeddings` 是图片搜索的主要召回路径（通过 DescribesEdge 关联到实体）
4. Deep search 对图片搜索至关重要——标准搜索只查 edge-fact-embeddings（跨模态分数低），deep search 额外查 describes-excerpt-embeddings 和 edge-source-embeddings（图片向量，分数高）

#### 13.15.5 实现过程中发现的问题与修复

| # | 问题 | 修复 | 影响 |
|---|------|------|------|
| 1 | `add_episode()` 收到预生成 UUID 时抛出 `NodeNotFoundError` | `get_by_uuid` 包裹 try/except，fallback 创建新节点（见 13.7.3） | `add_document_episode` 委托流程 |
| 2 | `search()` 空文本查询短路返回，阻断图片搜索 | 条件改为 `query.strip() == '' and query_vector is None`（见 13.8.4） | 图片搜索 |
| 3 | `model_dump()` 序列化 ContentBlock 时 enum 值为 Python 对象而非字符串 | 改用 `model_dump(mode='json')` | content_blocks Neo4j 存储 |

#### 13.15.6 测试脚本体系

目录：`examples/docx-manual/`

| 脚本 | 功能 | 验证点 |
|------|------|--------|
| `1_clear.py` | 清空 Neo4j + 重建 8 个 S3 Vectors 索引 | 索引创建 |
| `2_ingest.py` | 解析 docx → ContentBlocks → 按章节导入 | 完整多模态导入流程 |
| `3_search.py` | 标准 + 深度文本搜索 | 文本检索质量 |
| `4_describes.py` | 按 Episode 查看 DescribesEdge | 图片→实体描述边 |
| `5_uncovered.py` | 查看 episode narratives | 纯叙事文本保留 |
| `6_verify_blocks.py` | content_blocks 序列化/反序列化 + s3_uri 验证 | 数据完整性 |
| `7_image_search.py` | 图片搜索（跨模态向量检索） | 跨模态检索有效性 |

通过 `run.sh` 执行，支持 `run.sh all`（全流程）和 `run.sh <step>`（单步）。

### 13.16 单阶段多模态 LLM 提取改造

#### 13.16.1 问题分析

当前 `add_document_episode()` 的处理流程（13.7.2）存在一个关键缺陷：LLM 提取阶段无法看到原始图片。

**根因**：步骤顺序导致 `_raw_bytes` 在 LLM 提取前被清除。

```
当前流程（有问题）：
  Step 2: 生成图片 embedding（使用 _raw_bytes）  ← _raw_bytes 可用
  Step 3: 上传 S3（清除 _raw_bytes，填充 s3_uri） ← _raw_bytes 被清除
  Step 4: 生成图片描述（需要 _raw_bytes）          ← 实际跳过（_raw_bytes 已为 None）
  Step 5: 构建 content 文本
  Step 7: 委托 add_episode() → extract_nodes/extract_edges
          ↓
          extract_document() / edge_document() 检查 block._raw_bytes
          → None → 走 description 降级路径（文本替代图片）
```

**具体表现**：

1. `asset_storage.upload_blocks()` 在上传后执行 `block._raw_bytes = None` 释放内存
2. `generate_image_descriptions()` 过滤条件 `b._raw_bytes is not None` → 找不到任何 block → 跳过
3. `extract_document()` 和 `edge_document()` 中的 `if block._raw_bytes:` 分支走不到 → 走 `elif block.description:` 降级路径
4. 测试脚本中手动预填了 `description`（如 `操作截图（文档第X个内容块）`），所以流程没报错，但 LLM 看到的是文本占位符而非真正的图片

**信息失真**：LLM 从 `[image: 操作截图（文档第X个内容块）]` 这样的文本中提取实体/边，而非直接理解图片内容。图片中的 UI 元素、按钮文字、数据图表等细节信息全部丢失。

#### 13.16.2 改造目标

将"两阶段"提取（图片→文字描述→文本提取）改为"单阶段"提取（图片+文本直接发给 LLM），让 vision LLM 在理解原始图片的基础上提取实体和边。

**前提条件**：`extract_document()` 和 `edge_document()` 的多模态 prompt 构建逻辑已经实现（13.6.2），支持 `_raw_bytes` → base64 → `image_url` 格式。只是当前运行时因为 `_raw_bytes` 已被清除而走不到该分支。

#### 13.16.3 方案：延迟清除 _raw_bytes（方案 A）

核心思路：S3 上传后不立即清除 `_raw_bytes`，保留到 LLM 提取完成后再统一清除。

**改造后流程**：

```
add_document_episode()（改造后）
  │
  ├─ Step 1: 文档解析
  │   → list[ContentBlock]（含 _raw_bytes）
  │
  ├─ Step 2: 生成图片 embedding（使用 _raw_bytes）
  │   embedder.create_image(block._raw_bytes) → block._embedding
  │
  ├─ Step 3: 上传 S3（填充 s3_uri，但不清除 _raw_bytes）  ← 关键变更
  │   upload_blocks(..., clear_raw_bytes=False)
  │
  ├─ Step 4: 生成图片描述（使用 _raw_bytes，此时仍可用）  ← 现在能正常执行
  │   generate_image_descriptions(llm_client, blocks)
  │   → 填充 block.description
  │
  ├─ Step 5: 构建 content 文本表示
  │   content = build_content_from_blocks(blocks)
  │
  ├─ Step 6: 设置 image_embedding_map
  │   build_image_embedding_map(blocks) → clients.image_embedding_map
  │
  ├─ Step 7: 委托 add_episode()
  │   content_blocks 中 _raw_bytes 仍在
  │   → extract_document() 构建 base64 image_url（视觉模式）  ← 现在走到正确分支
  │   → edge_document() 构建 base64 image_url（视觉模式）
  │
  └─ finally:
      ├─ 清除所有 block._raw_bytes（释放内存）  ← 新增
      └─ 清除 clients.image_embedding_map
```

**与旧流程的对比**：

| 步骤 | 旧流程 | 新流程 | 变化 |
|------|--------|--------|------|
| Step 3 S3 上传 | 清除 `_raw_bytes` | 保留 `_raw_bytes` | `upload_blocks` 新增 `clear_raw_bytes` 参数 |
| Step 4 图片描述 | 跳过（`_raw_bytes` 已清除） | 正常执行（`_raw_bytes` 可用） | 描述质量提升 |
| Step 7 LLM 提取 | 走 description 降级路径 | 走 base64 视觉模式 | 提取质量显著提升 |
| finally | 只清除 image_embedding_map | 额外清除所有 `_raw_bytes` | 确保内存释放 |

#### 13.16.4 涉及文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `preprocessing/asset_storage.py` | `upload_blocks()` 新增 `clear_raw_bytes: bool = True` 参数 | 默认行为不变（向后兼容），`add_document_episode` 传 `False` |
| `graphiti.py` | `add_document_episode()` Step 3 传 `clear_raw_bytes=False`；finally 中遍历 blocks 清除 `_raw_bytes` | 步骤顺序不变，只改清除时机 |
| `llm_client/bedrock_client.py` | `PROMPT_TIMEOUTS` 配置化，支持环境变量覆盖 | 多模态 prompt 需要更长超时，见 13.16.10 |
| `prompts/extract_nodes.py` | 无需改动 | `extract_document()` 已支持 `_raw_bytes` → base64 |
| `prompts/extract_edges.py` | 无需改动 | `edge_document()` 已支持 `_raw_bytes` → base64 |
| `utils/maintenance/node_operations.py` | 无需改动 | 调度逻辑不变 |
| `utils/maintenance/edge_operations.py` | 无需改动 | 调度逻辑不变 |
| `preprocessing/description.py` | 无需改动 | `_raw_bytes` 可用后自然能正常执行 |

**改动量极小**：核心改动只有 3 个文件（`asset_storage.py`、`graphiti.py` 和 `bedrock_client.py`），其余文件的多模态支持代码已经就绪。

#### 13.16.5 内存影响分析

延迟清除 `_raw_bytes` 意味着图片二进制数据在内存中停留更久（从 S3 上传到 LLM 提取完成）。

| 场景 | 图片数 | 估算内存占用 | 停留时间 | 评估 |
|------|--------|-------------|---------|------|
| docx-manual 测试文档 | 9 张 | ~1.5 MB | ~45s/章节 | 可忽略 |
| 典型操作手册（20 页） | ~15 张 | ~5 MB | ~60s | 可接受 |
| 大型技术文档（100 页） | ~50 张 | ~50 MB | ~300s | 需关注 |
| 图片密集文档（画册等） | 100+ 张 | 100+ MB | ~600s | 可能需要分批处理 |

**缓解措施**（后续优化，非本次范围）：
- 大文档按章节拆分为多个 episode，每个 episode 的图片数量可控
- 可选的 `clear_after_embedding` 模式：embedding 生成后立即清除 `_raw_bytes`，LLM 提取走 description 降级（回退到旧行为）
- 图片压缩/缩放预处理：在解析阶段限制图片分辨率

#### 13.16.6 LLM Token 消耗影响

图片以 base64 编码嵌入 prompt 会显著增加 token 消耗：

| 图片大小 | base64 大小 | 估算 token 数 | 说明 |
|---------|------------|-------------|------|
| 50 KB | ~67 KB | ~800 tokens | 小截图 |
| 150 KB | ~200 KB | ~2000 tokens | 典型操作截图 |
| 500 KB | ~667 KB | ~5000 tokens | 高清截图 |

以 docx-manual 测试文档为例（9 张图片，平均 ~100 KB）：
- 旧流程：每章节 LLM 输入约 2000-3000 tokens（纯文本）
- 新流程：每章节 LLM 输入约 5000-10000 tokens（文本 + 图片 base64）
- 增幅约 2-4 倍，但仍在 Kimi K2.5 的上下文窗口内（128K tokens）

**注意**：`extract_nodes` 和 `extract_edges` 都会发送图片，即每张图片的 base64 会被发送两次（一次节点提取，一次边提取）。这是必要的开销，因为两个提取阶段都需要理解图片内容。

#### 13.16.7 与现有设计的关系

| 现有设计 | 关系 |
|---------|------|
| 13.6.2 Prompt 改造 | `extract_document()` 和 `edge_document()` 的多模态 prompt 构建逻辑不变，只是从"设计但未生效"变为"实际生效" |
| 13.6.3 图片描述生成 | 仍然保留。description 用于 content 文本表示、搜索结果展示、不支持视觉的 LLM 降级。但不再是 LLM 提取的唯一信息来源 |
| 13.7.2 处理流程 | 步骤顺序不变，只改 Step 3 的 `_raw_bytes` 清除时机和 finally 块 |
| 13.15 E2E 测试 | 改造后需要重新运行 E2E 测试，对比提取质量（实体数、边数、fact 准确性）。已完成，见 13.16.9 |

#### 13.16.8 验证计划

1. **单元测试**：确认 `upload_blocks(clear_raw_bytes=False)` 后 `_raw_bytes` 仍在、`s3_uri` 已填充
2. **E2E 对比测试**：使用 docx-manual 测试文档，对比改造前后：
   - 提取的实体数量和名称
   - 提取的边数量和 fact 质量
   - DescribesEdge 的 fact 和 excerpt 内容
   - source_excerpt 中的图片引用格式
3. **内存监控**：观察导入过程中的内存峰值
4. **回归测试**：纯文本 episode（sanguo 测试）行为不变

#### 13.16.9 实施结果（2026-03-05）

**代码改动**：

| 文件 | 改动 |
|------|------|
| `preprocessing/asset_storage.py` | `upload_blocks()` 新增 `clear_raw_bytes: bool = True` 参数，默认行为不变 |
| `graphiti.py` | `add_document_episode()` Step 3 传 `clear_raw_bytes=False`；finally 中遍历 blocks 清除 `_raw_bytes` |
| `llm_client/bedrock_client.py` | `PROMPT_TIMEOUTS` 配置化：5 个环境变量覆盖超时值，见 13.16.10 |

**单元测试**：429 passed, 4 skipped

**Timeout 修复**：首次 E2E 运行时，section 5（4 text + 2 img, 28 nodes）在 `extract_edges.edge_document` 阶段因 15s 默认超时而失败。多模态 prompt 包含 base64 图片，输入 token 数显著增加，需要更长的处理时间。新增专用超时配置后问题解决。

**E2E 对比（docx-manual 测试文档）**：

| 指标 | 改造前（两阶段） | 改造后（单阶段） | 变化 |
|------|-----------------|-----------------|------|
| Entities | 31 | 68 | +119% |
| Edges | 33 | 63 | +91% |
| DescribesEdges | 14 | 11 | -3 |
| Narratives | 8 | 10 | +2 |
| Content blocks | 30 (19t/9i/2tbl) | 30 (19t/9i/2tbl) | 不变 |
| 总耗时 | ~353s | ~672s | +90% |

**图片信息提取分析**：

63 条边中 35 条（56%）的 `source_excerpt` 来自图片。主要贡献者：

| 图片来源 | 边数 | 内容 |
|---------|------|------|
| section 5 `block_004.png`（831KB 标签对照表） | ~18 | 迁移场景→目标服务映射、tag key 对应关系 |
| section 7 `block_005.png`（148KB Cost Explorer） | ~10 | tag 下拉列表中各 tag key 并列关系 |
| section 7 `block_007.png`（47KB 费用表） | 2 | 具体 server 的费用数据 |
| 其他截图（section 3/4/6） | ~5 | S3 打标签状态、Billing 页面、Tag Editor 搜索 |

改造前这些图片信息完全无法提取（LLM 看到的是 `[image: 操作截图]` 占位符）。

#### 13.16.10 LLM 超时配置化

多模态 prompt 因 base64 图片导致输入 token 数显著增加，原有的硬编码超时值（`extract_edges.edge: 60s`，其余 `15s`）不再适用。将超时配置从代码硬编码改为环境变量可配置。

**环境变量**（`.env` 中配置，代码中的值为 fallback 默认值）：

| 环境变量 | 默认值 | 对应 prompt | 说明 |
|---------|-------|------------|------|
| `LLM_TIMEOUT_DEFAULT` | 15s | 其他所有 prompt | dedupe、attributes、summaries 等短 prompt |
| `LLM_TIMEOUT_NODE_EXTRACT_TXT` | 15s | `extract_nodes.extract_text/message/json` | 文本节点提取，与默认一致但有独立旋钮 |
| `LLM_TIMEOUT_NODE_EXTRACT_MM` | 30s | `extract_nodes.extract_document` | 多模态节点提取：输入大（base64）但输出短 |
| `LLM_TIMEOUT_EDGE_EXTRACT_TXT` | 60s | `extract_edges.edge` | 文本边提取：输出大（多条边） |
| `LLM_TIMEOUT_EDGE_EXTRACT_MM` | 90s | `extract_edges.edge_document` | 多模态边提取：输入大 + 输出大 |

**设计原则**：
- 按 NODE/EDGE × TXT/MM 对称分类，命名直观
- 超时不宜过长——LLM 调用不稳定时，长超时只会让导入时间不可控，不如快速 fail + retry
- `dedupe_edges.resolve_edge` 等 prompt 不涉及多模态内容，走默认 15s

**实现**：`bedrock_client.py` 中 `_load_prompt_timeouts()` 在模块加载时读取环境变量，无效值 warning 并回退到默认值。

**单元测试**：429 passed, 4 skipped（含 env-var override 测试和 invalid value fallback 测试）


## 14. 音视频数据处理

### 14.1 背景与动机

章节 13 的多模态数据支持聚焦于图文混排文档（Word 内嵌图片）。但真实场景中还存在大量音视频数据：产品介绍视频、会议录音、培训课程等。这些数据中的语音内容同样承载着重要的知识，需要提取并导入知识图谱。

与图片不同，音视频数据的处理路径是：先转录为文字，再作为纯文本导入 Graphiti。不需要以原始音视频文件信息保存到图谱中（不涉及 ContentBlock、S3 多模态资产存储等），本质上是一个预处理步骤。

### 14.2 处理流程

```
视频/音频文件
    │
    ├─ ffmpeg: 提取音频 → WAV (16kHz mono, PCM)
    │   - 视频文件提取音频轨
    │   - 音频文件直接转码为 Transcribe 兼容格式
    │
    ├─ AWS Transcribe: 语音识别 → JSON
    │   - 支持中文 (zh-CN)、英文等多语言
    │   - 返回逐词时间戳和置信度
    │   - 自动标点断句
    │
    ├─ 文本分段: 按句号/问号等标点分句 → 合并为 ~120 字段落
    │   - 每个段落作为一个 episode 导入
    │   - 段落长度可控，避免过长或过短
    │
    └─ Graphiti add_episode(): 标准文本导入流程
        ├─ LLM 实体提取 (extract_nodes)
        ├─ LLM 关系提取 (extract_edges)
        ├─ 去重 + 矛盾检测
        ├─ Neo4j 写入
        └─ S3 Vectors 向量同步
```

### 14.3 技术选型

| 组件 | 选择 | 原因 |
|------|------|------|
| 音频提取 | ffmpeg | 通用、稳定，支持所有主流音视频格式 |
| 语音识别 | AWS Transcribe | 中文识别质量好，与现有 AWS 基础设施一致，Serverless 按量付费 |
| 音频格式 | WAV 16kHz mono PCM | Transcribe 推荐格式，16kHz 足够语音识别 |
| 文本导入 | EpisodeType.text | 转录文本作为纯文本导入，复用现有全部 ingest 逻辑 |

### 14.4 端到端验证（2026-03-10）

#### 14.4.1 测试数据

- 源文件：`全新夜航系统.mp4`（农业无人机夜航灯系统产品介绍视频）
- 时长：~116 秒，1920x1080，H.264 + AAC 音频
- 语言：中文
- 转录文本：553 字符，分为 4 个段落导入

#### 14.4.2 导入结果

| 指标 | 数值 |
|------|------|
| 总耗时 | 168.3s（4 段落） |
| Entities | 22 |
| Episodes | 4 |
| Edges | 14（14/14 有 source_excerpt） |
| DescribesEdges | 7 |
| Narrative excerpts | 3 |

#### 14.4.3 搜索验证

| 查询 | 模式 | 结果质量 |
|------|------|---------|
| 夜航灯有什么特点？ | 标准 | ✅ 命中散热、视角、防水、一体化等核心特性 |
| 散热系统是怎么设计的？ | 标准 | ✅ 命中高通量散热风扇、大散热片基座、抗高温防水镜片 |
| 防水等级是多少？ | 标准 | ✅ 命中 IP67 防尘防水相关边 |
| 视角有多宽？ | 标准 | ✅ 命中 130 度超宽视角、横纵向对比 |
| 一体化设计包含哪些部分？ | 标准 | ✅ 命中像素无真数字夜航灯、超稳定稳向云台、高清 FPV 摄像头、三位一体机组 |
| 夜航灯如何实现夜间仿地作业？ | 深度 | ✅ 14 edges + 20 nodes + 3 narratives，召回视觉系统融合、深度图一致性等关键信息 |

深度搜索相比标准搜索多召回了 4 条边和 20 个节点，narrative excerpts 补充了"白天与夜间深度图一致"、"IP67 防尘防水"、"夜航灯与机身融合"等未被结构化的信息。

#### 14.4.4 提取质量分析

22 个实体覆盖了视频中提到的主要概念：

- 核心产品：夜航灯、全新夜航系统
- 关键组件：高通量散热风扇、大散热片数字夜航登基座、LED 灯珠、抗高温防水镜片、超稳定稳向云台、高清 FPV 摄像头
- 技术特性：一百三十度超宽视角、IP 六七级防尘防水、整机一体化设计、夜间仿地功能
- 参照对象：农业无人机夜航灯、汽车大灯
- 应用场景：丘陵山地、复杂山地树木环境

7 条 DescribesEdge 补充了实体的属性描述（如"夜航灯的防水等级达到 IP67"、"LED 灯珠亮度提升百分之三十"），这些信息无法表示为两个命名实体之间的关系，但通过 DescribesEdge 关联到了对应实体。

### 14.5 测试脚本

目录：`examples/audio-extraction/`

| 脚本 | 功能 |
|------|------|
| `0_transcribe.py` | 提取音频 + AWS Transcribe 转录（转录结果已预生成，通常不需要重跑） |
| `1_clear.py` | 清空 Neo4j + 重建 S3 Vectors 索引 |
| `2_ingest.py` | 加载转录文本，分段导入 Graphiti |
| `3_search.py` | 标准搜索 + 深度搜索测试 |
| `4_describes.py` | 查看 DescribesEdge 详情 |
| `5_narratives.py` | 查看 episode narrative excerpts |
| `run.sh` | Shell 包装器，支持 `run.sh all` 全流程和 `run.sh <step>` 单步 |

### 14.6 与章节 13 多模态设计的关系

音视频处理是章节 13 多模态框架的一个简化应用场景：

| 维度 | 章节 13（图文混排） | 章节 14（音视频） |
|------|-------------------|------------------|
| 原始文件保存 | ✅ ContentBlock + S3 资产桶 | ❌ 不保存原始文件 |
| 导入方式 | `add_document_episode()` | `add_episode()`（纯文本） |
| EpisodeType | `document` | `text` |
| 预处理 | Word 解析 → ContentBlock | ffmpeg + Transcribe → 文本 |
| LLM 提取 | 多模态 prompt（图片 base64） | 标准文本 prompt |
| 向量索引 | 图片向量共存于文本索引 | 纯文本向量 |

后续如果需要保存原始音视频文件信息（如时间戳对齐、音频片段检索），可以扩展为使用 `ContentBlock(block_type=audio)` + `add_document_episode()` 的完整多模态流程。当前阶段以文字提取和知识图谱构建为主。
