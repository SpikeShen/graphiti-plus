# Graphiti 改造路线

> 基于 Zep 开源项目 Graphiti 的深度改造，目标是构建一个通用的 Agent 记忆层 Python 包。
> 本文档梳理改造的核心理念、技术方向和实施路线，供团队讨论。
>
> 创建日期：2026-03-06 | 最后更新：2026-03-08

---

## 一、项目背景

### 1.1 原项目概况

[Graphiti](https://github.com/getzep/graphiti) 是 Zep 开源的时序知识图谱框架，核心能力是将非结构化文本转化为带时间线的知识图谱（实体 + 关系），支持 edge invalidation（新信息推翻旧关系）。原项目关注度较高但近期开发活跃度不高。

### 1.2 已完成的改造（截至 2026-03-06）

我们在原项目基础上做了大量改造，已显著偏离上游，决定 fork 独立发展：

| 改造项 | 说明 |
|--------|------|
| S3 Vectors | 替代 Neo4j 向量索引，8 个独立索引（entity-name / edge-fact / edge-source / episode-content / episode-narrative / describes-fact / describes-excerpt / community-name） |
| 多模态 ContentBlock | 支持 text / image / table 三种内容块，完整的图文混排文档处理链路 |
| add_document_episode | 多模态文档导入流程：解析 → 图片 embedding → S3 上传 → Vision LLM 描述 → 构建 content → 委托 add_episode |
| Saga 节点 | 文档级顺序链：Saga → HAS_EPISODE → Episodes，相邻 episode 通过 NEXT_EPISODE 串联 |
| DescribesEdge | 实体级叙述归因，保留原始描述与实体的关联（含图片引用） |
| narrative_excerpts | 未被结构化提取的叙述片段，向量化后存入 S3 Vectors 可检索 |
| entity_types 验证 | 验证了 LLM 属性提取 + pipeline 程序化写入共存的可行性 |
| Bedrock 全栈 | Nova Embedding MME（含图片向量）、Claude LLM、Reranker |
| S3 多模态资产存储 | 图片等二进制资产持久化到 S3，ContentBlock 保留 s3_uri 引用 |
| S3 结构化日志 | LLM 调用日志按时间分区存储到 S3 |

### 1.3 改造动机：三个目标场景

**场景 A：个人知识管理**
- 开发工具（如 Kiro）的上下文窗口有限，LLM 短期内不会有质的飞跃
- 传统文件记录写入容易但检索困难，信息随时间"变味"（历史决策与当前状态冲突）
- 多模态信息（照片、截图）中的高价值逻辑关系无法被文件系统管理

**场景 B：团队协作记忆**
- 代码编写只占开发项目约 30% 时间，大量时间花在沟通、拉齐、决策确认上
- 这些沟通和决策本质是知识，但频繁被丢弃，需要反复向不同对象解释
- 时序知识图谱天然适合记录"谁在什么时候因为什么做了什么决定"

**场景 C：Agent 记忆层**
- 当前流行的 Agent 应用（OpenClaw、豆包等）记忆能力浅，缺乏深度
- 随着 Agent 大面积普及，记忆问题会被广泛关注
- 需要一个低集成成本的记忆层，通过简洁 API 接入各种 Agent 框架

三个场景对底层能力的要求高度重合：多模态 ingest、时序 edge invalidation、结构化搜索、属性扩展。差异主要在接入方式和上层抽象上。

---

## 二、核心设计原则

### 2.1 双层信息架构（核心差异化）

**问题：LLM 提取是有损压缩**

当前所有 Agent 记忆项目（Graphiti、MemU、Mem0 等）无一例外都基于 LLM 提取后的信息做整理。LLM 的提取过程本质是一个有损压缩，**损失函数是"当前上下文的相关性"而非"全局时间线的潜在价值"**。

典型的信息丢失场景：
- 会议中某人随口提到"上次跟客户吃饭时他提到要换供应商" — 当前议题无关，LLM 忽略，但三个月后可能是理解客户流失的关键线索
- 一张合照的背景中有某个地标 — LLM 描述了人物但忽略了地点，未来回忆"那次去XX的活动"时无法关联
- 技术讨论中被否决的方案细节 — LLM 只保留了最终决策，但否决原因在未来技术选型时有参考价值

这些信息在产生时刻价值不高，但在未来某个时间点或与过去某个节点产生关联时，价值可能极高。

**解决方案：结构层 + 原始层**

```
┌─────────────────────────────────────────────────┐
│  结构层（Structure Layer）— 高效检索和推理        │
│  LLM 提取的实体、关系、摘要、时序演化              │
│  EntityNode / EntityEdge / summary / attributes  │
├─────────────────────────────────────────────────┤
│  原始层（Raw Layer）— 回溯和跨时间关联            │
│  原文片段、图片、音频、完整上下文                   │
│  content_blocks / source_excerpt / s3_uri /      │
│  narrative_excerpts / DescribesEdge              │
└─────────────────────────────────────────────────┘
```

- 结构层负责"找到"：通过实体名、关系类型、时间线快速定位
- 原始层负责"找全"：定位后回溯完整上下文，发现 LLM 遗漏的关联
- 类比人类记忆：快速联想（结构层）→ 遇到矛盾时深挖细节（原始层）

已实现的原始层锚点：

| 锚点 | 位置 | 作用 |
|------|------|------|
| `content_blocks` | EpisodicNode | episode 的完整多模态内容（文本+图片+表格） |
| `source_excerpt` | EntityEdge | 产生该关系的原文片段（含图片引用） |
| `narrative_excerpts` | EpisodicNode + S3 Vectors | 未被结构化提取的叙述片段，向量化可检索 |
| `DescribesEdge` | Episode → Entity | 实体级叙述归因 |
| `s3_uri` | ContentBlock | 二进制资产的持久化引用 |
| `image_embedding` | S3 Vectors | 图片跨模态向量，支持以图搜图 |

### 2.2 Agent-Oriented API（导入与搜索对称）

导入和搜索都应设计为面向 Agent 的工具接口，而非面向终端用户的黑盒 API：

```
写入侧：Agent → pipeline steps → 提取/判断/存储
读取侧：Agent → search methods → 定位/回溯/深挖
```

Agent 根据场景自主选择调用方式、组合多次调用、沿关系链深入。这模拟了人类"遇到矛盾就深挖"的记忆检索模式。

### 2.3 Schema 约束（广度 vs 深度）

原项目通过 `entity_types` + `edge_types` + `edge_type_map` 三件套约束 LLM 提取行为：

- **entity_types**：收敛实体类型（不传则万物皆 Entity，传了则限定为 Person/City 等）
- **edge_types**：收敛关系类型（IS_PRESIDENT_OF / LOCATED_IN 等）
- **edge_type_map**：约束节点对→关系的合法映射（Person+City 只能有 LOCATED_IN）

不加约束时 LLM 会提取所有能识别的关系，数量多但价值密度低。加了 schema 约束后图谱紧凑、信息含量高，edge invalidation 判断也更准确。

这是一个深思熟虑的设计，不同场景需要不同的 schema profile（详见第四节）。

---

## 三、六大改造方向

### 3.1 Pipeline 改造（信息导入与更新）

**现状：** 三种 pipeline 全部写在 `graphiti.py` 单文件中（2000+ 行），与 LLM 紧耦合。

| Pipeline | 说明 | 问题 |
|----------|------|------|
| `add_episode()` | 逐条导入，6 阶段，支持 edge invalidation | 性能瓶颈（每条多轮 LLM + 向量搜索） |
| `add_document_episode()` | 多模态文档，7 步预处理后委托 add_episode | 与 add_episode 共享问题 |
| `add_episode_bulk()` | 批量导入，跳过 invalidation | 丢失时序演化（吕布效忠问题） |

**核心问题：** `add_episode` 内部直接调 LLM 做实体提取、边提取、去重判断、属性填充，调用方无法介入中间步骤。Agent 场景下，Agent 自己就是 LLM 驱动的，却要通过黑盒 pipeline 调另一个预配置 LLM，两个 LLM 之间没有协调。

**目标：双模式 Pipeline**

```
库模式：  调用方 → pipeline.run(content) → [内置 LLM 自动执行] → 存图谱
Agent 模式：Agent → pipeline steps → Agent 自己做提取/判断 → 存图谱
```

Pipeline 步骤拆分（每步可独立调用，也可一键执行）：

| 步骤 | 方法 | 库模式 | Agent 模式 |
|------|------|--------|-----------|
| 1. 预处理 | `prepare(content)` | 自动解析文档 | Agent 自定义预处理 |
| 2. 实体提取 | `extract_nodes(content)` | 内置 LLM | Agent 自己提取 |
| 3. 实体去重 | `resolve_nodes(nodes)` | 内置 LLM | Agent 参与决策 |
| 4. 关系提取 | `extract_edges(nodes, content)` | 内置 LLM | Agent 自己提取 |
| 5. 关系去重 + invalidation | `resolve_edges(edges)` | 内置 LLM | Agent 参与矛盾判断 |
| 6. 属性填充 | `hydrate(nodes, edges)` | 内置 LLM | Agent 填充 + pipeline 补充 |
| 7. 持久化 | `save(episode, nodes, edges)` | 纯存储 | 同左 |
| 8. 向量同步 | `sync_vectors(...)` | 自动 embedding | 同左 |

其他改进：
- 分段批量：按时间窗口分批 bulk，批次之间执行 invalidation pass
- 删除 pipeline：节点/边级联删除、向量索引清理、关联 episode 处理
- Saga 作为文档级导入的标准模式（已验证）

### 3.2 数据预处理（多源多模态）

**现状：** Word 文档已实现（docx-manual 端到端验证通过），ContentBlock 抽象支持 text/image/table。

**待探索格式：**

| 格式 | 方案 | 复杂度 |
|------|------|--------|
| PDF | PageIndex 项目已有基础，对接 ContentBlock | 中 |
| Excel/CSV | 结构化数据映射为 episode，行列关系建模 | 中 |
| 音视频 | Whisper/Transcribe 转写 + 说话人分离 | 高 |
| 网页/HTML | 正文提取、链接关系 | 低 |
| 代码仓库 | 文件结构、函数调用关系、注释文档 | 高 |

关键设计点：
- `DocumentParserRegistry` 已有扩展机制，各格式 parser 输出统一为 `list[ContentBlock]`
- 预处理质量直接影响 LLM 提取效果，需要针对不同格式调优 block 切分策略
- 当前开源项目（MemU、Mem0、Cognee、Letta）的多模态处理模式高度一致：图片/视频/音频 → VLM/STT 转文本 → 存文本 → 用文本检索。本项目已超越这一模式，通过 ContentBlock 保留原始资产 + Nova MME 生成跨模态向量，支持以图搜图等感知级检索（详见第六节 6.4 信息论分析）

### 3.3 Prompt 微调

**现状：** prompt_library 模块化管理，支持 custom_extraction_instructions 注入。

**问题：**
- 通用 prompt 在特定领域效果不稳定（entity_types 测试中 role 字段被填为 "assistant"）
- 中文场景下准确率需要评估
- 不同 LLM（Claude/Nova/Kimi）对同一 prompt 响应差异大

**设想：**
- 按场景建立 prompt profile，与 schema profile 配套
- Field description 需要更精确（如 role 应明确说"从文本中提取的职务头衔，不是 AI 角色"）
- 建立 prompt 效果评估基准
- 考虑 few-shot 示例注入机制

### 3.4 扩展属性 Attributes + Schema Profile

**attributes 的双重角色：**

EntityNode 和 EntityEdge 都有 `attributes: dict[str, Any]`，实际承载两种数据：

| 来源 | 示例 | 填充时机 |
|------|------|----------|
| LLM 提取 | role="项目经理", organization="云智科技" | Phase 4 hydrate，LLM 从文本推断 |
| Pipeline 写入 | photo_s3_uri, face_embedding, confidence | add_episode 返回后，调用方更新 |

两者共存于同一个 dict，互不干扰（已验证 round-trip）。

**待解决：**
- LLM 字段 vs pipeline 字段没有明确区分机制
- 多次导入时 attributes 是 update 语义，缺乏版本追踪
- 不同调用方可能传不同的 entity_types model，schema 不一致

**Schema Profile 预设：**

不同场景需要不同的 entity_types + edge_types + edge_type_map + prompt 组合：

```
profiles/
├── narrative.py      # 小说叙事：Character + Faction + Battle
│                     #   效忠、敌对、参战、继承
├── technical.py      # 技术文档：Service + Concept + Operation
│                     #   依赖、包含、触发、配置
├── meeting.py        # 会议纪要：Person + Task + Decision + Blocker
│                     #   负责、依赖、推翻、阻塞
├── personal.py       # 个人知识：Person + Event + Decision + Place
│                     #   参与、决定、影响、位于
└── multimedia.py     # 多模态：Person + Scene + Object
                      #   出现于、包含、关联
```

用户选择 profile 即自动带上 schema 约束 + prompt 微调，也可基于 profile 自定义扩展。

### 3.5 搜索逻辑（Agent-Oriented Search API）

**现状：** 原项目 16 种搜索配置，我们扩展了 edge-source-embeddings（原始信息向量检索）和图片搜索。

**问题：**
- 单次搜索只能触达局部子图，复杂问题需要多跳推理
- 搜索方法选择靠调用方硬编码，缺乏智能路由
- 关系链、时间线、社区结构等维度的搜索能力未充分暴露

**设想：** 将搜索方法封装为 Agent 可调用的工具集（MCP server 或 function calling）：

| 方法 | 说明 |
|------|------|
| `search_by_entity(name)` | 查实体及其所有关系 |
| `search_timeline(entity, time_range)` | 查实体在时间段内的关系演化 |
| `search_path(entity_a, entity_b)` | 查两个实体之间的关系路径 |
| `search_community(entity)` | 查实体所在社区及社区成员 |
| `search_saga(saga_name)` | 按 saga 顺序检索文档章节 |
| `search_by_image(image)` | 跨模态图片检索 |
| `search_similar_faces(embedding)` | 人脸向量检索（未来） |
| `search_contradictions(entity)` | 查实体相关的矛盾/推翻记录 |

Agent 根据问题自主选择方法、组合多次调用、沿关系链深入。搜索结果带元数据（来源 episode、置信度、时间戳），Agent 据此判断信息可靠性。

### 3.6 包化改造（Python Package）

**目标：** 将项目改造为可 pip 安装的通用 Python 包，降低集成门槛。

**当前代码的抽象化现状：**

已有的好设计：
- 4 层可插拔后端（graph / llm / embedder / cross_encoder）都有 ABC
- entity_types / edge_types 通过 Pydantic model 实现用户自定义 schema
- DocumentParserRegistry 支持按文件类型注册 parser
- group_id 实现了数据分区隔离

需要改造的部分：

| 层次 | 现状 | 包化需求 |
|------|------|----------|
| 安装 | 从源码 import | `pip install graphiti-memory` |
| 配置 | 硬编码 env vars + 构造函数 | Builder pattern 或 config object |
| vector_store | 只有 S3 Vectors，深度耦合 | 抽象 VectorStore ABC，支持多种实现 |
| 资产存储 | 硬编码 S3 bucket | 抽象 AssetStorage ABC，支持本地/云端 |
| 日志 | S3 logger 硬编码 | 可选的 structured logging |
| 搜索 API | 内部方法 | 面向 Agent 的工具接口 |
| graphiti.py | 2000+ 行，3 种 pipeline 混合 | 拆分为 pipeline 模块 |

**包化分层设计（草案）：**

```
graphiti-memory/
├── graphiti_memory/
│   ├── core/                    # 核心抽象（零外部依赖）
│   │   ├── types.py             # Node, Edge, Episode, Saga, ContentBlock
│   │   ├── pipeline.py          # IngestPipeline ABC
│   │   ├── search.py            # SearchEngine ABC + SearchConfig
│   │   └── config.py            # MemoryConfig
│   │
│   ├── backends/                # 可插拔后端（extras 安装）
│   │   ├── graph/               # Neo4j / Neptune / Kuzu
│   │   ├── llm/                 # OpenAI / Anthropic / Bedrock / Gemini
│   │   ├── embedder/            # OpenAI / Nova / Voyage
│   │   ├── vector/              # S3 Vectors / Qdrant / FAISS（新增 ABC）
│   │   └── storage/             # S3 / 本地文件系统（新增 ABC）
│   │
│   ├── pipelines/               # Pipeline 实现（从 graphiti.py 拆出）
│   │   ├── episode.py           # add_episode
│   │   ├── document.py          # add_document_episode
│   │   ├── bulk.py              # add_episode_bulk
│   │   └── delete.py            # 删除流程（新增）
│   │
│   ├── prompts/                 # Prompt 模板 + 场景 profile
│   ├── search/                  # 搜索方法库（standard / deep / timeline / path）
│   ├── integrations/            # Agent 集成（MCP / OpenAI tools / LangChain）
│   └── presets/                 # 开箱即用配置
│       ├── local.py             # Kuzu + FAISS + Ollama（零外部依赖）
│       ├── aws.py               # Neo4j/Neptune + S3 Vectors + Bedrock
│       └── openai.py            # Neo4j + OpenAI
│
├── pyproject.toml               # extras: [local], [aws], [openai], [mcp], [all]
└── README.md
```

**安装体验目标：**

```bash
pip install graphiti-memory[local]    # 本地开发，零外部依赖
pip install graphiti-memory[aws]      # AWS 全栈
pip install graphiti-memory[openai]   # OpenAI 全栈
pip install graphiti-memory[mcp]      # MCP server 集成
```

**API 体验目标：**

```python
from graphiti_memory import Memory
from graphiti_memory.presets import local_preset

# 一行初始化
memory = Memory(local_preset())

# 写入（库模式）
await memory.add("张三今天升任CTO，接替李四")

# 搜索
results = await memory.search("谁是CTO")

# Agent 工具集
tools = memory.as_openai_tools()
```

---

## 四、关键设计决策（待讨论）

| # | 决策项 | 现状/倾向 | 备注 |
|---|--------|----------|------|
| 1 | 包名 | `graphiti-memory` 待定 | 需考虑与原项目 `graphiti-core` 的区分 |
| 2 | 与上游关系 | **已决定 fork 独立发展** | 改造已显著偏离上游 |
| 3 | 零依赖本地部署 | **确定支持** | Kuzu + FAISS，降低试用门槛 |
| 4 | VectorStore 抽象层 | **确定要做** | 当前只有 S3 Vectors，需补 ABC |
| 5 | Prompt 自定义 | 待讨论 | 完全覆盖 vs 仅 custom_instructions 注入 |
| 6 | 多租户 / 数据隔离 | **多维元数据过滤（已确认可行）** | group_id 升级为多维过滤体系，S3 Vectors / OpenSearch / Qdrant 均原生支持（详见 6.3） |
| 7 | 核心差异化 | **双层信息架构** | 区别于 Mem0/MemU 的关键特性 |
| 8 | Pipeline LLM 解耦 | 双模式（库+Agent） | Agent 模式下 pipeline 不绑定 LLM |

---

## 五、优先级参考

| 方向 | 紧迫度 | 复杂度 | 说明 |
|------|--------|--------|------|
| 3.4 Attributes + Schema Profile | 高 | 中 | entity_types 已验证可行，需完善区分机制 |
| 3.3 Prompt 微调 | 高 | 中 | 直接影响提取质量，可逐步迭代 |
| 3.1 Pipeline 改造 | 中 | 高 | graphiti.py 拆分 + 双模式设计 |
| 3.5 Agent Search API | 中 | 高 | 需先稳定搜索基础能力 |
| 3.6 包化改造 | 中 | 高 | 依赖 3.1 的 pipeline 拆分 |
| 3.2 多源预处理 | 低 | 中 | ContentBlock 框架已就绪，按需扩展 |

建议路径：3.4 → 3.3 → 3.1 → 3.5 → 3.6 → 3.2

---

## 六、参考资料

### 6.1 项目文档

- 原项目：[Graphiti by Zep](https://github.com/getzep/graphiti)
- 架构总览：[docs/Graphiti_架构总览.md](./Graphiti_架构总览.md)
- S3 Vectors 改造方案：[docs/Graphiti_S3Vectors_改造方案.md](./Graphiti_S3Vectors_改造方案.md)

### 6.2 同类项目对比

| 项目 | 核心抽象 | 记忆写入方式 | 时序能力 | 原始信息保留 | 适用场景 |
|------|---------|-------------|---------|-------------|---------|
| **Graphiti**（本项目） | Node + Edge 知识图谱 | Pipeline 自动提取 | 原生 edge invalidation | ✅ 双层架构 | 通用记忆层 |
| **Letta** | Block 文本块 | Agent 主动调工具编辑 | 无（Agent 自己写日期） | ❌ Core Memory 是压缩后的 | Agent 对话记忆 |
| **MemOS** | MemCube（三类记忆容器） | LLM 提取 + 图谱存储 | 无 | 部分（SourceMessage 溯源） | 通用记忆 OS |
| **AgentCore Memory** | Event + Memory Record | 异步 LLM 提取 + 合并 | 无（但保留时间戳，合并时优先新信息） | ❌ 短期原始事件有 TTL，长期只存提取后的 | AWS 托管 Agent 记忆 |
| **Mem0** | Memory 条目 | LLM 提取后存储 | 有限 | ❌ 仅提取后信息 | Agent 个性化记忆 |
| **MemU** | Resource→MemoryItem→Category 三层模型 | LLM 提取 6 种类型（profile/event/knowledge/behavior/skill/tool） | 有限 | ❌ Resource 记录 local_path 但检索只用文本 | Agent 记忆 |
| **OpenClaw** | 向量 passage | 向量化存储 + 检索 | 无 | 部分（保留原文片段） | Agent 对话记忆 |
| **LangMem** | Memory 条目 | LLM 提取 | 无 | ❌ | LangChain 生态记忆 |

**Letta 记忆机制要点：**

Letta 采用三层记忆模型，设计理念是"让 Agent 自己管理记忆"：

1. **Core Memory**（核心记忆）：多个 Block 文本块（如 persona、human），始终注入 system prompt，Agent 通过工具函数主动编辑（append/replace/rethink）。每个 block 有字符上限，Agent 自己管理信息密度。
2. **Archival Memory**（归档记忆）：长期向量存储，Agent 主动调用 `archival_memory_insert/search` 存取，支持标签和语义搜索，底层用 pgvector 或 Turbopuffer。
3. **Recall Memory**（回忆记忆）：对话历史的混合搜索（文本 + 语义），通过 `conversation_search` 检索。

Letta 最有特色的设计是 **Sleeptime Agent**：一个独立的后台 Agent，在主 Agent 回复用户后异步运行，审阅对话记录并更新共享的 Core Memory blocks。这将记忆整理从实时对话中解耦，类似我们讨论的 Agent 模式 pipeline。但本质上 Letta 的记忆仍是"LLM 压缩后的文本块"，没有解决信息丢失问题。

**MemU 记忆机制要点：**

MemU（v1.4.0）面向 24/7 常驻 Agent，核心是三层数据模型：`Resource`（原始数据）→ `MemoryItem`（提取的事实，6 种类型：profile/event/knowledge/behavior/skill/tool）→ `MemoryCategory`（自动归类的主题）。检索通过三层向量匹配（Category → Item → Resource）逐级定位。

MemU 的多模态处理是典型的"全部转文本"策略：图片走 Vision API 生成描述，视频抽中间帧当图片处理（只分析一帧，信息损失大），音频走 STT 转文字。预处理后统一变成 `[{text, caption}]`，后续记忆提取只处理文本。检索入口也只接受文本 query，不支持多模态 query。

MemU 的工作流引擎（声明式 step-based pipeline，支持运行时插入/替换/删除步骤及拦截器钩子）设计值得参考，但其记忆存储层没有为原始多模态数据留位置，向量空间是单一的文本空间（非 CLIP 跨模态模型）。

**各项目共同局限：** 无一例外都基于 LLM 提取/压缩后的信息做整理，原始信息在提取过程中丢失。多模态支持模式高度一致：图片/视频/音频 → VLM/STT 转文本 → 存文本 → 用文本检索。这正是本项目双层信息架构（结构层 + 原始层）要解决的核心问题。

**MemOS 记忆机制要点：**

MemOS（MemTensor 团队，arXiv:2507.03724）自称"Memory Operating System"，是目前看到的架构最复杂的记忆项目。核心概念是 **MemCube** — 一个可组合的记忆容器，内部封装三类记忆：

1. **Textual Memory**（文本记忆）：最核心的部分，有多种实现：
   - `GeneralTextMemory`：简单的向量存储，LLM 提取 key-value 对后存入 Qdrant
   - `TreeTextMemory`：图谱结构记忆，用 Neo4j 存储节点和边，支持 WorkingMemory / LongTermMemory / UserMemory 三级分层，有 BM25 + 向量 + 图遍历的混合搜索
   - `PreferenceTextMemory`：用户偏好记忆，区分显式/隐式偏好
   - `NaiveTextMemory`：最简实现，直接存原文

2. **Activation Memory**（激活记忆）：KV Cache 级别的记忆，直接操作 LLM 的 KV Cache 做拼接/复用。这是模型层面的记忆，其他项目都没有涉及。目前支持 vLLM KV Cache。

3. **Parametric Memory**（参数记忆）：LoRA 级别的记忆，通过微调模型参数来"记住"信息。目前是 placeholder，尚未实现。

MemOS 的几个值得关注的设计：

- **MemScheduler**：异步记忆处理调度器，基于 Redis Streams 实现任务队列，支持优先级、自动恢复、配额调度。类似 Letta 的 Sleeptime Agent，但更偏基础设施层面。
- **Memory Feedback**：自然语言反馈修正记忆，用户可以说"这个记忆不对，应该是..."来纠正。内部有 conflict/duplicate/extract/unrelated 四种更新类型判断。
- **SourceMessage 溯源**：每条记忆保留 `SourceMessage`（来源消息），记录 type/role/content/doc_path 等，可以追溯到原始对话或文档。这是一种轻量级的原始信息保留，但只保留了来源引用，不保留完整上下文。
- **ArchivedTextualMemory**：记忆更新时保留历史版本（version + history 字段），类似简化版的 edge invalidation，但没有时间线语义。
- **Multi-Cube 组合**：多个 MemCube 可以组合、共享、隔离，支持跨用户/跨项目的记忆管理。
- **TreeTextMemory 的图谱搜索**：搜索流程是 TaskGoalParser → MemoryPathResolver → GraphMemoryRetriever → MemoryReranker → MemoryReasoner，有 fast/fine 两种模式。

MemOS vs Graphiti 的关键差异：MemOS 的图谱是"记忆条目之间的关系"（SUMMARY/MATERIAL/FOLLOWING/PRECEDING），不是"实体之间的语义关系"（Person→WORKS_AT→Company）。MemOS 没有时序演化能力（edge invalidation）。MemOS 的 Activation Memory（KV Cache）和 Parametric Memory（LoRA）是独特的方向，但跟我们的改造目标不在同一层面。

**定位差异：Agent 时代的"内存" vs "数据库"**

用一个比喻来理解两者的本质区别：MemOS 要解决的是 Agent 时代的**内存管理**问题 — Agent 在推理过程中需要什么信息、KV Cache 怎么复用、上下文窗口怎么填充，核心是运行时状态调度；Graphiti 改造要解决的是 Agent 时代的**数据库**问题 — 信息怎么结构化存储、怎么跨时间演化、怎么在未来被精确检索，核心是持久化知识组织。

两者的关系是互补的：Agent 的"内存"需要从"数据库"里加载数据。MemOS 的 TreeTextMemory 搜索结果最终要填进上下文窗口，而这些搜索结果的质量取决于底层知识图谱的组织质量 — 这正是 Graphiti 改造要做的事。也正因为定位不同，MemOS 会自然地向更深层次的 LLM 上下文管理甚至模型参数调优方向探索（Activation Memory、Parametric Memory），而我们的方向是信息的持久化组织、时序演化和多模态原始信息保留。

**AWS Bedrock AgentCore Memory 要点：**

AgentCore Memory 是 AWS 在 2025 年 Summit NYC 发布的全托管 Agent 记忆服务，属于 Bedrock AgentCore 产品线。作为闭源商业产品，它的设计选择代表了 AWS 对 Agent 记忆问题的工程判断。

核心架构是 Short-Term Memory + Long-Term Memory 双层：

1. **Short-Term Memory**（短期记忆）：同步存储原始对话事件（Event），按 memoryId / actorId / sessionId 三级层次组织。事件是不可变的，有 TTL（最长 365 天）。支持两种事件类型：Conversational（对话消息）和 Blob（二进制数据，用于 checkpoint/状态快照）。
2. **Long-Term Memory**（长期记忆）：异步从短期事件中提取洞察，存入向量存储。提取过程由 Memory Strategy 驱动，内置三种策略：
   - **Semantic Strategy**：提取事实和知识（"客户公司有 500 名员工，分布在西雅图、奥斯汀和波士顿"）
   - **User Preference Strategy**：提取用户偏好（"偏好用 Python 开发"），带 categories 和 context
   - **Summary Strategy**：生成对话摘要，按 session 范围，结构化 XML 格式（`<topic>...</topic>`）

**提取 + 合并 Pipeline：**

这是 AgentCore Memory 最核心的技术点，跟 Graphiti 的 pipeline 有直接可比性：

1. **Extraction**（提取）：事件存入短期记忆后，异步触发 LLM 提取。每个 strategy 独立处理，可并行。
2. **Consolidation**（合并）：对每条新提取的记忆，语义搜索已有记忆中最相似的 top-k 条，然后 LLM 判断：
   - ADD：新信息，直接添加
   - UPDATE：补充或更新已有记忆
   - NO-OP：冗余信息，跳过
   - 矛盾处理：优先新信息，旧记忆标记为 INVALID（不删除，保留审计轨迹）

这个 consolidation 流程跟 Graphiti 的 edge invalidation 思路相似 — 都是"新信息推翻旧信息时标记旧的为无效"。但 AgentCore 是在记忆条目级别做合并，Graphiti 是在实体关系级别做 invalidation，粒度不同。

**其他值得关注的设计：**

- **Namespace**：层级化命名空间（如 `/org_id/user_id/preferences`），支持多租户隔离和跨 Agent 共享。类似文件系统路径，检索时可按前缀匹配。
- **Branching**：从对话历史的某个点创建分支，支持消息编辑、what-if 场景探索。
- **Checkpointing**：保存对话状态快照，支持跨 session 恢复。
- **Custom Strategy**：支持自定义 LLM 和 prompt 覆盖提取/合并逻辑，也支持 Self-Managed Strategy（完全自定义处理，只用 AgentCore 做存储和检索）。
- **PII 过滤**：所有内置策略默认忽略个人身份信息。

**性能数据（官方 benchmark）：**

| 策略 | 数据集 | 正确率 | 压缩率 |
|------|--------|--------|--------|
| RAG baseline（完整对话历史） | LoCoMo | 77.73% | 0% |
| Semantic Memory | LoCoMo | 70.58% | 89% |
| RAG baseline | LongMemEval-S | 75.2% | 0% |
| Semantic Memory | LongMemEval-S | 73.60% | 94% |
| Preference Memory | PrefEval | 79% | 68% |
| Summarization | PolyBench-QA | 83.02% | 95% |

关键 trade-off：事实类任务正确率略低于 RAG baseline（因为压缩丢信息），但压缩率 89-95%，大幅降低 token 消耗。偏好类任务反而优于 baseline（提取后的洞察比原始对话更有用）。提取+合并延迟 20-40 秒，语义检索约 200ms。

**AgentCore Memory vs Graphiti 的关键差异：**

- AgentCore 的记忆是扁平的"条目"（fact/preference/summary），没有实体-关系图谱结构。检索只有语义搜索，没有图遍历。
- AgentCore 的 consolidation 是条目级合并（相似记忆合并为一条），Graphiti 是关系级 invalidation（同一实体对的旧关系被新关系推翻）。后者保留了更丰富的时序演化信息。
- AgentCore 的短期事件有 TTL（最长 365 天），过期后原始对话丢失。长期记忆只保留提取后的洞察，不保留原始上下文。这正是我们双层架构要解决的问题。
- AgentCore 作为托管服务，开发者无需管理基础设施，但也意味着对记忆处理逻辑的控制有限（虽然 Custom Strategy 提供了一定灵活性）。
- AgentCore 的 Namespace + Branching + Checkpointing 设计面向企业级多租户场景，这些运维层面的能力是我们包化改造时可以参考的。

### 6.3 多维元数据过滤与数据隔离设计

#### 问题：group_id 的局限性

当前所有 8 个向量索引的过滤维度只有 `group_id`（edge 索引额外有 `source_node_uuid` / `target_node_uuid`）。在包化后多场景共用存储介质的情况下，单一 `group_id` 无法满足需求：

- 同一个 group 内可能有不同类型的实体（Person vs Service），搜索时需要按类型缩小范围
- 不同场景的 schema profile 不同，需要按场景隔离或交叉查询
- Agent 在搜索时应能根据当前任务上下文动态选择过滤维度，而不是只能按 group 过滤

#### 设计思路：语义软隔离（Semantic Soft Isolation）

区别于 AgentCore 的硬命名空间隔离（`/org_id/user_id/preferences`），我们的思路是**灵活的自定义元数据维度**，与 entity_types / edge_types 的设计理念一致：

```
硬隔离（AgentCore 风格）：  /namespace/path → 物理分区，跨分区查询需显式指定
软隔离（本项目方向）：      metadata dimensions → 逻辑标签，查询时按需组合过滤
```

核心设计：
1. **用户可定义维度**：类似 entity_types 的自定义 schema，用户在场景配置中声明需要哪些过滤维度（如 `entity_type`、`domain`、`source_type`、`project_id`）
2. **与 LLM prompt 联动**：Pipeline 提取时，LLM 根据 schema profile 自动填充维度值（如识别出 entity_type="Person"），也支持调用方程序化写入
3. **Agent 搜索时动态过滤**：Agent 根据当前任务上下文选择过滤条件，如"只搜索 Person 类型的实体"或"只搜索 project_id=X 的关系"

```python
# 写入时：metadata 携带多维标签
upsert_entity_vector(
    uuid="...",
    embedding=[...],
    metadata={
        "group_id": "team-alpha",
        "entity_type": "Person",        # 来自 LLM 提取
        "domain": "engineering",         # 来自 schema profile
        "project_id": "proj-123",        # 来自调用方
    }
)

# 搜索时：Agent 按需组合过滤
results = query_entity_vectors(
    query_vector=[...],
    metadata_filter={
        "$and": [
            {"group_id": {"$eq": "team-alpha"}},
            {"entity_type": {"$in": ["Person", "Team"]}},
            {"project_id": {"$eq": "proj-123"}},
        ]
    }
)
```

#### 向量引擎元数据过滤能力矩阵

| 引擎 | 原生元数据过滤 | 过滤方式 | 支持的操作符 | 适用场景 |
|------|--------------|---------|-------------|---------|
| **S3 Vectors** | ✅ 原生支持 | Pre-filtering（查询时过滤） | `$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$exists/$and/$or` | 当前已用，扩展成本最低 |
| **OpenSearch + FAISS** | ✅ v2.9+ 支持 | Efficient pre-filtering（Lucene 过滤 + FAISS ANN） | 完整 OpenSearch query DSL | 需要全文搜索 + 向量混合的场景 |
| **Qdrant** | ✅ 原生支持 | Payload filtering（索引化过滤） | `match/range/geo/values_count/is_empty/has_id` 等 | 自建部署，过滤能力最丰富 |
| **FAISS（纯库）** | ❌ 不支持 | 只能 post-filtering（先 ANN 再过滤） | 无 | 仅适合单一向量空间，不适合多维过滤 |
| **pgvector** | ✅ 通过 SQL WHERE | Pre-filtering（PostgreSQL 原生） | 完整 SQL 表达式 | 已有 PostgreSQL 基础设施的场景 |

关键结论：
- **S3 Vectors 完全可以支持**：当前 `upsert_vectors` 已接受任意 metadata dict，`query_vectors` 已支持完整的 filter expression 透传。扩展自定义维度只需在 upsert 时多写几个字段，查询时构造对应的 filter 即可，代码改动量极小。
- **纯 FAISS 不适合**：FAISS 只做 ANN 检索，没有元数据概念。要实现过滤只能先取 top-k*N 再在应用层过滤，效率低且结果不稳定。但 OpenSearch 封装的 FAISS 引擎（v2.9+）通过 Lucene 索引做 pre-filtering 解决了这个问题。
- **包化时的 VectorStore ABC 设计**：抽象层需要定义 metadata 写入和过滤的标准接口，各实现按自身能力适配。对于不支持原生过滤的引擎（如纯 FAISS），可以降级为 post-filtering 并在文档中标注性能影响。

#### 实施路径

1. **Phase 1（当前可做）**：在现有 S3 Vectors 的 upsert 方法中扩展 metadata 字段，将 `entity_type`、`edge_type` 等已有信息写入向量元数据。搜索方法增加可选的 `metadata_filter` 参数透传。
2. **Phase 2（Schema Profile 配套）**：Schema Profile 定义时声明该 profile 需要的过滤维度，Pipeline 提取时自动填充。
3. **Phase 3（Agent Search 配套）**：Agent Search API 暴露过滤维度信息，Agent 可以查询"当前有哪些维度和值"，然后构造精确的过滤条件。

这个设计与决策表中 #6（多租户隔离）直接相关，将 group_id 从唯一隔离手段升级为多维过滤体系中的一个维度。

### 6.4 前沿学术研究

以下两篇论文在多模态记忆架构方向做了前沿探索，虽然场景偏向视频理解，但其记忆组织思路对本项目有直接参考价值。

#### M3-Agent（字节跳动 Seed，2025，ICLR 2026）

论文：[Seeing, Listening, Remembering, and Reasoning](https://arxiv.org/abs/2508.09736) | 代码：[github.com/ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent)

模拟人类记忆形成过程，两个并行流程：
- **记忆化（Memorization）**：持续处理实时视频流和音频流，在线构建长期记忆
- **控制（Control）**：收到指令后多轮迭代推理，从长期记忆中检索信息完成任务

记忆组织为以实体为中心的多模态图谱（multimodal graph）：
- **情景记忆（Episodic）**：具体事件，谁在什么时候做了什么
- **语义记忆（Semantic）**：从多次经历中提炼的通用知识
- 节点包含多种模态：text / image（base64 人脸）/ audio（声纹片段），每个节点有 embedding + weight（置信度）
- 边表示逻辑关系，最核心的是"属于同一个实体"（如人脸节点和声纹节点被识别为同一人后连边）
- 冲突解决通过权重投票：每次正确关联被确认，对应边 weight 增加，高权重自然胜出

检索支持跨模态：`search_node` 接受文本/图片/音频 query，返回 top-k 最相关节点。

**与本项目的关联：** M3-Agent 的"以实体为中心的多模态图谱"与 Graphiti 的 EntityNode 设计高度吻合。其节点直接保存原始模态内容（人脸图片、声纹音频）的做法，验证了我们双层架构中"原始层保留多模态资产"的方向。权重投票的冲突解决机制可作为 edge invalidation 的补充参考。

**工程局限：** 实际实现非常朴素（Python 自定义数据结构序列化为 pickle，向量检索是内存暴力扫描），论文实验规模（~26,943 片段、几万节点）下可以跑，但真实 24/7 场景（百万级节点）完全撑不住。需要 ANN 索引替代线性扫描、图数据库替代 pickle、原始多模态内容与 embedding 分离存储（对象存储 + 向量库）。

#### WorldMM（KAIST + NTU，2025）

论文：[Dynamic Multimodal Memory Agent for Long Video Reasoning](https://arxiv.org/abs/2512.02425) | 项目页：[worldmm.github.io](https://worldmm.github.io/)

解决超长视频（几小时甚至一整天）的理解问题，构建三种互补记忆：
- **情景记忆（Episodic）**：多尺度文本事件图谱，从细粒度到粗粒度的层级结构
- **语义记忆（Semantic）**：持续更新的知识图谱，积累跨事件的高层关系和习惯模式
- **视觉记忆（Visual）**：直接保存视频帧的特征 embedding 和原始帧（关键区别）

自适应检索 Agent 根据问题类型动态决定从哪种记忆取信息，在五个长视频问答 benchmark 上比之前 SOTA 平均高 8.4%。

**与本项目的关联：** WorldMM 的三种记忆与我们的双层架构有对应关系——情景记忆 + 语义记忆 ≈ 结构层，视觉记忆 ≈ 原始层。其"自适应检索 Agent 根据问题类型选择记忆源"的设计，与我们 Agent-Oriented Search API 的理念一致。

#### 两篇论文的核心启示

| 维度 | M3-Agent | WorldMM | 本项目方向 |
|------|----------|---------|-----------|
| 记忆结构 | 以实体为中心的多模态图谱 | 三种互补记忆（情景+语义+视觉） | 双层架构（结构层+原始层） |
| 原始模态保留 | ✅ 节点直接存 base64 | ✅ 保存原始帧 + embedding | ✅ S3 资产 + 跨模态向量 |
| 冲突解决 | 权重投票 | 知识图谱持续更新 | edge invalidation + 时序演化 |
| 检索方式 | 跨模态 search_node | 自适应多源检索 | Agent-Oriented 多方法组合 |
| 工程成熟度 | 低（pickle + 暴力扫描） | 低（学术原型） | 中（Neo4j + S3 Vectors + Bedrock） |

### 6.5 多模态记忆的信息论视角

基于上述项目和论文的分析，关于"文本记忆是否足够"有一个信息论层面的判断：

**文本的表现力与经济性悖论：** 人类的文字表现力理论上可以超越事实本身，一个好的描述可能比照片还传神。问题不在于"文字能不能表达"，而在于经济性：
- 一张图片的 CLIP embedding 是 512/768 个浮点数，几 KB
- 要用文字达到同等信息密度，可能需要几千字描述（颜色色调、饱和度、材质纹理、光影方向、空间比例……）
- 存储文字 → 对文字做 embedding → 用 embedding 检索，绕了一大圈最后还是向量检索，中间多了"视觉→文字→向量"的转换，每步都有信息损失和计算开销
- 直接"视觉→向量"一步到位，又快又准

**本质上是信息论的问题：** 视觉信息天然是高维连续的，用离散文字编码它，要么丢信息，要么编码成本远超原始表示。向量是这类信息的原生表达形式，文字不是。

**本项目的分层策略：**
- **主路径（覆盖 90%+ 场景）**：VLM 提取文本事实 → 结构层存储 + 文本检索，随模型进化覆盖面持续扩大
- **增强层（覆盖剩余感知级场景）**：保留原始模态 embedding（已实现 Nova MME 图片向量），用于风格相似度、人脸匹配等文本无法经济地表达的场景

当前整个领域处于"思路已验证，工程和模型能力还没完全跟上"的阶段。论文（M3-Agent、WorldMM）在提前探路，告诉后来者哪条路是通的。本项目的工程基础（Neo4j + S3 Vectors + Bedrock MME + ContentBlock 多模态抽象）已经具备了落地条件。
