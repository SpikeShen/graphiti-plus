# Graphiti 调用日志与 Athena 查询指南

## Athena 数据库

数据库名：`graphiti`

查询结果输出：`s3://spike-graphiti-logs/athena-results/`

## 表结构

### 1. `graphiti_llm_invocation_logs` — Graphiti 运行时日志

来源：S3 Logger 在 Graphiti 运行时写入的 gzip JSON Lines 文件。记录所有 LLM 和 Embedding 调用。

S3 路径：`s3://spike-graphiti-logs/GraphitiLogs/year={Y}/month={M}/day={D}/hour={H}/`

分区方式：Hive-style（`year=`, `month=`, `day=`, `hour=`）

| 列名 | 类型 | 说明 |
|------|------|------|
| timestamp | STRING | ISO 8601 时间戳 |
| operation | STRING | `llm.generate` / `embedding.create` |
| model_id | STRING | 模型 ID，如 `moonshotai.kimi-k2.5` |
| prompt_name | STRING | prompt 名称，如 `extract_edges.edge`、`dedupe_nodes.nodes` |
| group_id | STRING | 图分区 ID |
| input_tokens | INT | 输入 token 数 |
| output_tokens | INT | 输出 token 数 |
| latency_ms | DOUBLE | 调用延迟（毫秒） |
| status | STRING | `ok` / `error` |
| error_message | STRING | 错误信息（仅 status=error 时） |
| input_preview | STRING | 输入前 200 字符预览 |
| metadata | MAP<STRING,STRING> | 扩展元数据 |
| year | INT | 分区列 |
| month | INT | 分区列 |
| day | INT | 分区列 |
| hour | INT | 分区列 |

新增分区后需执行：
```sql
MSCK REPAIR TABLE graphiti.graphiti_llm_invocation_logs;
```

### 2. `bedrock_invocation_logs` — Bedrock 原生调用日志

来源：Bedrock Model Invocation Logging 自动写入的 JSON Lines 文件。

S3 路径：`s3://spike-bedrock-invocation-logs/AWSLogs/{account_id}/BedrockModelInvocationLogs/{region}/`

分区方式：projection-based（非 Hive 路径，通过 partition projection 自动映射，无需 MSCK REPAIR）

| 列名 | 类型 | 说明 |
|------|------|------|
| timestamp | STRING | 调用时间戳 |
| accountid | STRING | AWS 账户 ID |
| region | STRING | AWS 区域 |
| requestid | STRING | 请求 ID |
| operation | STRING | API 操作名 |
| modelid | STRING | 模型 ID |
| input | STRUCT | `<inputcontenttype:string, inputbodyjson:string, inputtokencount:int>` |
| output | STRUCT | `<outputcontenttype:string, outputbodyjson:string, outputtokencount:int>` |
| identity | STRUCT | `<arn:string>` |
| schematype | STRING | 日志 schema 类型 |
| schemaversion | STRING | 日志 schema 版本 |

访问 struct 字段用点号：`input.inputtokencount`、`output.outputtokencount`、`input.inputbodyjson`

重要限制：此表只记录通过标准 Bedrock API（`InvokeModel`/`InvokeModelWithResponseStream`）的调用。
通过 Bedrock Mantle（OpenAI 兼容端点）的调用（如 Kimi K2.5）不会出现在此表中。
因此 LLM 调用的 token 数只能从 `graphiti_llm_invocation_logs` 获取，Embedding 调用（Nova MME）两个表都有。

## 两个表的覆盖范围

| 调用类型 | graphiti_llm_invocation_logs | bedrock_invocation_logs |
|---------|------------------------------|------------------------|
| Kimi K2.5 LLM（Mantle） | ✅ 有（含 latency、prompt_name） | ❌ 无（Mantle 不走 InvokeModel） |
| Nova Embedding（InvokeModel） | ✅ 有 | ✅ 有（含完整 request/response body） |
| 其他标准 Bedrock 模型 | ✅ 有（如果集成了 S3 Logger） | ✅ 有 |

## prompt_name 对照表

| prompt_name | 阶段 | 说明 | 调用次数/段落 |
|-------------|------|------|--------------|
| `extract_nodes.extract_text` | Phase 1 | 从原文提取实体 | 1 |
| `dedupe_nodes.nodes` | Phase 2 | 批量节点去重 | 0~1 |
| `extract_edges.edge` | Phase 3 | 从原文提取关系边 + uncovered excerpts | 1（最慢，可达 200s+） |
| `dedupe_edges.resolve_edge` | Phase 3 | 单条边去重（并行，受 SEMAPHORE_LIMIT 控制） | N（边数） |
| `extract_edges.extract_attributes` | Phase 3 | 边属性提取（有自定义 edge type 时） | 0~N |
| `extract_nodes.extract_summaries_batch` | Phase 4 | 节点 summary 生成 | 若干 |

## 常用查询

### 查看新分区（新数据写入后执行）

```sql
MSCK REPAIR TABLE graphiti.graphiti_llm_invocation_logs;
```

### LLM 调用延迟分布（按 prompt 类型）

```sql
SELECT prompt_name,
       COUNT(*) AS calls,
       SUM(input_tokens) AS total_input,
       SUM(output_tokens) AS total_output,
       ROUND(AVG(latency_ms)) AS avg_ms,
       ROUND(APPROX_PERCENTILE(latency_ms, 0.5)) AS p50_ms,
       ROUND(APPROX_PERCENTILE(latency_ms, 0.9)) AS p90_ms,
       ROUND(MAX(latency_ms)) AS max_ms
FROM graphiti.graphiti_llm_invocation_logs
WHERE year = 2026 AND month = 3 AND day = 2
  AND operation = 'llm.generate'
GROUP BY prompt_name
ORDER BY avg_ms DESC;
```

### 找出超长耗时的 LLM 调用

```sql
SELECT timestamp, prompt_name, model_id,
       input_tokens, output_tokens,
       ROUND(latency_ms / 1000, 1) AS latency_sec,
       input_preview
FROM graphiti.graphiti_llm_invocation_logs
WHERE year = 2026 AND month = 3 AND day = 2
  AND operation = 'llm.generate'
  AND latency_ms > 30000
ORDER BY latency_ms DESC;
```

### Embedding 调用统计

```sql
SELECT COUNT(*) AS calls,
       SUM(input_tokens) AS total_tokens,
       ROUND(AVG(latency_ms)) AS avg_ms,
       ROUND(APPROX_PERCENTILE(latency_ms, 0.9)) AS p90_ms
FROM graphiti.graphiti_llm_invocation_logs
WHERE year = 2026 AND month = 3 AND day = 2
  AND operation = 'embedding.create';
```

### 每小时调用量趋势

```sql
SELECT hour, operation,
       COUNT(*) AS calls,
       SUM(input_tokens) AS input_tokens,
       SUM(output_tokens) AS output_tokens
FROM graphiti.graphiti_llm_invocation_logs
WHERE year = 2026 AND month = 3 AND day = 2
GROUP BY hour, operation
ORDER BY hour, operation;
```

### 从 Bedrock 原生日志查 Embedding token 数

```sql
SELECT timestamp, modelid,
       input.inputtokencount AS input_tokens,
       output.outputtokencount AS output_tokens,
       requestid
FROM graphiti.bedrock_invocation_logs
WHERE modelid LIKE '%embedding%'
ORDER BY timestamp DESC
LIMIT 20;
```

### 对比两个表的 Embedding 调用

Graphiti 日志有 `prompt_name`（业务语义）和 `latency_ms`，Bedrock 日志有完整 request/response body。
两者通过时间戳近似关联：

```sql
-- 先从 graphiti 日志找到慢调用的时间范围
-- 再到 bedrock 日志查对应时间段的完整请求
SELECT b.timestamp, b.modelid, b.requestid,
       b.input.inputtokencount AS input_tokens,
       b.output.outputtokencount AS output_tokens
FROM graphiti.bedrock_invocation_logs b
WHERE b.timestamp BETWEEN '2026-03-02T14:50:00' AND '2026-03-02T15:05:00'
  AND b.modelid LIKE '%embedding%'
ORDER BY b.timestamp
LIMIT 20;
```
