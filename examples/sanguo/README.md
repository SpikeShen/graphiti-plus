# 三国演义端到端测试

基于三国演义第一回语料，验证 Graphiti + S3 Vectors 改造的完整链路：导入 → 搜索 → 深度搜索 → DescribesEdge → Episode Narratives 持久化。

## 前置条件

- Neo4j 运行中（本地 docker compose）
- `.env` 配置完整（`NEO4J_URI`、`AWS_REGION`、`BEDROCK_MODEL`、`S3_VECTORS_BUCKET` 等）
- IAM role 有 Bedrock、S3 Vectors、S3 权限

## 脚本说明

| 脚本 | 功能 | 耗时 |
|------|------|------|
| `1_clear.py` | 清空 Neo4j 全部数据 + 删除并重建 S3 Vectors 8 个索引 | ~3s |
| `2_ingest.py N` | 导入第一回前 N 段落（默认全部），调用真实 Bedrock LLM + Nova Embedding | ~100s/段落 |
| `3_search.py` | 标准搜索 + 深度搜索对比（"张角是什么人"、"黄巾起义的原因"、"念咒者何人"） | ~10s |
| `4_describes.py` | 按 Episode 查看 DescribesEdge（excerpt + fact） | ~1s |
| `5_uncovered.py` | 查看 Neo4j 中持久化的 episode narratives | ~1s |

## 使用方式

```bash
cd graphiti

# 全流程（清库 → 导入3段落 → 搜索 → describes → episode narratives）
examples/sanguo/run.sh all 3

# 单步执行
examples/sanguo/run.sh 1           # 清库
examples/sanguo/run.sh 2 3         # 导入前3段落
examples/sanguo/run.sh 3           # 搜索测试
examples/sanguo/run.sh 4           # 查看 DescribesEdge
examples/sanguo/run.sh 5           # 查看 episode narratives
```

不要直接 `python examples/sanguo/2_ingest.py` 运行，绕过了 `run.sh` 的冲突检测。

## 典型输出（3 段落）

导入结果：
- 77 实体、86 边（100% 有 source_excerpt）、20 episode narratives
- 段落 0 ~97s、段落 1 ~244s、段落 2 ~240s，总计 ~580s

搜索对比（"念咒者何人"）：
- 标准搜索：10 条边，未命中"念咒"相关内容
- 深度搜索：20 条边 + 20 episode narratives，首条即"角有徒弟五百余人，云游四方，皆能书符念咒"（score=0.787）

## 注意事项

- `1_clear.py` 会清空 Neo4j **全部数据**（不限 group_id），谨慎执行
- 导入耗时受 Bedrock Mantle 网关稳定性影响，偶发超时会自动重试（15s/60s 超时配置）
- 可通过 `GRAPHITI_LLM_TRACE=true` 启用 LLM 调用详细日志
