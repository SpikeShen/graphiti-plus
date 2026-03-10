# 多模态文档端到端测试（docx-manual）

基于 "Global MAP 2.0 CCS Tagging 操作指南" Word 文档，验证多模态文档（图文混排）的完整链路：
文档解析 → ContentBlock 构建 → 导入 → 搜索 → DescribesEdge → Episode Narratives → 图片搜索。

## 测试文档

`test-document.docx`（与脚本同目录）— 一份 AWS MAP 2.0 CCS 打标签操作手册：
- 6 个章节（Heading 1/2）
- 9 张嵌入图片（操作截图）
- 2 个表格（文档信息、维护记录）
- 图文混排：文字说明 + 操作截图交替出现

## 前置条件

- 与 sanguo 相同（Neo4j、.env、IAM 权限）
- `python-docx` 已安装（`uv pip install python-docx`）

## 脚本说明

| 脚本 | 功能 | 耗时 |
|------|------|------|
| `1_clear.py` | 清空 Neo4j 全部数据 + 删除并重建 S3 Vectors 索引 | ~3s |
| `2_ingest.py` | 解析 docx → ContentBlocks → 按章节导入 episode（完整多模态流程） | ~45s/章节 |
| `3_search.py` | 文本搜索测试（标准 + 深度搜索） | ~10s |
| `4_describes.py` | 按 Episode 查看 DescribesEdge | ~1s |
| `5_uncovered.py` | 查看 episode narratives | ~1s |
| `6_verify_blocks.py` | 验证 content_blocks 序列化/反序列化 + s3_uri | ~1s |
| `7_image_search.py` | 图片搜索测试（跨模态向量检索） | ~10s |

## 使用方式

```bash
cd graphiti

# 全流程
examples/docx-manual/run.sh all

# 单步执行
examples/docx-manual/run.sh 1           # 清库
examples/docx-manual/run.sh 2           # 解析并导入文档
examples/docx-manual/run.sh 2 3         # 只导入前 3 个章节
examples/docx-manual/run.sh 3           # 文本搜索测试
examples/docx-manual/run.sh 4           # 查看 DescribesEdge
examples/docx-manual/run.sh 5           # 查看 episode narratives
examples/docx-manual/run.sh 6           # 验证 content_blocks
examples/docx-manual/run.sh 7           # 图片搜索（默认 test-image-query.png）
examples/docx-manual/run.sh 7 /path/to/img.png  # 图片搜索（自定义图片）
```

不要直接 `python examples/docx-manual/xxx.py` 运行，绕过了 `run.sh` 的冲突检测。

## 与 sanguo 的区别

| 维度 | sanguo | docx-manual |
|------|--------|-------------|
| 数据源 | 纯文本（三国演义.txt） | Word 文档（图文混排） |
| EpisodeType | `text` | `document` |
| content_blocks | `[]`（空） | 包含 text + image + table 块 |
| 导入方式 | `add_episode()` | `add_document_episode()`（完整多模态流程） |
| 导入粒度 | 按段落 | 按章节 |
| 测试重点 | 纯文本兼容性 | 多模态 ContentBlock + 图片搜索 |

## 注意事项

- `1_clear.py` 会清空 Neo4j **全部数据**，谨慎执行
- 导入使用 `add_document_episode()` 完整多模态流程（章节 13 设计）：
  1. 解析 docx → ContentBlock 列表
  2. Nova MME 生成图片 embedding（在 S3 上传前）
  3. 上传图片到 S3 多模态资产桶（`MULTIMODAL_ASSET_BUCKET`），保留 `_raw_bytes`
  4. Vision LLM 生成图片描述（使用 `_raw_bytes`）
  5. 构建 content text（文本 + `[image:s3://...]` 引用）
  6. 设置 `image_embedding_map`（用于 edge source_excerpt 的图片向量）
  7. 委托 `add_episode()` 完成图谱构建（LLM 直接看到原始图片，单阶段提取）
  8. finally 清除 `_raw_bytes` + `image_embedding_map`
- 图片的 source_excerpt 使用 `[image:s3://...]` 格式引用
- 图片搜索使用 vector-only config（无 BM25），因为空文本 query 无法驱动 Lucene fulltext

## 测试结果（2026-03-05，单阶段多模态提取）

### 导入（脚本 2）
- 8 sections, 672s 总耗时
- 68 entities, 8 episodes, 63 edges (63/63 有 source_excerpt)
- 11 DescribesEdges, 10 narrative excerpts
- 30 content_blocks (19 text, 9 image, 2 table)
- 所有 9 张图片均有 s3_uri，指向 `graphiti-multimodal-assets-poc` bucket
- Vision LLM 直接看到原始图片，提取出图片中的 UI 元素、按钮文字、表格数据等细节

### 文本搜索（脚本 3）
- Standard search: 4 个查询均返回高质量相关边，包含图片来源的具体 fact
- Deep search: 20 edges + 20 nodes + 10 narratives
- 中文内容提取准确

### DescribesEdge（脚本 4）
- 11 个 DescribesEdges 分布在 8 个 episodes
- 图片类 describes 的 excerpt 正确包含 `[image:s3://...]` 引用
- fact 由 vision LLM 生成

### 图片搜索（脚本 7）
- 搜索目标：Cost Explorer "按标签分组" 页面的局部截图
- edge-source-embeddings 中图片向量 score=0.50（最高），确认跨模态检索有效
- Deep search 返回 Cost Explorer 节点和多个图片来源的边（标记 📷）
- 所有验证检查通过
