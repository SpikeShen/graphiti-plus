# Audio Extraction 示例

从视频文件中提取音频 → AWS Transcribe 语音识别 → 文字分段 → Graphiti 知识图谱导入 → 搜索验证。

## 数据源

- `全新夜航系统.mp4`：农业无人机夜航灯系统产品介绍视频（~116s，中文）
- `yehang-transcript.json`：AWS Transcribe 转录结果（预生成）

## 脚本

| 脚本 | 功能 | 说明 |
|------|------|------|
| `0_transcribe.py` | 提取音频 + AWS Transcribe 转录 | 转录结果已预生成，通常不需要重跑 |
| `1_clear.py` | 清空 Neo4j + 重建 S3 Vectors 索引 | ⚠️ 会清空所有数据 |
| `2_ingest.py` | 加载转录文本，分段导入 Graphiti | 支持 `run.sh 2 N` 只导入前 N 段 |
| `3_search.py` | 标准搜索 + 深度搜索测试 | 验证检索质量 |
| `4_describes.py` | 查看 DescribesEdge 详情 | 实体描述边 |
| `5_narratives.py` | 查看 episode narrative excerpts | 未结构化的叙事文本 |

## 用法

```bash
cd graphiti

# 全流程（清库 → 导入 → 搜索 → 验证）
examples/audio-extraction/run.sh all

# 单步执行
examples/audio-extraction/run.sh 1    # 清库
examples/audio-extraction/run.sh 2    # 导入
examples/audio-extraction/run.sh 3    # 搜索
```

## 处理流程

```
全新夜航系统.mp4
    │
    ├─ ffmpeg: 提取音频 → WAV (16kHz mono)
    │
    ├─ AWS Transcribe: 语音识别 → JSON (含时间戳)
    │
    ├─ 文本分段: 按句号/问号分句 → 合并为 ~120 字段落
    │
    └─ Graphiti add_episode(): 每段落一个 episode
        ├─ LLM 实体提取 (extract_nodes)
        ├─ LLM 关系提取 (extract_edges)
        ├─ 去重 + 矛盾检测
        ├─ Neo4j 写入
        └─ S3 Vectors 向量同步
```
