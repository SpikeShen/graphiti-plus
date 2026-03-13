"""
Shared utilities for audio-extraction test scripts.

Scenario: Extract audio from a video file → transcribe via AWS Transcribe →
split into paragraphs → ingest into Graphiti → search & verify.

Source: 全新夜航系统.mp4 (agricultural drone night-flight lighting system product intro)
"""

import json
import logging
import os
import re
import time as _time
from datetime import datetime, timezone

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.bedrock_reranker_client import BedrockRerankerClient
from graphiti_core.embedder.bedrock_nova import BedrockNovaEmbedder, BedrockNovaEmbedderConfig
from graphiti_core.llm_client.bedrock_client import BedrockLLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.logging import create_s3_logger
from graphiti_core.nodes import EpisodeType
from graphiti_core.vector_store.s3_vectors_client import S3VectorsClient, S3VectorsConfig

load_dotenv()

os.environ.setdefault('GRAPHITI_LLM_TRACE', 'false')

logging.getLogger('neo4j').setLevel(logging.WARNING)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

REGION = os.environ['AWS_REGION']
LLM_MODEL = os.environ['BEDROCK_MODEL']
EMBEDDING_MODEL = os.environ['BEDROCK_EMBEDDING_MODEL']
EMBEDDING_DIM = int(os.environ.get('BEDROCK_EMBEDDING_DIM', '1024'))
LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.1'))
S3_VECTORS_BUCKET = os.environ['S3_VECTORS_BUCKET']

SCRIPT_DIR = os.path.dirname(__file__)
VIDEO_PATH = os.path.join(SCRIPT_DIR, '全新夜航系统.mp4')
TRANSCRIPT_PATH = os.path.join(SCRIPT_DIR, 'yehang-transcript.json')
GROUP_ID = 'audio-extraction-test'


def build_graphiti() -> Graphiti:
    """Build Graphiti instance with all clients from .env config."""
    llm_client = BedrockLLMClient(
        config=LLMConfig(model=LLM_MODEL, small_model=LLM_MODEL, temperature=LLM_TEMPERATURE),
        region_name=REGION,
    )
    embedder = BedrockNovaEmbedder(
        config=BedrockNovaEmbedderConfig(
            model_id=EMBEDDING_MODEL, region_name=REGION, embedding_dim=EMBEDDING_DIM,
        )
    )
    cross_encoder = BedrockRerankerClient(client=llm_client.client, model_id=LLM_MODEL)
    s3_vectors = S3VectorsClient(
        config=S3VectorsConfig(
            vector_bucket_name=S3_VECTORS_BUCKET, region_name=REGION, embedding_dim=EMBEDDING_DIM,
        )
    )
    return Graphiti(
        os.environ['NEO4J_URI'], os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'],
        llm_client=llm_client, embedder=embedder,
        cross_encoder=cross_encoder, s3_vectors=s3_vectors,
        s3_logger=create_s3_logger(),
    )


def load_transcript_paragraphs() -> list[str]:
    """Load transcript from AWS Transcribe JSON and split into paragraphs.

    Merges short sentences into paragraphs of reasonable length for Graphiti
    ingest (each paragraph becomes one episode).
    """
    with open(TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    transcript = data['results']['transcripts'][0]['transcript']

    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[。？！?!])', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Merge into paragraphs: group sentences so each paragraph is ~100-200 chars
    paragraphs: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for s in sentences:
        buf.append(s)
        buf_len += len(s)
        if buf_len >= 120:
            paragraphs.append(''.join(buf))
            buf = []
            buf_len = 0
    if buf:
        paragraphs.append(''.join(buf))

    return paragraphs


async def clear_all(graphiti: Graphiti):
    """Clear Neo4j data and rebuild S3 Vectors indices."""
    print('Clearing all data...', flush=True)
    await graphiti.driver.execute_query("MATCH (n) DETACH DELETE n")
    print('  Neo4j cleared.', flush=True)

    if graphiti.s3_vectors is not None:
        graphiti.s3_vectors.delete_all_indices()
        print('  S3 Vectors indices deleted.', flush=True)

    await graphiti.build_indices_and_constraints()
    print('  Indices rebuilt.', flush=True)


async def ingest_paragraphs(graphiti: Graphiti, paragraphs: list[str]):
    """Ingest paragraphs as text episodes."""
    print(f'\n--- Ingesting {len(paragraphs)} paragraphs ---', flush=True)
    total_start = _time.time()
    for i, paragraph in enumerate(paragraphs):
        ep_start = _time.time()
        await graphiti.add_episode(
            name=f'夜航系统-段落{i}',
            episode_body=paragraph,
            source=EpisodeType.text,
            source_description='全新夜航系统 产品介绍视频（音频转录）',
            reference_time=datetime.now(timezone.utc),
            group_id=GROUP_ID,
        )
        print(f'  [{i}] {_time.time() - ep_start:.1f}s  ({len(paragraph)} chars)', flush=True)
    print(f'Total: {_time.time() - total_start:.1f}s', flush=True)


async def print_graph_stats(graphiti: Graphiti):
    """Print entity/edge/episode counts."""
    print('\n--- Graph Stats ---', flush=True)
    for label, display in [('Entity', 'Entities'), ('Episodic', 'Episodes')]:
        records, _, _ = await graphiti.driver.execute_query(
            f"MATCH (n:{label}) WHERE n.group_id = $gid RETURN count(n) AS cnt",
            params={'gid': GROUP_ID},
        )
        print(f'  {display}: {records[0]["cnt"]}', flush=True)

    records, _, _ = await graphiti.driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid RETURN count(e) AS cnt",
        params={'gid': GROUP_ID},
    )
    print(f'  Edges: {records[0]["cnt"]}', flush=True)

    records, _, _ = await graphiti.driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid "
        "RETURN count(e) AS total, "
        "sum(CASE WHEN e.source_excerpt IS NOT NULL AND e.source_excerpt <> '' THEN 1 ELSE 0 END) AS with_excerpt",
        params={'gid': GROUP_ID},
    )
    r = records[0]
    print(f'  Edges with source_excerpt: {r["with_excerpt"]}/{r["total"]}', flush=True)

    # DescribesEdge stats
    records, _, _ = await graphiti.driver.execute_query(
        "MATCH ()-[d:DESCRIBES]->() WHERE d.group_id = $gid RETURN count(d) AS cnt",
        params={'gid': GROUP_ID},
    )
    print(f'  DescribesEdges: {records[0]["cnt"]}', flush=True)

    # Narrative excerpts stats
    records, _, _ = await graphiti.driver.execute_query(
        "MATCH (e:Episodic) WHERE e.group_id = $gid "
        "RETURN e.name AS name, e.narrative_excerpts AS narrative_excerpts",
        params={'gid': GROUP_ID},
    )
    total_narratives = 0
    for r in records:
        raw = r.get('narrative_excerpts', '[]')
        try:
            narr_list = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            narr_list = []
        total_narratives += len(narr_list)
    print(f'  Narrative excerpts: {total_narratives}', flush=True)
