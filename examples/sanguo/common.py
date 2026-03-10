"""
Shared utilities for 三国演义 test scripts.

Provides: Graphiti client construction, corpus loading, clear/ingest helpers.
"""

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

# Ensure LLM trace is off by default (can be overridden by shell env)
os.environ.setdefault('GRAPHITI_LLM_TRACE', 'false')

# Suppress noisy third-party logs
logging.getLogger('neo4j').setLevel(logging.WARNING)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

REGION = os.environ['AWS_REGION']
LLM_MODEL = os.environ['BEDROCK_MODEL']
EMBEDDING_MODEL = os.environ['BEDROCK_EMBEDDING_MODEL']
EMBEDDING_DIM = int(os.environ.get('BEDROCK_EMBEDDING_DIM', '1024'))
LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.1'))
S3_VECTORS_BUCKET = os.environ['S3_VECTORS_BUCKET']

CORPUS_PATH = os.path.join(os.path.dirname(__file__), '三国演义.txt')
GROUP_ID = 'sanguo-test'


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


def load_chapter1_paragraphs(n: int | None = None) -> list[str]:
    """Load first n paragraphs of Chapter 1 from 三国演义. None = all."""
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    m_start = re.search(r'第一回\s+.+', text)
    m_end = re.search(r'第二回\s+', text)
    if not m_start:
        raise ValueError('Cannot find 第一回')
    ch1_text = text[m_start.end() : m_end.start()] if m_end else text[m_start.end() :]
    raw = re.split(r'\n\s*\n', ch1_text)
    paragraphs = [p.strip() for p in raw if len(p.strip()) > 30]
    return paragraphs[:n] if n else paragraphs


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
    """Ingest paragraphs as episodes, print timing."""
    print(f'\n--- Ingesting {len(paragraphs)} paragraphs ---', flush=True)
    total_start = _time.time()
    for i, paragraph in enumerate(paragraphs):
        ep_start = _time.time()
        await graphiti.add_episode(
            name=f'三国演义-第一回-段落{i}',
            episode_body=paragraph,
            source=EpisodeType.text,
            source_description='三国演义 第一回',
            reference_time=datetime.now(timezone.utc),
            group_id=GROUP_ID,
        )
        print(f'  [{i}] {_time.time() - ep_start:.1f}s', flush=True)
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
