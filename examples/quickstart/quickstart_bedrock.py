"""
Quickstart: Graphiti with AWS Bedrock + S3 Vectors + Neo4j.

Demonstrates the full AWS-native stack:
  - LLM: Kimi K2.5 via Bedrock Mantle
  - Embeddings: Nova Multimodal Embeddings via Bedrock
  - Vector search: S3 Vectors (ANN index, replaces Neo4j brute-force cosine)
  - Graph store: Neo4j
  - Invocation logging: S3 (optional, for Athena analytics)

Prerequisites:
1. Neo4j running locally: cd graphiti && docker compose up neo4j -d
2. AWS credentials configured with Bedrock + S3 Vectors access (us-east-1)
3. Bedrock model access enabled for:
   - moonshotai.kimi-k2.5 (LLM)
   - amazon.nova-2-multimodal-embeddings-v1:0 (Embeddings)
4. Install deps:
   cd graphiti
   uv sync --extra dev
   uv pip install aws-bedrock-token-generator

Usage:
    cd graphiti
    uv run python examples/quickstart/quickstart_bedrock.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.bedrock_reranker_client import BedrockRerankerClient
from graphiti_core.embedder.bedrock_nova import BedrockNovaEmbedder, BedrockNovaEmbedderConfig
from graphiti_core.llm_client.bedrock_client import BedrockLLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.logging import create_s3_logger
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.vector_store.s3_vectors_client import S3VectorsClient, S3VectorsConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration — all from .env
REGION = os.environ['AWS_REGION']
LLM_MODEL = os.environ['BEDROCK_MODEL']
EMBEDDING_MODEL = os.environ['BEDROCK_EMBEDDING_MODEL']
EMBEDDING_DIM = int(os.environ.get('BEDROCK_EMBEDDING_DIM', '1024'))
LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.1'))
S3_VECTORS_BUCKET = os.environ.get('S3_VECTORS_BUCKET', '')

# Neo4j
neo4j_uri = os.environ['NEO4J_URI']
neo4j_user = os.environ['NEO4J_USER']
neo4j_password = os.environ['NEO4J_PASSWORD']


async def main():
    # Bedrock LLM client (Kimi K2.5 via Mantle OpenAI-compatible endpoint)
    llm_client = BedrockLLMClient(
        config=LLMConfig(model=LLM_MODEL, small_model=LLM_MODEL, temperature=LLM_TEMPERATURE),
        region_name=REGION,
    )

    # Nova Multimodal Embeddings
    embedder = BedrockNovaEmbedder(
        config=BedrockNovaEmbedderConfig(
            model_id=EMBEDDING_MODEL,
            region_name=REGION,
            embedding_dim=EMBEDDING_DIM,
        )
    )

    # Reranker shares the same OpenAI client as the LLM
    cross_encoder = BedrockRerankerClient(
        client=llm_client.client,
        model_id=LLM_MODEL,
    )

    # S3 Vectors client (optional — set S3_VECTORS_BUCKET in .env to enable)
    s3_vectors = None
    if S3_VECTORS_BUCKET:
        s3_vectors = S3VectorsClient(
            config=S3VectorsConfig(
                vector_bucket_name=S3_VECTORS_BUCKET,
                region_name=REGION,
                embedding_dim=EMBEDDING_DIM,
            )
        )

    # S3 invocation logger (optional — set S3_LOG_BUCKET in .env to enable)
    s3_logger = create_s3_logger()

    # Initialize Graphiti
    graphiti = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        s3_vectors=s3_vectors,
        s3_logger=s3_logger,
    )

    try:
        await graphiti.build_indices_and_constraints()
        print('Indices and constraints built.')

        # Add test episodes
        episodes = [
            {
                'content': 'Alice is a software engineer at TechCorp. '
                'She works on the backend team and specializes in Python.',
                'type': EpisodeType.text,
                'description': 'team info',
            },
            {
                'content': 'Bob is the CTO of TechCorp. He founded the company in 2020.',
                'type': EpisodeType.text,
                'description': 'team info',
            },
            {
                'content': json.dumps({
                    'name': 'TechCorp',
                    'industry': 'Software',
                    'founded': '2020',
                    'headquarters': 'San Francisco',
                }),
                'type': EpisodeType.json,
                'description': 'company metadata',
            },
        ]

        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Team Update {i}',
                episode_body=episode['content'],
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode {i}: {episode["description"]}')

        # Edge search
        print("\n--- Edge Search: 'Who works at TechCorp?' ---")
        results = await graphiti.search('Who works at TechCorp?')
        for r in results:
            print(f'  Fact: {r.fact}')

        # Node search
        print("\n--- Node Search: 'TechCorp' ---")
        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = 5
        node_results = await graphiti._search(query='TechCorp', config=config)
        for node in node_results.nodes:
            summary = node.summary[:120] + '...' if len(node.summary) > 120 else node.summary
            print(f'  {node.name}: {summary}')

        # S3 Vectors direct query (only when enabled)
        if s3_vectors:
            print('\n--- S3 Vectors Direct Query ---')
            test_embedding = await embedder.create(input_data=['TechCorp'])
            s3_results = s3_vectors.query_entity_vectors(test_embedding, top_k=5)
            print(f'Entity vectors found: {len(s3_results)}')
            for r in s3_results:
                print(f'  key={r.key}, score={r.score:.4f}, name={r.metadata.get("name", "?")}')

            s3_edge_results = s3_vectors.query_edge_vectors(test_embedding, top_k=5)
            print(f'Edge vectors found: {len(s3_edge_results)}')
            for r in s3_edge_results:
                print(f'  key={r.key}, score={r.score:.4f}, fact={r.metadata.get("fact", "?")[:80]}')

    finally:
        await graphiti.close()
        print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
