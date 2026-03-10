"""
Unit tests for S3 Vectors data structure & ingest sync (Chapter 4 & 5).

Chapter 4: S3VectorsClient — upsert, batch splitting, metadata, lifecycle, delete
Chapter 5: Graphiti._sync_*_to_s3_vectors — ingest pipeline writing to S3 Vectors

All tests use mock mode — no real AWS calls.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, call, patch

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode, EpisodeType
from graphiti_core.vector_store.s3_vectors_client import (
    MAX_BATCH_SIZE,
    S3VectorsClient,
    S3VectorsConfig,
)

pytest_plugins = ('pytest_asyncio',)

FAKE_EMBEDDING = [0.1] * 1024
NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_boto_client():
    """Create a mock boto3 s3vectors client."""
    return MagicMock()


@pytest.fixture
def s3v_client(mock_boto_client):
    """Create an S3VectorsClient with mocked boto3."""
    with patch('graphiti_core.vector_store.s3_vectors_client.boto3') as boto3_mock:
        boto3_mock.client.return_value = mock_boto_client
        client = S3VectorsClient(S3VectorsConfig(vector_bucket_name='test-bucket'))
    return client


# ===========================================================================
# Chapter 4: S3VectorsClient data structure
# ===========================================================================

class TestS3VectorsConfig:
    def test_default_index_names(self):
        cfg = S3VectorsConfig(vector_bucket_name='b')
        assert cfg.entity_index_name == 'entity-name-embeddings'
        assert cfg.edge_index_name == 'edge-fact-embeddings'
        assert cfg.edge_source_index_name == 'edge-source-embeddings'
        assert cfg.community_index_name == 'community-name-embeddings'
        assert cfg.narrative_index_name == 'episode-narrative-embeddings'
        assert cfg.describes_fact_index_name == 'describes-fact-embeddings'
        assert cfg.describes_excerpt_index_name == 'describes-excerpt-embeddings'
        assert cfg.episode_content_index_name == 'episode-content-embeddings'

    def test_eight_indices_registered(self):
        with patch('graphiti_core.vector_store.s3_vectors_client.boto3') as boto3_mock:
            boto3_mock.client.return_value = MagicMock()
            client = S3VectorsClient(S3VectorsConfig(vector_bucket_name='b'))
        assert len(client._index_names) == 8


class TestUpsertVectors:
    def test_single_vector_upsert(self, s3v_client, mock_boto_client):
        vectors = [{'key': 'k1', 'data': FAKE_EMBEDDING, 'metadata': {'group_id': 'g1'}}]
        count = s3v_client.upsert_vectors('test-index', vectors)

        assert count == 1
        mock_boto_client.put_vectors.assert_called_once()
        call_kwargs = mock_boto_client.put_vectors.call_args.kwargs
        assert call_kwargs['vectorBucketName'] == 'test-bucket'
        assert call_kwargs['indexName'] == 'test-index'
        api_vec = call_kwargs['vectors'][0]
        assert api_vec['key'] == 'k1'
        assert api_vec['data'] == {'float32': FAKE_EMBEDDING}
        assert api_vec['metadata'] == {'group_id': 'g1'}

    def test_metadata_omitted_when_empty(self, s3v_client, mock_boto_client):
        vectors = [{'key': 'k1', 'data': FAKE_EMBEDDING, 'metadata': {}}]
        s3v_client.upsert_vectors('idx', vectors)

        api_vec = mock_boto_client.put_vectors.call_args.kwargs['vectors'][0]
        assert 'metadata' not in api_vec

    def test_batch_splitting(self, s3v_client, mock_boto_client):
        """Vectors exceeding MAX_BATCH_SIZE should be split into multiple put_vectors calls."""
        n = MAX_BATCH_SIZE + 10  # 60 vectors → 2 batches (50 + 10)
        vectors = [
            {'key': f'k{i}', 'data': FAKE_EMBEDDING, 'metadata': {'i': i}}
            for i in range(n)
        ]
        count = s3v_client.upsert_vectors('idx', vectors)

        assert count == n
        assert mock_boto_client.put_vectors.call_count == 2
        # First batch should have MAX_BATCH_SIZE vectors
        first_call = mock_boto_client.put_vectors.call_args_list[0].kwargs
        assert len(first_call['vectors']) == MAX_BATCH_SIZE
        # Second batch should have the remainder
        second_call = mock_boto_client.put_vectors.call_args_list[1].kwargs
        assert len(second_call['vectors']) == 10

    def test_empty_vectors_noop(self, s3v_client, mock_boto_client):
        count = s3v_client.upsert_vectors('idx', [])
        assert count == 0
        mock_boto_client.put_vectors.assert_not_called()


class TestUpsertEntityVector:
    def test_metadata_fields(self, s3v_client, mock_boto_client):
        s3v_client.upsert_entity_vector(
            uuid='u1', embedding=FAKE_EMBEDDING,
            group_id='g1', name='刘备', created_at_ts=NOW.timestamp(),
        )
        call_kwargs = mock_boto_client.put_vectors.call_args.kwargs
        vec = call_kwargs['vectors'][0]
        assert vec['key'] == 'u1'
        assert vec['metadata']['name'] == '刘备'
        assert vec['metadata']['group_id'] == 'g1'
        assert call_kwargs['indexName'] == 'entity-name-embeddings'


class TestUpsertEdgeVector:
    def test_metadata_fields(self, s3v_client, mock_boto_client):
        s3v_client.upsert_edge_vector(
            uuid='e1', embedding=FAKE_EMBEDDING,
            group_id='g1', source_node_uuid='src', target_node_uuid='tgt',
            fact='刘备是蜀汉的开国皇帝', created_at_ts=NOW.timestamp(),
        )
        vec = mock_boto_client.put_vectors.call_args.kwargs['vectors'][0]
        assert vec['metadata']['source_node_uuid'] == 'src'
        assert vec['metadata']['target_node_uuid'] == 'tgt'
        assert vec['metadata']['fact'] == '刘备是蜀汉的开国皇帝'

    def test_fact_truncated_to_200(self, s3v_client, mock_boto_client):
        long_fact = 'x' * 500
        s3v_client.upsert_edge_vector(
            uuid='e1', embedding=FAKE_EMBEDDING,
            group_id='g1', source_node_uuid='s', target_node_uuid='t',
            fact=long_fact, created_at_ts=NOW.timestamp(),
        )
        vec = mock_boto_client.put_vectors.call_args.kwargs['vectors'][0]
        assert len(vec['metadata']['fact']) == 200


class TestUpsertEdgeSourceVector:
    def test_metadata_fields(self, s3v_client, mock_boto_client):
        s3v_client.upsert_edge_source_vector(
            uuid='e1', embedding=FAKE_EMBEDDING,
            group_id='g1', source_excerpt='原文片段', created_at_ts=NOW.timestamp(),
        )
        call_kwargs = mock_boto_client.put_vectors.call_args.kwargs
        assert call_kwargs['indexName'] == 'edge-source-embeddings'
        vec = call_kwargs['vectors'][0]
        assert vec['metadata']['source_excerpt'] == '原文片段'

    def test_excerpt_truncated_to_200(self, s3v_client, mock_boto_client):
        long_excerpt = '字' * 300
        s3v_client.upsert_edge_source_vector(
            uuid='e1', embedding=FAKE_EMBEDDING,
            group_id='g1', source_excerpt=long_excerpt, created_at_ts=NOW.timestamp(),
        )
        vec = mock_boto_client.put_vectors.call_args.kwargs['vectors'][0]
        assert len(vec['metadata']['source_excerpt']) == 200


class TestUpsertNarrativeVector:
    def test_metadata_fields(self, s3v_client, mock_boto_client):
        s3v_client.upsert_narrative_vector(
            key='ep1:narrative:abc12345', embedding=FAKE_EMBEDDING,
            group_id='g1', episode_uuid='ep1', excerpt='残余文本',
            created_at_ts=NOW.timestamp(),
        )
        call_kwargs = mock_boto_client.put_vectors.call_args.kwargs
        assert call_kwargs['indexName'] == 'episode-narrative-embeddings'
        vec = call_kwargs['vectors'][0]
        assert vec['key'] == 'ep1:narrative:abc12345'
        assert vec['metadata']['episode_uuid'] == 'ep1'
        assert vec['metadata']['excerpt'] == '残余文本'


class TestUpsertCommunityVector:
    def test_metadata_fields(self, s3v_client, mock_boto_client):
        s3v_client.upsert_community_vector(
            uuid='c1', embedding=FAKE_EMBEDDING,
            group_id='g1', name='蜀汉阵营', created_at_ts=NOW.timestamp(),
        )
        call_kwargs = mock_boto_client.put_vectors.call_args.kwargs
        assert call_kwargs['indexName'] == 'community-name-embeddings'
        vec = call_kwargs['vectors'][0]
        assert vec['metadata']['name'] == '蜀汉阵营'


class TestDeleteVectors:
    def test_delete_by_keys(self, s3v_client, mock_boto_client):
        s3v_client.delete_vectors('idx', ['k1', 'k2'])
        mock_boto_client.delete_vectors.assert_called_once_with(
            vectorBucketName='test-bucket', indexName='idx', keys=['k1', 'k2'],
        )

    def test_delete_empty_noop(self, s3v_client, mock_boto_client):
        s3v_client.delete_vectors('idx', [])
        mock_boto_client.delete_vectors.assert_not_called()

    def test_delete_batch_splitting(self, s3v_client, mock_boto_client):
        keys = [f'k{i}' for i in range(MAX_BATCH_SIZE + 5)]
        s3v_client.delete_vectors('idx', keys)
        assert mock_boto_client.delete_vectors.call_count == 2

    def test_delete_entity_vectors(self, s3v_client, mock_boto_client):
        s3v_client.delete_entity_vectors(['u1'])
        call_kwargs = mock_boto_client.delete_vectors.call_args.kwargs
        assert call_kwargs['indexName'] == 'entity-name-embeddings'

    def test_delete_edge_vectors(self, s3v_client, mock_boto_client):
        s3v_client.delete_edge_vectors(['e1'])
        call_kwargs = mock_boto_client.delete_vectors.call_args.kwargs
        assert call_kwargs['indexName'] == 'edge-fact-embeddings'

    def test_delete_edge_source_vectors(self, s3v_client, mock_boto_client):
        s3v_client.delete_edge_source_vectors(['e1'])
        call_kwargs = mock_boto_client.delete_vectors.call_args.kwargs
        assert call_kwargs['indexName'] == 'edge-source-embeddings'

    def test_delete_community_vectors(self, s3v_client, mock_boto_client):
        s3v_client.delete_community_vectors(['c1'])
        call_kwargs = mock_boto_client.delete_vectors.call_args.kwargs
        assert call_kwargs['indexName'] == 'community-name-embeddings'

    def test_delete_narrative_vectors(self, s3v_client, mock_boto_client):
        s3v_client.delete_narrative_vectors(['k1'])
        call_kwargs = mock_boto_client.delete_vectors.call_args.kwargs
        assert call_kwargs['indexName'] == 'episode-narrative-embeddings'


class TestLifecycle:
    def test_ensure_bucket_and_indices_creates_all(self, s3v_client, mock_boto_client):
        """Should create bucket + 5 indices when none exist."""
        from botocore.exceptions import ClientError
        error_response = {'Error': {'Code': 'VectorBucketNotFoundException'}}
        mock_boto_client.get_vector_bucket.side_effect = ClientError(error_response, 'GetVectorBucket')

        index_error = {'Error': {'Code': 'VectorIndexNotFoundException'}}
        mock_boto_client.get_index.side_effect = ClientError(index_error, 'GetIndex')

        s3v_client.ensure_bucket_and_indices()

        mock_boto_client.create_vector_bucket.assert_called_once()
        assert mock_boto_client.create_index.call_count == 8

    def test_ensure_bucket_already_exists(self, s3v_client, mock_boto_client):
        """Should not create bucket if it already exists."""
        mock_boto_client.get_vector_bucket.return_value = {}
        mock_boto_client.get_index.return_value = {}

        s3v_client.ensure_bucket_and_indices()

        mock_boto_client.create_vector_bucket.assert_not_called()
        mock_boto_client.create_index.assert_not_called()

    def test_delete_all_indices(self, s3v_client, mock_boto_client):
        s3v_client.delete_all_indices()
        assert mock_boto_client.delete_index.call_count == 8


# ===========================================================================
# Chapter 5: Graphiti._sync_*_to_s3_vectors (ingest pipeline → S3 Vectors)
# ===========================================================================

def _make_graphiti_with_mock_s3v():
    """Create a Graphiti instance with mocked dependencies and a mock S3VectorsClient."""
    mock_s3v = MagicMock(spec=S3VectorsClient)

    with patch('graphiti_core.graphiti.GraphDriver'), \
         patch('graphiti_core.graphiti.EmbedderClient'):
        from graphiti_core.graphiti import Graphiti
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.client import LLMClient

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.config = LLMConfig()
        mock_embedder = MagicMock()
        mock_embedder.create_batch = AsyncMock()

        g = object.__new__(Graphiti)
        g.driver = MagicMock()
        g.llm_client = mock_llm
        g.embedder = mock_embedder
        g.cross_encoder = MagicMock()
        g.s3_vectors = mock_s3v
        g.tracer = MagicMock()
        g.s3_logger = None

    return g, mock_s3v


class TestSyncNodesToS3Vectors:
    def test_syncs_nodes_with_embeddings(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        node = EntityNode(
            uuid='n1', group_id='g1', name='刘备', summary='',
            created_at=NOW, name_embedding=FAKE_EMBEDDING,
        )
        g._sync_nodes_to_s3_vectors([node])

        mock_s3v.upsert_entity_vector.assert_called_once_with(
            uuid='n1', embedding=FAKE_EMBEDDING,
            group_id='g1', name='刘备', created_at_ts=NOW.timestamp(),
        )

    def test_skips_nodes_without_embedding(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        node = EntityNode(
            uuid='n1', group_id='g1', name='刘备', summary='',
            created_at=NOW, name_embedding=None,
        )
        g._sync_nodes_to_s3_vectors([node])
        mock_s3v.upsert_entity_vector.assert_not_called()

    def test_noop_when_s3_vectors_is_none(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        g.s3_vectors = None
        node = EntityNode(
            uuid='n1', group_id='g1', name='刘备', summary='',
            created_at=NOW, name_embedding=FAKE_EMBEDDING,
        )
        g._sync_nodes_to_s3_vectors([node])
        mock_s3v.upsert_entity_vector.assert_not_called()


class TestSyncEdgesToS3Vectors:
    def test_syncs_fact_and_source_excerpt(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        edge = EntityEdge(
            uuid='e1', group_id='g1', source_node_uuid='src', target_node_uuid='tgt',
            fact='刘备是蜀汉开国皇帝', name='edge1', episodes=[],
            created_at=NOW, expired_at=None,
            fact_embedding=FAKE_EMBEDDING,
            source_excerpt='原文片段', source_excerpt_embedding=FAKE_EMBEDDING,
        )
        g._sync_edges_to_s3_vectors([edge])

        # Both fact and source_excerpt should be upserted
        mock_s3v.upsert_edge_vector.assert_called_once()
        mock_s3v.upsert_edge_source_vector.assert_called_once()

    def test_skips_source_excerpt_when_missing(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        edge = EntityEdge(
            uuid='e1', group_id='g1', source_node_uuid='src', target_node_uuid='tgt',
            fact='test', name='edge1', episodes=[], created_at=NOW, expired_at=None,
            fact_embedding=FAKE_EMBEDDING,
            source_excerpt='', source_excerpt_embedding=None,
        )
        g._sync_edges_to_s3_vectors([edge])

        mock_s3v.upsert_edge_vector.assert_called_once()
        mock_s3v.upsert_edge_source_vector.assert_not_called()

    def test_skips_edge_without_fact_embedding(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        edge = EntityEdge(
            uuid='e1', group_id='g1', source_node_uuid='src', target_node_uuid='tgt',
            fact='test', name='edge1', episodes=[], created_at=NOW, expired_at=None,
            fact_embedding=None,
        )
        g._sync_edges_to_s3_vectors([edge])

        mock_s3v.upsert_edge_vector.assert_not_called()
        mock_s3v.upsert_edge_source_vector.assert_not_called()

    def test_noop_when_s3_vectors_is_none(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        g.s3_vectors = None
        edge = EntityEdge(
            uuid='e1', group_id='g1', source_node_uuid='src', target_node_uuid='tgt',
            fact='test', name='edge1', episodes=[], created_at=NOW, expired_at=None,
            fact_embedding=FAKE_EMBEDDING,
        )
        g._sync_edges_to_s3_vectors([edge])
        mock_s3v.upsert_edge_vector.assert_not_called()


class TestSyncCommunitiesToS3Vectors:
    def test_syncs_community_with_embedding(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        comm = CommunityNode(
            uuid='c1', group_id='g1', name='蜀汉', summary='',
            created_at=NOW, name_embedding=FAKE_EMBEDDING,
        )
        g._sync_communities_to_s3_vectors([comm])

        mock_s3v.upsert_community_vector.assert_called_once_with(
            uuid='c1', embedding=FAKE_EMBEDDING,
            group_id='g1', name='蜀汉', created_at_ts=NOW.timestamp(),
        )

    def test_skips_community_without_embedding(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        comm = CommunityNode(
            uuid='c1', group_id='g1', name='蜀汉', summary='',
            created_at=NOW, name_embedding=None,
        )
        g._sync_communities_to_s3_vectors([comm])
        mock_s3v.upsert_community_vector.assert_not_called()


class TestSyncNarrativesToS3Vectors:
    def _make_episode(self):
        return EpisodicNode(
            uuid='ep1', group_id='g1', name='ep1',
            source=EpisodeType.text, content='test', created_at=NOW,
            source_description='test', valid_at=NOW,
        )

    @pytest.mark.asyncio
    async def test_embeds_and_upserts_excerpts(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        g.embedder.create_batch.return_value = [FAKE_EMBEDDING, FAKE_EMBEDDING]

        episode = self._make_episode()
        await g._sync_narratives_to_s3_vectors(
            ['残余文本1', '残余文本2'], episode, 'g1', NOW,
        )

        assert mock_s3v.upsert_narrative_vector.call_count == 2
        # Verify key format: {episode_uuid}:narrative:{hash}
        first_call = mock_s3v.upsert_narrative_vector.call_args_list[0]
        assert first_call.kwargs['key'].startswith('ep1:narrative:')
        assert len(first_call.kwargs['key'].split(':')[-1]) == 8  # SHA256 first 8 chars

    @pytest.mark.asyncio
    async def test_filters_empty_excerpts(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        g.embedder.create_batch.return_value = [FAKE_EMBEDDING]

        episode = self._make_episode()
        await g._sync_narratives_to_s3_vectors(
            ['', '  ', '有效文本'], episode, 'g1', NOW,
        )

        # Only the valid excerpt should be embedded and upserted
        g.embedder.create_batch.assert_called_once_with(['有效文本'])
        mock_s3v.upsert_narrative_vector.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_s3_vectors_is_none(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()
        g.s3_vectors = None

        episode = self._make_episode()
        await g._sync_narratives_to_s3_vectors(
            ['text'], episode, 'g1', NOW,
        )
        mock_s3v.upsert_narrative_vector.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_excerpts_empty(self):
        g, mock_s3v = _make_graphiti_with_mock_s3v()

        episode = self._make_episode()
        await g._sync_narratives_to_s3_vectors(
            [], episode, 'g1', NOW,
        )
        mock_s3v.upsert_narrative_vector.assert_not_called()
        g.embedder.create_batch.assert_not_called()
