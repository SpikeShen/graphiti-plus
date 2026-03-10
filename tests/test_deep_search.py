"""
Unit tests for deep search (Chapter 6 — S3 Vectors 检索流程 & 深度搜索).

Covers:
  - S3VectorsClient: query_vectors distance→similarity, filter building, min_score filtering
  - search_utils: s3_vectors_*_search bridge functions (S3→Neo4j, order preservation, dedup)
  - search.py: routing to S3 Vectors when s3_vectors client is provided, uncovered excerpts trigger

All tests use mock mode — no real AWS/Neo4j calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType
from graphiti_core.search.search_config import (
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.vector_store.s3_vectors_client import S3VectorsClient, S3VectorsConfig, VectorSearchResult

pytest_plugins = ('pytest_asyncio',)

FAKE_VECTOR = [0.1] * 1024
NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers: build minimal domain objects
# ---------------------------------------------------------------------------

def _make_entity_edge(uuid: str, fact: str = 'test fact') -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id='g1',
        source_node_uuid='src',
        target_node_uuid='tgt',
        fact=fact,
        name='test_edge',
        episodes=[],
        created_at=NOW,
        expired_at=None,
    )


def _make_entity_node(uuid: str, name: str = 'TestNode') -> EntityNode:
    return EntityNode(
        uuid=uuid,
        group_id='g1',
        name=name,
        summary='',
        created_at=NOW,
    )


def _make_community_node(uuid: str, name: str = 'TestCommunity') -> CommunityNode:
    return CommunityNode(
        uuid=uuid,
        group_id='g1',
        name=name,
        summary='',
        created_at=NOW,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_s3v_client():
    """Create a mock S3VectorsClient."""
    client = MagicMock(spec=S3VectorsClient)
    client.config = S3VectorsConfig(vector_bucket_name='test-bucket')
    return client


@pytest.fixture
def mock_driver():
    """Create a mock GraphDriver."""
    return MagicMock()


# ---------------------------------------------------------------------------
# S3VectorsClient: query_vectors
# ---------------------------------------------------------------------------

class TestS3VectorsClientQueryVectors:
    def test_distance_to_similarity_conversion(self):
        """S3 Vectors returns cosine distance; client should convert to similarity = 1 - distance."""
        mock_boto = MagicMock()
        mock_boto.query_vectors.return_value = {
            'vectors': [
                {'key': 'u1', 'distance': 0.2, 'metadata': {'group_id': 'g1'}},
                {'key': 'u2', 'distance': 0.8, 'metadata': {'group_id': 'g1'}},
            ]
        }
        with patch('graphiti_core.vector_store.s3_vectors_client.boto3') as boto3_mock:
            boto3_mock.client.return_value = mock_boto
            client = S3VectorsClient(S3VectorsConfig(vector_bucket_name='b'))
            results = client.query_vectors('idx', [0.1] * 1024, top_k=10)

        assert len(results) == 2
        assert results[0].score == pytest.approx(0.8)
        assert results[1].score == pytest.approx(0.2)

    def test_min_score_filtering(self):
        """query_entity_vectors should filter results below min_score."""
        mock_boto = MagicMock()
        mock_boto.query_vectors.return_value = {
            'vectors': [
                {'key': 'u1', 'distance': 0.1, 'metadata': {}},  # score=0.9
                {'key': 'u2', 'distance': 0.6, 'metadata': {}},  # score=0.4
                {'key': 'u3', 'distance': 0.9, 'metadata': {}},  # score=0.1
            ]
        }
        with patch('graphiti_core.vector_store.s3_vectors_client.boto3') as boto3_mock:
            boto3_mock.client.return_value = mock_boto
            client = S3VectorsClient(S3VectorsConfig(vector_bucket_name='b'))
            results = client.query_entity_vectors([0.1] * 1024, min_score=0.5)

        assert len(results) == 1
        assert results[0].key == 'u1'

    def test_group_id_single_filter(self):
        """Single group_id should produce $eq filter."""
        mock_boto = MagicMock()
        mock_boto.query_vectors.return_value = {'vectors': []}
        with patch('graphiti_core.vector_store.s3_vectors_client.boto3') as boto3_mock:
            boto3_mock.client.return_value = mock_boto
            client = S3VectorsClient(S3VectorsConfig(vector_bucket_name='b'))
            client.query_entity_vectors([0.1] * 1024, group_ids=['g1'])

        call_kwargs = mock_boto.query_vectors.call_args.kwargs
        assert call_kwargs['filter'] == {'group_id': {'$eq': 'g1'}}

    def test_group_id_multi_filter(self):
        """Multiple group_ids should produce $in filter."""
        mock_boto = MagicMock()
        mock_boto.query_vectors.return_value = {'vectors': []}
        with patch('graphiti_core.vector_store.s3_vectors_client.boto3') as boto3_mock:
            boto3_mock.client.return_value = mock_boto
            client = S3VectorsClient(S3VectorsConfig(vector_bucket_name='b'))
            client.query_entity_vectors([0.1] * 1024, group_ids=['g1', 'g2'])

        call_kwargs = mock_boto.query_vectors.call_args.kwargs
        assert call_kwargs['filter'] == {'group_id': {'$in': ['g1', 'g2']}}

    def test_edge_compound_filter(self):
        """Edge query with group_id + source_node_uuid should produce $and filter."""
        mock_boto = MagicMock()
        mock_boto.query_vectors.return_value = {'vectors': []}
        with patch('graphiti_core.vector_store.s3_vectors_client.boto3') as boto3_mock:
            boto3_mock.client.return_value = mock_boto
            client = S3VectorsClient(S3VectorsConfig(vector_bucket_name='b'))
            client.query_edge_vectors(
                [0.1] * 1024, group_ids=['g1'], source_node_uuid='src1'
            )

        call_kwargs = mock_boto.query_vectors.call_args.kwargs
        f = call_kwargs['filter']
        assert '$and' in f
        assert {'group_id': {'$eq': 'g1'}} in f['$and']
        assert {'source_node_uuid': {'$eq': 'src1'}} in f['$and']

    def test_top_k_capped_at_100(self):
        """top_k should be capped at MAX_TOP_K=100."""
        mock_boto = MagicMock()
        mock_boto.query_vectors.return_value = {'vectors': []}
        with patch('graphiti_core.vector_store.s3_vectors_client.boto3') as boto3_mock:
            boto3_mock.client.return_value = mock_boto
            client = S3VectorsClient(S3VectorsConfig(vector_bucket_name='b'))
            client.query_vectors('idx', [0.1] * 1024, top_k=500)

        call_kwargs = mock_boto.query_vectors.call_args.kwargs
        assert call_kwargs['topK'] == 100


# ---------------------------------------------------------------------------
# search_utils: s3_vectors_*_search bridge functions
# ---------------------------------------------------------------------------

class TestS3VectorsEdgeSearch:
    @pytest.mark.asyncio
    async def test_returns_edges_in_s3_order(self, mock_s3v_client, mock_driver):
        """Edges should be returned in S3 Vectors ranking order, not Neo4j order."""
        mock_s3v_client.query_edge_vectors.return_value = [
            VectorSearchResult(key='e2', score=0.9, metadata={}),
            VectorSearchResult(key='e1', score=0.7, metadata={}),
        ]
        edge1 = _make_entity_edge('e1', 'fact1')
        edge2 = _make_entity_edge('e2', 'fact2')

        with patch.object(EntityEdge, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            # Neo4j returns in arbitrary order
            mock_get.return_value = [edge1, edge2]

            from graphiti_core.search.search_utils import s3_vectors_edge_similarity_search
            result = await s3_vectors_edge_similarity_search(
                mock_s3v_client, mock_driver, FAKE_VECTOR,
                SearchFilters(), group_ids=['g1'], limit=10,
            )

        assert [e.uuid for e in result] == ['e2', 'e1']

    @pytest.mark.asyncio
    async def test_empty_s3_results_returns_empty(self, mock_s3v_client, mock_driver):
        mock_s3v_client.query_edge_vectors.return_value = []

        from graphiti_core.search.search_utils import s3_vectors_edge_similarity_search
        result = await s3_vectors_edge_similarity_search(
            mock_s3v_client, mock_driver, FAKE_VECTOR, SearchFilters(),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_neo4j_edge_skipped(self, mock_s3v_client, mock_driver):
        """If Neo4j doesn't have an edge returned by S3, it should be silently skipped."""
        mock_s3v_client.query_edge_vectors.return_value = [
            VectorSearchResult(key='e1', score=0.9, metadata={}),
            VectorSearchResult(key='e_missing', score=0.8, metadata={}),
        ]
        with patch.object(EntityEdge, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [_make_entity_edge('e1')]

            from graphiti_core.search.search_utils import s3_vectors_edge_similarity_search
            result = await s3_vectors_edge_similarity_search(
                mock_s3v_client, mock_driver, FAKE_VECTOR, SearchFilters(),
            )

        assert len(result) == 1
        assert result[0].uuid == 'e1'


class TestS3VectorsEdgeSourceSearch:
    @pytest.mark.asyncio
    async def test_returns_edges_via_source_index(self, mock_s3v_client, mock_driver):
        mock_s3v_client.query_edge_source_vectors.return_value = [
            VectorSearchResult(key='e1', score=0.85, metadata={}),
        ]
        with patch.object(EntityEdge, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [_make_entity_edge('e1')]

            from graphiti_core.search.search_utils import s3_vectors_edge_source_similarity_search
            result = await s3_vectors_edge_source_similarity_search(
                mock_s3v_client, mock_driver, FAKE_VECTOR, SearchFilters(),
            )

        assert len(result) == 1
        mock_s3v_client.query_edge_source_vectors.assert_called_once()


class TestS3VectorsNodeSearch:
    @pytest.mark.asyncio
    async def test_returns_nodes_in_s3_order(self, mock_s3v_client, mock_driver):
        mock_s3v_client.query_entity_vectors.return_value = [
            VectorSearchResult(key='n2', score=0.9, metadata={}),
            VectorSearchResult(key='n1', score=0.7, metadata={}),
        ]
        with patch.object(EntityNode, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [_make_entity_node('n1'), _make_entity_node('n2')]

            from graphiti_core.search.search_utils import s3_vectors_node_similarity_search
            result = await s3_vectors_node_similarity_search(
                mock_s3v_client, mock_driver, FAKE_VECTOR, SearchFilters(),
            )

        assert [n.uuid for n in result] == ['n2', 'n1']


class TestS3VectorsCommunitySearch:
    @pytest.mark.asyncio
    async def test_returns_communities_in_s3_order(self, mock_s3v_client, mock_driver):
        mock_s3v_client.query_community_vectors.return_value = [
            VectorSearchResult(key='c1', score=0.95, metadata={}),
        ]
        with patch.object(CommunityNode, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [_make_community_node('c1')]

            from graphiti_core.search.search_utils import s3_vectors_community_similarity_search
            result = await s3_vectors_community_similarity_search(
                mock_s3v_client, mock_driver, FAKE_VECTOR,
            )

        assert len(result) == 1
        assert result[0].uuid == 'c1'


class TestNarrativeSearch:
    @pytest.mark.asyncio
    async def test_returns_deduped_excerpts(self, mock_s3v_client):
        """Duplicate excerpts (differing only in trailing punctuation) should be deduped."""
        mock_s3v_client.query_narrative_vectors.return_value = [
            VectorSearchResult(key='k1', score=0.9, metadata={
                'excerpt': '刘备三顾茅庐。', 'episode_uuid': 'ep1',
            }),
            VectorSearchResult(key='k2', score=0.85, metadata={
                'excerpt': '刘备三顾茅庐', 'episode_uuid': 'ep1',
            }),
            VectorSearchResult(key='k3', score=0.8, metadata={
                'excerpt': '诸葛亮出山', 'episode_uuid': 'ep2',
            }),
        ]

        from graphiti_core.search.search_utils import s3_vectors_narrative_search
        result = await s3_vectors_narrative_search(
            mock_s3v_client, FAKE_VECTOR, group_ids=['g1'],
        )

        assert len(result) == 2
        assert result[0]['excerpt'] == '刘备三顾茅庐。'
        assert result[1]['excerpt'] == '诸葛亮出山'

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_s3v_client):
        mock_s3v_client.query_narrative_vectors.return_value = []

        from graphiti_core.search.search_utils import s3_vectors_narrative_search
        result = await s3_vectors_narrative_search(
            mock_s3v_client, FAKE_VECTOR,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_result_dict_structure(self, mock_s3v_client):
        mock_s3v_client.query_narrative_vectors.return_value = [
            VectorSearchResult(key='k1', score=0.9, metadata={
                'excerpt': 'some text', 'episode_uuid': 'ep1',
            }),
        ]

        from graphiti_core.search.search_utils import s3_vectors_narrative_search
        result = await s3_vectors_narrative_search(
            mock_s3v_client, FAKE_VECTOR,
        )

        assert len(result) == 1
        r = result[0]
        assert set(r.keys()) == {'key', 'score', 'excerpt', 'episode_uuid'}
        assert r['key'] == 'k1'
        assert r['score'] == 0.9
        assert r['excerpt'] == 'some text'
        assert r['episode_uuid'] == 'ep1'


# ---------------------------------------------------------------------------
# search.py: edge_search / node_search / community_search routing
# ---------------------------------------------------------------------------

class TestEdgeSearchRouting:
    @pytest.mark.asyncio
    async def test_uses_s3_vectors_when_provided(self, mock_s3v_client, mock_driver):
        """When s3_vectors is provided, cosine_similarity should route to S3 Vectors."""
        mock_s3v_client.query_edge_vectors.return_value = [
            VectorSearchResult(key='e1', score=0.9, metadata={}),
        ]
        with patch.object(EntityEdge, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [_make_entity_edge('e1')]

            from graphiti_core.search.search import edge_search
            edges, scores = await edge_search(
                driver=mock_driver,
                cross_encoder=MagicMock(),
                query='test',
                query_vector=FAKE_VECTOR,
                group_ids=['g1'],
                config=EdgeSearchConfig(
                    search_methods=[EdgeSearchMethod.cosine_similarity],
                ),
                search_filter=SearchFilters(),
                s3_vectors=mock_s3v_client,
            )

        assert len(edges) == 1
        mock_s3v_client.query_edge_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_source_similarity_routes_to_s3(self, mock_s3v_client, mock_driver):
        """source_similarity search method should use S3 Vectors edge source index."""
        mock_s3v_client.query_edge_source_vectors.return_value = [
            VectorSearchResult(key='e1', score=0.85, metadata={}),
        ]
        with patch.object(EntityEdge, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [_make_entity_edge('e1')]

            from graphiti_core.search.search import edge_search
            edges, scores = await edge_search(
                driver=mock_driver,
                cross_encoder=MagicMock(),
                query='test',
                query_vector=FAKE_VECTOR,
                group_ids=['g1'],
                config=EdgeSearchConfig(
                    search_methods=[EdgeSearchMethod.source_similarity],
                ),
                search_filter=SearchFilters(),
                s3_vectors=mock_s3v_client,
            )

        assert len(edges) == 1
        mock_s3v_client.query_edge_source_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_s3_vectors_skips_source_similarity(self, mock_driver):
        """Without s3_vectors, source_similarity should produce no results (no fallback)."""
        from graphiti_core.search.search import edge_search
        edges, scores = await edge_search(
            driver=mock_driver,
            cross_encoder=MagicMock(),
            query='test',
            query_vector=FAKE_VECTOR,
            group_ids=['g1'],
            config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.source_similarity],
            ),
            search_filter=SearchFilters(),
            s3_vectors=None,
        )
        assert edges == []


class TestNodeSearchRouting:
    @pytest.mark.asyncio
    async def test_uses_s3_vectors_when_provided(self, mock_s3v_client, mock_driver):
        mock_s3v_client.query_entity_vectors.return_value = [
            VectorSearchResult(key='n1', score=0.9, metadata={}),
        ]
        with patch.object(EntityNode, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [_make_entity_node('n1')]

            from graphiti_core.search.search import node_search
            nodes, scores = await node_search(
                driver=mock_driver,
                cross_encoder=MagicMock(),
                query='test',
                query_vector=FAKE_VECTOR,
                group_ids=['g1'],
                config=NodeSearchConfig(
                    search_methods=[NodeSearchMethod.cosine_similarity],
                ),
                search_filter=SearchFilters(),
                s3_vectors=mock_s3v_client,
            )

        assert len(nodes) == 1
        mock_s3v_client.query_entity_vectors.assert_called_once()


class TestCommunitySearchRouting:
    @pytest.mark.asyncio
    async def test_uses_s3_vectors_when_provided(self, mock_s3v_client, mock_driver):
        mock_s3v_client.query_community_vectors.return_value = [
            VectorSearchResult(key='c1', score=0.9, metadata={}),
        ]
        with patch.object(CommunityNode, 'get_by_uuids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [_make_community_node('c1')]
            # Also need to mock fulltext search since community_search always runs it
            with patch(
                'graphiti_core.search.search.community_fulltext_search',
                new_callable=AsyncMock, return_value=[],
            ):
                from graphiti_core.search.search import community_search
                communities, scores = await community_search(
                    driver=mock_driver,
                    cross_encoder=MagicMock(),
                    query='test',
                    query_vector=FAKE_VECTOR,
                    group_ids=['g1'],
                    config=CommunitySearchConfig(
                        search_methods=[CommunitySearchMethod.cosine_similarity],
                    ),
                    s3_vectors=mock_s3v_client,
                )

        assert len(communities) == 1
        mock_s3v_client.query_community_vectors.assert_called_once()


# ---------------------------------------------------------------------------
# search.py: top-level search() uncovered excerpts trigger
# ---------------------------------------------------------------------------

class TestSearchNarrativesTrigger:
    @pytest.mark.asyncio
    async def test_narratives_triggered_by_source_similarity(self, mock_s3v_client, mock_driver):
        """search() should query narrative excerpts when source_similarity is in edge config."""
        mock_s3v_client.query_edge_source_vectors.return_value = []
        mock_s3v_client.query_narrative_vectors.return_value = [
            VectorSearchResult(key='k1', score=0.8, metadata={
                'excerpt': 'narrative text', 'episode_uuid': 'ep1',
            }),
        ]
        mock_embedder = MagicMock()
        mock_embedder.create = AsyncMock(return_value=FAKE_VECTOR)

        clients = MagicMock()
        clients.driver = mock_driver
        clients.embedder = mock_embedder
        clients.cross_encoder = MagicMock()
        clients.s3_vectors = mock_s3v_client

        from graphiti_core.search.search import search
        config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.source_similarity],
            ),
        )

        with patch('graphiti_core.search.search.edge_search', new_callable=AsyncMock, return_value=([], [])), \
             patch('graphiti_core.search.search.node_search', new_callable=AsyncMock, return_value=([], [])), \
             patch('graphiti_core.search.search.episode_search', new_callable=AsyncMock, return_value=([], [])), \
             patch('graphiti_core.search.search.community_search', new_callable=AsyncMock, return_value=([], [])):
            results = await search(clients, 'query', ['g1'], config, SearchFilters())

        assert len(results.narrative_excerpts) == 1
        assert results.narrative_excerpts[0]['excerpt'] == 'narrative text'
        mock_s3v_client.query_narrative_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_narratives_not_triggered_without_source_similarity(self, mock_s3v_client, mock_driver):
        """Without source_similarity in edge config, narrative excerpts should not be searched."""
        mock_embedder = MagicMock()
        mock_embedder.create = AsyncMock(return_value=FAKE_VECTOR)

        clients = MagicMock()
        clients.driver = mock_driver
        clients.embedder = mock_embedder
        clients.cross_encoder = MagicMock()
        clients.s3_vectors = mock_s3v_client

        from graphiti_core.search.search import search
        config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.cosine_similarity],
            ),
        )

        mock_s3v_client.query_edge_vectors.return_value = []
        with patch.object(EntityEdge, 'get_by_uuids', new_callable=AsyncMock, return_value=[]), \
             patch('graphiti_core.search.search.episode_search', new_callable=AsyncMock, return_value=([], [])), \
             patch('graphiti_core.search.search.community_search', new_callable=AsyncMock, return_value=([], [])), \
             patch('graphiti_core.search.search.node_search', new_callable=AsyncMock, return_value=([], [])):
            results = await search(clients, 'query', ['g1'], config, SearchFilters())

        assert results.narrative_excerpts == []
        mock_s3v_client.query_narrative_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_narratives_not_triggered_without_s3_vectors(self, mock_driver):
        """Without s3_vectors client, narrative excerpts should not be searched."""
        mock_embedder = MagicMock()
        mock_embedder.create = AsyncMock(return_value=FAKE_VECTOR)

        clients = MagicMock()
        clients.driver = mock_driver
        clients.embedder = mock_embedder
        clients.cross_encoder = MagicMock()
        clients.s3_vectors = None

        from graphiti_core.search.search import search
        config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.source_similarity],
            ),
        )

        with patch('graphiti_core.search.search.edge_search', new_callable=AsyncMock, return_value=([], [])), \
             patch('graphiti_core.search.search.node_search', new_callable=AsyncMock, return_value=([], [])), \
             patch('graphiti_core.search.search.episode_search', new_callable=AsyncMock, return_value=([], [])), \
             patch('graphiti_core.search.search.community_search', new_callable=AsyncMock, return_value=([], [])):
            results = await search(clients, 'query', ['g1'], config, SearchFilters())

        assert results.narrative_excerpts == []


# ---------------------------------------------------------------------------
# extract_edges: empty / no-entity edge cases (Chapter 8.8)
# ---------------------------------------------------------------------------

class TestExtractEdgesEmptyInput:
    """Test extract_edges behavior with empty or no-entity paragraphs."""

    @pytest.mark.asyncio
    async def test_no_edges_extracted_returns_empty_list_with_narratives(self):
        """When LLM returns no edges, extract_edges returns ([], narrative_excerpts)."""
        from graphiti_core.utils.maintenance.edge_operations import extract_edges

        mock_llm = AsyncMock()
        mock_llm.generate_response = AsyncMock(return_value={
            'edges': [],
            'narrative_excerpts': ['这是一段没有实体关系的文本。'],
        })

        mock_clients = MagicMock()
        mock_clients.llm_client = mock_llm

        episode = MagicMock()
        episode.content = '这是一段没有实体关系的文本。'
        episode.valid_at = NOW

        edges, uncovered = await extract_edges(
            clients=mock_clients,
            episode=episode,
            nodes=[],
            previous_episodes=[],
            edge_type_map={},
            group_id='test',
        )

        assert edges == []
        assert len(uncovered) == 1
        assert uncovered[0].excerpt == '这是一段没有实体关系的文本。'
        assert uncovered[0].related_entity is None

    @pytest.mark.asyncio
    async def test_edges_with_invalid_entity_names_filtered_out(self):
        """When LLM returns edges referencing non-existent entities, they are filtered."""
        from graphiti_core.utils.maintenance.edge_operations import extract_edges

        mock_llm = AsyncMock()
        mock_llm.generate_response = AsyncMock(return_value={
            'edges': [
                {
                    'source_entity_name': '不存在的实体A',
                    'target_entity_name': '不存在的实体B',
                    'relation_type': 'KNOWS',
                    'fact': '虚假关系',
                    'source_excerpt': '原文片段',
                    'valid_at': None,
                    'invalid_at': None,
                },
            ],
            'narrative_excerpts': [],
        })

        mock_clients = MagicMock()
        mock_clients.llm_client = mock_llm

        episode = MagicMock()
        episode.content = '一些文本'
        episode.valid_at = NOW

        # Provide a node that doesn't match the edge's entity names
        node = _make_entity_node('n1', name='真实实体')

        edges, narratives = await extract_edges(
            clients=mock_clients,
            episode=episode,
            nodes=[node],
            previous_episodes=[],
            edge_type_map={},
            group_id='test',
        )

        assert edges == []
        assert narratives == []

    @pytest.mark.asyncio
    async def test_valid_edge_with_source_excerpt_returned(self):
        """When LLM returns a valid edge with source_excerpt, it is preserved."""
        from graphiti_core.utils.maintenance.edge_operations import extract_edges

        mock_llm = AsyncMock()
        mock_llm.generate_response = AsyncMock(return_value={
            'edges': [
                {
                    'source_entity_name': '刘备',
                    'target_entity_name': '关羽',
                    'relation_type': 'ALLIED_WITH',
                    'fact': '刘备与关羽结为兄弟',
                    'source_excerpt': '玄德遂以关羽、张飞为左右手',
                    'valid_at': None,
                    'invalid_at': None,
                },
            ],
            'narrative_excerpts': ['角有徒弟五百余人'],
        })

        mock_clients = MagicMock()
        mock_clients.llm_client = mock_llm

        episode = MagicMock()
        episode.content = '玄德遂以关羽、张飞为左右手'
        episode.valid_at = NOW
        episode.uuid = 'ep-001'

        node_liubei = _make_entity_node('n1', name='刘备')
        node_guanyu = _make_entity_node('n2', name='关羽')

        edges, uncovered = await extract_edges(
            clients=mock_clients,
            episode=episode,
            nodes=[node_liubei, node_guanyu],
            previous_episodes=[],
            edge_type_map={},
            group_id='test',
        )

        assert len(edges) == 1
        assert edges[0].source_excerpt == '玄德遂以关羽、张飞为左右手'
        assert edges[0].fact == '刘备与关羽结为兄弟'
        assert len(uncovered) == 1
        assert uncovered[0].excerpt == '角有徒弟五百余人'
