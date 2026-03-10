"""
S3 Vectors client for Graphiti.

Provides vector storage and similarity search using Amazon S3 Vectors,
replacing the brute-force cosine computation in Neo4j.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# S3 Vectors API limits
MAX_BATCH_SIZE = 50  # max vectors per PutVectors call
MAX_TOP_K = 100  # max results per QueryVectors call


@dataclass
class VectorSearchResult:
    """A single result from a vector similarity search."""

    key: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class S3VectorsConfig:
    """Configuration for S3 Vectors client."""

    vector_bucket_name: str
    region_name: str = 'us-east-1'
    embedding_dim: int = 1024
    distance_metric: str = 'cosine'

    # Index names
    entity_index_name: str = 'entity-name-embeddings'
    edge_index_name: str = 'edge-fact-embeddings'
    edge_source_index_name: str = 'edge-source-embeddings'
    community_index_name: str = 'community-name-embeddings'
    narrative_index_name: str = 'episode-narrative-embeddings'
    describes_fact_index_name: str = 'describes-fact-embeddings'
    describes_excerpt_index_name: str = 'describes-excerpt-embeddings'
    episode_content_index_name: str = 'episode-content-embeddings'


class S3VectorsClient:
    """Client for Amazon S3 Vectors, providing vector CRUD and similarity search."""

    def __init__(self, config: S3VectorsConfig):
        self.config = config
        self.client = boto3.client('s3vectors', region_name=config.region_name)
        self._index_names = [
            config.entity_index_name,
            config.edge_index_name,
            config.edge_source_index_name,
            config.community_index_name,
            config.narrative_index_name,
            config.describes_fact_index_name,
            config.describes_excerpt_index_name,
            config.episode_content_index_name,
        ]

    # ------------------------------------------------------------------ #
    #  Lifecycle: bucket & index creation / deletion
    # ------------------------------------------------------------------ #

    def ensure_bucket_and_indices(self) -> None:
        """Create the vector bucket and all indices if they don't exist."""
        self._ensure_bucket()
        for index_name in self._index_names:
            self._ensure_index(index_name)

    def _ensure_bucket(self) -> None:
        try:
            self.client.get_vector_bucket(vectorBucketName=self.config.vector_bucket_name)
            logger.info('Vector bucket %s already exists', self.config.vector_bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] in ('VectorBucketNotFoundException', 'NotFoundException'):
                self.client.create_vector_bucket(
                    vectorBucketName=self.config.vector_bucket_name,
                )
                logger.info('Created vector bucket %s', self.config.vector_bucket_name)
            else:
                raise

    def _ensure_index(self, index_name: str) -> None:
        try:
            self.client.get_index(
                vectorBucketName=self.config.vector_bucket_name,
                indexName=index_name,
            )
            logger.info('Vector index %s already exists', index_name)
        except ClientError as e:
            if e.response['Error']['Code'] in ('VectorIndexNotFoundException', 'NotFoundException'):
                self.client.create_index(
                    vectorBucketName=self.config.vector_bucket_name,
                    indexName=index_name,
                    dataType='float32',
                    dimension=self.config.embedding_dim,
                    distanceMetric=self.config.distance_metric,
                )
                logger.info('Created vector index %s (dim=%d)', index_name, self.config.embedding_dim)
            else:
                raise

    def delete_all_indices(self) -> None:
        """Delete all vector indices (for testing/reset)."""
        for index_name in self._index_names:
            try:
                self.client.delete_index(
                    vectorBucketName=self.config.vector_bucket_name,
                    indexName=index_name,
                )
                logger.info('Deleted vector index %s', index_name)
            except ClientError as e:
                if e.response['Error']['Code'] not in ('VectorIndexNotFoundException', 'NotFoundException'):
                    raise

    # ------------------------------------------------------------------ #
    #  Write: upsert vectors
    # ------------------------------------------------------------------ #

    def upsert_vectors(
        self,
        index_name: str,
        vectors: list[dict[str, Any]],
    ) -> int:
        """Upsert vectors into an index.

        Parameters
        ----------
        index_name : str
            Target index name.
        vectors : list[dict]
            Each dict must have: key (str), data (list[float]), metadata (dict).

        Returns
        -------
        int
            Number of vectors successfully upserted.
        """
        total = 0
        for i in range(0, len(vectors), MAX_BATCH_SIZE):
            batch = vectors[i : i + MAX_BATCH_SIZE]
            api_vectors = []
            for v in batch:
                entry: dict[str, Any] = {
                    'key': v['key'],
                    'data': {'float32': v['data']},
                }
                if v.get('metadata'):
                    entry['metadata'] = v['metadata']
                api_vectors.append(entry)

            self.client.put_vectors(
                vectorBucketName=self.config.vector_bucket_name,
                indexName=index_name,
                vectors=api_vectors,
            )
            total += len(batch)

        logger.debug('Upserted %d vectors to %s', total, index_name)
        return total

    def upsert_entity_vector(
        self,
        uuid: str,
        embedding: list[float],
        group_id: str,
        name: str,
        created_at_ts: float,
    ) -> None:
        """Upsert a single entity node embedding."""
        self.upsert_vectors(
            self.config.entity_index_name,
            [
                {
                    'key': uuid,
                    'data': embedding,
                    'metadata': {
                        'uuid': uuid,
                        'group_id': group_id,
                        'name': name,
                        'created_at': created_at_ts,
                    },
                }
            ],
        )

    def upsert_edge_vector(
        self,
        uuid: str,
        embedding: list[float],
        group_id: str,
        source_node_uuid: str,
        target_node_uuid: str,
        fact: str,
        created_at_ts: float,
    ) -> None:
        """Upsert a single edge fact embedding."""
        # Truncate fact for filterable metadata size limit
        truncated_fact = fact[:200] if fact else ''
        self.upsert_vectors(
            self.config.edge_index_name,
            [
                {
                    'key': uuid,
                    'data': embedding,
                    'metadata': {
                        'uuid': uuid,
                        'group_id': group_id,
                        'source_node_uuid': source_node_uuid,
                        'target_node_uuid': target_node_uuid,
                        'fact': truncated_fact,
                        'created_at': created_at_ts,
                    },
                }
            ],
        )

    def upsert_edge_source_vector(
        self,
        uuid: str,
        embedding: list[float],
        group_id: str,
        source_excerpt: str,
        created_at_ts: float,
    ) -> None:
        """Upsert a single edge source_excerpt embedding."""
        truncated_excerpt = source_excerpt[:200] if source_excerpt else ''
        self.upsert_vectors(
            self.config.edge_source_index_name,
            [
                {
                    'key': uuid,
                    'data': embedding,
                    'metadata': {
                        'uuid': uuid,
                        'group_id': group_id,
                        'source_excerpt': truncated_excerpt,
                        'created_at': created_at_ts,
                    },
                }
            ],
        )

    def upsert_community_vector(
        self,
        uuid: str,
        embedding: list[float],
        group_id: str,
        name: str,
        created_at_ts: float,
    ) -> None:
        """Upsert a single community node embedding."""
        self.upsert_vectors(
            self.config.community_index_name,
            [
                {
                    'key': uuid,
                    'data': embedding,
                    'metadata': {
                        'uuid': uuid,
                        'group_id': group_id,
                        'name': name,
                        'created_at': created_at_ts,
                    },
                }
            ],
        )

    def upsert_narrative_vector(
        self,
        key: str,
        embedding: list[float],
        group_id: str,
        episode_uuid: str,
        excerpt: str,
        created_at_ts: float,
    ) -> None:
        """Upsert a single episode narrative embedding."""
        truncated_excerpt = excerpt[:200] if excerpt else ''
        self.upsert_vectors(
            self.config.narrative_index_name,
            [
                {
                    'key': key,
                    'data': embedding,
                    'metadata': {
                        'group_id': group_id,
                        'episode_uuid': episode_uuid,
                        'excerpt': truncated_excerpt,
                        'created_at': created_at_ts,
                    },
                }
            ],
        )

    def upsert_describes_fact_vector(
        self,
        uuid: str,
        embedding: list[float],
        group_id: str,
        target_node_uuid: str,
        fact: str,
        created_at_ts: float,
    ) -> None:
        """Upsert a single DescribesEdge fact embedding."""
        truncated_fact = fact[:200] if fact else ''
        self.upsert_vectors(
            self.config.describes_fact_index_name,
            [
                {
                    'key': uuid,
                    'data': embedding,
                    'metadata': {
                        'uuid': uuid,
                        'group_id': group_id,
                        'target_node_uuid': target_node_uuid,
                        'fact': truncated_fact,
                        'created_at': created_at_ts,
                    },
                }
            ],
        )

    def upsert_describes_excerpt_vector(
        self,
        uuid: str,
        embedding: list[float],
        group_id: str,
        target_node_uuid: str,
        excerpt: str,
        created_at_ts: float,
    ) -> None:
        """Upsert a single DescribesEdge excerpt embedding."""
        truncated_excerpt = excerpt[:200] if excerpt else ''
        self.upsert_vectors(
            self.config.describes_excerpt_index_name,
            [
                {
                    'key': uuid,
                    'data': embedding,
                    'metadata': {
                        'uuid': uuid,
                        'group_id': group_id,
                        'target_node_uuid': target_node_uuid,
                        'excerpt': truncated_excerpt,
                        'created_at': created_at_ts,
                    },
                }
            ],
        )

    # ------------------------------------------------------------------ #
    #  Read: query vectors
    # ------------------------------------------------------------------ #

    def query_vectors(
        self,
        index_name: str,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Query an index for similar vectors.

        Parameters
        ----------
        index_name : str
            Index to search.
        query_vector : list[float]
            Query embedding.
        top_k : int
            Number of results.
        metadata_filter : dict | None
            S3 Vectors filter expression (e.g. {"group_id": "xxx"} or
            {"$and": [{"group_id": {"$eq": "xxx"}}, ...]}).

        Returns
        -------
        list[VectorSearchResult]
        """
        top_k = min(top_k, MAX_TOP_K)

        kwargs: dict[str, Any] = {
            'vectorBucketName': self.config.vector_bucket_name,
            'indexName': index_name,
            'queryVector': {'float32': query_vector},
            'topK': top_k,
            'returnMetadata': True,
            'returnDistance': True,
        }
        if metadata_filter:
            kwargs['filter'] = metadata_filter

        response = self.client.query_vectors(**kwargs)

        results = []
        for v in response.get('vectors', []):
            # S3 Vectors returns distance; convert to similarity score for cosine
            distance = v.get('distance', 0.0)
            score = 1.0 - distance  # cosine distance → cosine similarity
            results.append(
                VectorSearchResult(
                    key=v['key'],
                    score=score,
                    metadata=v.get('metadata', {}),
                )
            )
        return results

    def query_entity_vectors(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search entity name embeddings with optional group_id filter."""
        metadata_filter = None
        if group_ids:
            if len(group_ids) == 1:
                metadata_filter = {'group_id': {'$eq': group_ids[0]}}
            else:
                metadata_filter = {'group_id': {'$in': group_ids}}

        results = self.query_vectors(
            self.config.entity_index_name,
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return [r for r in results if r.score >= min_score]

    def query_edge_vectors(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        source_node_uuid: str | None = None,
        target_node_uuid: str | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search edge fact embeddings with optional filters."""
        filters: list[dict[str, Any]] = []

        if group_ids:
            if len(group_ids) == 1:
                filters.append({'group_id': {'$eq': group_ids[0]}})
            else:
                filters.append({'group_id': {'$in': group_ids}})

        if source_node_uuid:
            filters.append({'source_node_uuid': {'$eq': source_node_uuid}})

        if target_node_uuid:
            filters.append({'target_node_uuid': {'$eq': target_node_uuid}})

        metadata_filter = None
        if len(filters) == 1:
            metadata_filter = filters[0]
        elif len(filters) > 1:
            metadata_filter = {'$and': filters}

        results = self.query_vectors(
            self.config.edge_index_name,
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return [r for r in results if r.score >= min_score]

    def query_edge_source_vectors(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search edge source_excerpt embeddings with optional group_id filter."""
        metadata_filter = None
        if group_ids:
            if len(group_ids) == 1:
                metadata_filter = {'group_id': {'$eq': group_ids[0]}}
            else:
                metadata_filter = {'group_id': {'$in': group_ids}}

        results = self.query_vectors(
            self.config.edge_source_index_name,
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return [r for r in results if r.score >= min_score]

    def query_community_vectors(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search community name embeddings with optional group_id filter."""
        metadata_filter = None
        if group_ids:
            if len(group_ids) == 1:
                metadata_filter = {'group_id': {'$eq': group_ids[0]}}
            else:
                metadata_filter = {'group_id': {'$in': group_ids}}

        results = self.query_vectors(
            self.config.community_index_name,
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return [r for r in results if r.score >= min_score]

    def query_narrative_vectors(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search episode narrative embeddings with optional group_id filter."""
        metadata_filter = None
        if group_ids:
            if len(group_ids) == 1:
                metadata_filter = {'group_id': {'$eq': group_ids[0]}}
            else:
                metadata_filter = {'group_id': {'$in': group_ids}}

        results = self.query_vectors(
            self.config.narrative_index_name,
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return [r for r in results if r.score >= min_score]

    def query_describes_fact_vectors(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search DescribesEdge fact embeddings with optional group_id filter."""
        metadata_filter = None
        if group_ids:
            if len(group_ids) == 1:
                metadata_filter = {'group_id': {'$eq': group_ids[0]}}
            else:
                metadata_filter = {'group_id': {'$in': group_ids}}

        results = self.query_vectors(
            self.config.describes_fact_index_name,
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return [r for r in results if r.score >= min_score]

    def query_describes_excerpt_vectors(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search DescribesEdge excerpt embeddings with optional group_id filter."""
        metadata_filter = None
        if group_ids:
            if len(group_ids) == 1:
                metadata_filter = {'group_id': {'$eq': group_ids[0]}}
            else:
                metadata_filter = {'group_id': {'$in': group_ids}}

        results = self.query_vectors(
            self.config.describes_excerpt_index_name,
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return [r for r in results if r.score >= min_score]

    # ------------------------------------------------------------------ #
    #  Delete: remove vectors
    # ------------------------------------------------------------------ #

    def delete_vectors(self, index_name: str, keys: list[str]) -> None:
        """Delete vectors by key from an index."""
        if not keys:
            return
        for i in range(0, len(keys), MAX_BATCH_SIZE):
            batch = keys[i : i + MAX_BATCH_SIZE]
            self.client.delete_vectors(
                vectorBucketName=self.config.vector_bucket_name,
                indexName=index_name,
                keys=batch,
            )
        logger.debug('Deleted %d vectors from %s', len(keys), index_name)

    def delete_entity_vectors(self, uuids: list[str]) -> None:
        """Delete entity vectors by UUID."""
        self.delete_vectors(self.config.entity_index_name, uuids)

    def delete_edge_vectors(self, uuids: list[str]) -> None:
        """Delete edge vectors by UUID."""
        self.delete_vectors(self.config.edge_index_name, uuids)

    def delete_edge_source_vectors(self, uuids: list[str]) -> None:
        """Delete edge source_excerpt vectors by UUID."""
        self.delete_vectors(self.config.edge_source_index_name, uuids)

    def delete_community_vectors(self, uuids: list[str]) -> None:
        """Delete community vectors by UUID."""
        self.delete_vectors(self.config.community_index_name, uuids)

    def delete_narrative_vectors(self, keys: list[str]) -> None:
        """Delete episode narrative vectors by key."""
        self.delete_vectors(self.config.narrative_index_name, keys)

    def delete_describes_fact_vectors(self, uuids: list[str]) -> None:
        """Delete describes fact vectors by UUID."""
        self.delete_vectors(self.config.describes_fact_index_name, uuids)

    def delete_describes_excerpt_vectors(self, uuids: list[str]) -> None:
        """Delete describes excerpt vectors by UUID."""
        self.delete_vectors(self.config.describes_excerpt_index_name, uuids)
    def upsert_episode_content_vector(
        self,
        uuid: str,
        embedding: list[float],
        group_id: str,
        name: str,
        content_preview: str,
        created_at_ts: float,
    ) -> None:
        """Upsert a single episode content embedding for dedup detection."""
        self.upsert_vectors(
            self.config.episode_content_index_name,
            [
                {
                    'key': uuid,
                    'data': embedding,
                    'metadata': {
                        'uuid': uuid,
                        'group_id': group_id,
                        'name': name,
                        'content_preview': content_preview[:200] if content_preview else '',
                        'created_at': created_at_ts,
                    },
                }
            ],
        )

    def query_episode_content_vectors(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        top_k: int = 1,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search episode content embeddings for dedup detection."""
        metadata_filter = None
        if group_ids:
            if len(group_ids) == 1:
                metadata_filter = {'group_id': {'$eq': group_ids[0]}}
            else:
                metadata_filter = {'group_id': {'$in': group_ids}}

        results = self.query_vectors(
            self.config.episode_content_index_name,
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return [r for r in results if r.score >= min_score]

    def delete_episode_content_vectors(self, uuids: list[str]) -> None:
        """Delete episode content vectors by UUID."""
        self.delete_vectors(self.config.episode_content_index_name, uuids)



    # ------------------------------------------------------------------ #
    #  Utility: get vectors by key
    # ------------------------------------------------------------------ #

    def get_vectors(self, index_name: str, keys: list[str]) -> list[dict[str, Any]]:
        """Retrieve vectors by key (for MMR reranker that needs embeddings)."""
        if not keys:
            return []
        response = self.client.get_vectors(
            vectorBucketName=self.config.vector_bucket_name,
            indexName=index_name,
            keys=keys,
        )
        return response.get('vectors', [])
