"""
Bedrock Nova Multimodal Embeddings client for Graphiti.

Uses boto3 bedrock-runtime invoke_model API with the
amazon.nova-2-multimodal-embeddings-v1:0 model.
"""

import base64
import json
import logging
from collections.abc import Iterable
from time import time

import boto3

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = 'amazon.nova-2-multimodal-embeddings-v1:0'
DEFAULT_EMBEDDING_DIM = 1024


class BedrockNovaEmbedderConfig(EmbedderConfig):
    model_id: str = DEFAULT_MODEL_ID
    region_name: str = 'us-east-1'
    embedding_dim: int = DEFAULT_EMBEDDING_DIM


class BedrockNovaEmbedder(EmbedderClient):
    """
    Embedder using Amazon Nova Multimodal Embeddings via Bedrock invoke_model API.

    Uses IAM credentials from the environment (boto3 default credential chain).
    Supports embedding dimensions: 256, 384, 1024, 3072.
    """

    def __init__(self, config: BedrockNovaEmbedderConfig | None = None):
        if config is None:
            config = BedrockNovaEmbedderConfig()
        self.config = config
        self.bedrock = boto3.client('bedrock-runtime', region_name=config.region_name)
        self.s3_logger = None  # Optional S3InvocationLogger

    def set_s3_logger(self, s3_logger) -> None:
        """Set the S3 invocation logger for this embedder."""
        self.s3_logger = s3_logger

    def _build_request_body(self, text: str) -> dict:
        return {
            'schemaVersion': 'nova-multimodal-embed-v1',
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': {
                'embeddingPurpose': 'GENERIC_INDEX',
                'embeddingDimension': self.config.embedding_dim,
                'text': {
                    'truncationMode': 'END',
                    'value': text,
                },
            },
        }

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], str):
            text = input_data[0]
        else:
            raise ValueError(f'Unsupported input type for Nova embedder: {type(input_data)}')

        body = json.dumps(self._build_request_body(text))

        try:
            _start_time = time()
            response = self.bedrock.invoke_model(
                modelId=self.config.model_id,
                body=body,
                contentType='application/json',
                accept='application/json',
            )
            result = json.loads(response['body'].read())
            embeddings = result.get('embeddings', [])
            # Nova MME returns token count in HTTP header, not response body
            input_tokens = int(
                response.get('ResponseMetadata', {})
                .get('HTTPHeaders', {})
                .get('x-amzn-bedrock-input-token-count', 0)
            )
            if embeddings:
                _latency = (time() - _start_time) * 1000
                # Record to S3 logger
                if self.s3_logger:
                    self.s3_logger.record(
                        operation='embedding.create',
                        model_id=self.config.model_id,
                        input_tokens=input_tokens,
                        latency_ms=_latency,
                        input_preview=text,
                    )
                return embeddings[0]['embedding'][: self.config.embedding_dim]
            raise ValueError(f'No embeddings in Nova response: {result}')
        except Exception as e:
            logger.error(f'Bedrock Nova embedding error: {e}')
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        # Nova MME doesn't support batch in a single call, so we call sequentially
        results = []
        for text in input_data_list:
            embedding = await self.create(text)
            results.append(embedding)
        return results

    def _build_image_request_body(
        self,
        image_bytes: bytes,
        image_format: str,
        text: str | None = None,
    ) -> dict:
        """Build Nova MME request body for image embedding.

        Note: Nova MME requires exactly one modality per request
        (text, image, audio, or video). Combined image+text in a single
        request is NOT supported. The `text` parameter is accepted for
        interface compatibility but ignored — image-only embedding is used.
        Text and image vectors share the same 1024-dim semantic space,
        so cross-modal retrieval works without combined embeddings.
        """
        params: dict = {
            'embeddingPurpose': 'GENERIC_INDEX',
            'embeddingDimension': self.config.embedding_dim,
            'image': {
                'format': image_format,
                'source': {
                    'bytes': base64.b64encode(image_bytes).decode('utf-8'),
                },
            },
        }
        # Nova MME: exactly one of text/image/audio/video per request.
        # text parameter is intentionally NOT added here.
        return {
            'schemaVersion': 'nova-multimodal-embed-v1',
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': params,
        }

    async def create_image(
        self,
        image_bytes: bytes,
        image_format: str = 'jpeg',
        text: str | None = None,
    ) -> list[float]:
        """Generate embedding for an image, optionally with text context.

        Uses Nova MME's native multimodal embedding. The resulting vector
        lives in the same 1024-dim semantic space as text embeddings,
        enabling cross-modal retrieval (image query → text results and vice versa).
        """
        body = json.dumps(self._build_image_request_body(image_bytes, image_format, text))

        try:
            _start_time = time()
            response = self.bedrock.invoke_model(
                modelId=self.config.model_id,
                body=body,
                contentType='application/json',
                accept='application/json',
            )
            result = json.loads(response['body'].read())
            embeddings = result.get('embeddings', [])
            input_tokens = int(
                response.get('ResponseMetadata', {})
                .get('HTTPHeaders', {})
                .get('x-amzn-bedrock-input-token-count', 0)
            )
            if embeddings:
                _latency = (time() - _start_time) * 1000
                if self.s3_logger:
                    self.s3_logger.record(
                        operation='embedding.create_image',
                        model_id=self.config.model_id,
                        input_tokens=input_tokens,
                        latency_ms=_latency,
                        input_preview=f'[image:{image_format},{len(image_bytes)}b]'
                                      + (f' + text:{text[:50]}' if text else ''),
                    )
                return embeddings[0]['embedding'][: self.config.embedding_dim]
            raise ValueError(f'No embeddings in Nova image response: {result}')
        except Exception as e:
            logger.error(f'Bedrock Nova image embedding error: {e}')
            raise
