"""
S3 multimodal asset storage.

Uploads binary content (images, audio, video, etc.) to a dedicated S3 bucket,
separate from the S3 Vectors bucket. ContentBlocks get their s3_uri populated
after upload, and _raw_bytes is cleared.

Bucket structure:
    s3://{bucket}/{group_id}/{episode_uuid}/block_{index:03d}.{ext}
"""

import logging
import os

import boto3
from pydantic import BaseModel, Field

from ..nodes import ContentBlock

logger = logging.getLogger(__name__)

# MIME type → file extension mapping
_MIME_TO_EXT: dict[str, str] = {
    'image/jpeg': 'jpeg',
    'image/jpg': 'jpg',
    'image/png': 'png',
    'image/gif': 'gif',
    'image/svg+xml': 'svg',
    'image/webp': 'webp',
    'image/tiff': 'tiff',
    'image/bmp': 'bmp',
    'audio/wav': 'wav',
    'audio/mp3': 'mp3',
    'audio/mpeg': 'mp3',
    'video/mp4': 'mp4',
    'video/webm': 'webm',
}


def _ext_from_mime(mime_type: str | None) -> str:
    """Derive file extension from MIME type."""
    if mime_type and mime_type in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime_type]
    if mime_type and '/' in mime_type:
        return mime_type.split('/')[-1]
    return 'bin'


class MultimodalStorageConfig(BaseModel):
    bucket: str = Field(description='S3 bucket name for multimodal assets')
    region: str = Field(default='us-east-1')


class MultimodalAssetStorage:
    """Upload and manage multimodal assets in S3."""

    def __init__(self, config: MultimodalStorageConfig | None = None):
        if config is None:
            bucket = os.environ.get('MULTIMODAL_ASSET_BUCKET')
            if not bucket:
                raise ValueError(
                    'MULTIMODAL_ASSET_BUCKET environment variable is required '
                    'for multimodal asset storage'
                )
            region = os.environ.get('AWS_REGION', 'us-east-1')
            config = MultimodalStorageConfig(bucket=bucket, region=region)
        self.config = config
        self.s3 = boto3.client('s3', region_name=config.region)

    def _build_key(self, group_id: str, episode_uuid: str, block: ContentBlock) -> str:
        ext = _ext_from_mime(block.mime_type)
        return f'{group_id}/{episode_uuid}/block_{block.index:03d}.{ext}'

    async def upload_blocks(
        self,
        blocks: list[ContentBlock],
        group_id: str,
        episode_uuid: str,
        clear_raw_bytes: bool = True,
    ) -> list[ContentBlock]:
        """Upload binary blocks to S3 and populate s3_uri.

        Only blocks with is_binary=True and _raw_bytes != None are uploaded.

        Parameters
        ----------
        clear_raw_bytes : bool
            If True (default), ``_raw_bytes`` is set to None after upload to
            free memory.  Pass False when downstream steps (e.g. vision LLM
            extraction) still need the raw bytes — the caller is then
            responsible for clearing ``_raw_bytes`` later.

        Returns the same list with s3_uri populated.
        """
        for block in blocks:
            if not block.is_binary or block._raw_bytes is None:
                continue

            key = self._build_key(group_id, episode_uuid, block)
            content_type = block.mime_type or 'application/octet-stream'

            self.s3.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=block._raw_bytes,
                ContentType=content_type,
            )

            block.s3_uri = f's3://{self.config.bucket}/{key}'
            size = len(block._raw_bytes)
            if clear_raw_bytes:
                block._raw_bytes = None  # Free memory

            logger.info(
                'Uploaded block %d (%s, %d bytes) → %s',
                block.index, content_type, size, block.s3_uri,
            )

        return blocks

    async def delete_episode_assets(self, group_id: str, episode_uuid: str) -> int:
        """Delete all assets for an episode. Returns number of objects deleted."""
        prefix = f'{group_id}/{episode_uuid}/'
        response = self.s3.list_objects_v2(
            Bucket=self.config.bucket, Prefix=prefix,
        )
        objects = response.get('Contents', [])
        if not objects:
            return 0

        self.s3.delete_objects(
            Bucket=self.config.bucket,
            Delete={'Objects': [{'Key': obj['Key']} for obj in objects]},
        )
        logger.info('Deleted %d assets for episode %s', len(objects), episode_uuid)
        return len(objects)

    async def delete_group_assets(self, group_id: str) -> int:
        """Delete all assets for a group. Returns number of objects deleted."""
        prefix = f'{group_id}/'
        total = 0
        continuation_token = None

        while True:
            kwargs: dict = {'Bucket': self.config.bucket, 'Prefix': prefix}
            if continuation_token:
                kwargs['ContinuationToken'] = continuation_token

            response = self.s3.list_objects_v2(**kwargs)
            objects = response.get('Contents', [])
            if not objects:
                break

            self.s3.delete_objects(
                Bucket=self.config.bucket,
                Delete={'Objects': [{'Key': obj['Key']} for obj in objects]},
            )
            total += len(objects)

            if not response.get('IsTruncated'):
                break
            continuation_token = response.get('NextContinuationToken')

        if total:
            logger.info('Deleted %d assets for group %s', total, group_id)
        return total

    def generate_presigned_url(self, s3_uri: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for an S3 asset."""
        # Parse s3://bucket/key
        if not s3_uri.startswith('s3://'):
            raise ValueError(f'Invalid S3 URI: {s3_uri}')
        parts = s3_uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f'Invalid S3 URI: {s3_uri}')
        bucket, key = parts
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expires_in,
        )
