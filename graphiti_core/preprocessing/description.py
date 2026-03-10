"""
Image description generation using vision-capable LLMs.

Calls the LLM with base64-encoded images to generate text descriptions,
which are stored in ContentBlock.description for:
1. Text representation fallback (non-vision LLMs)
2. Search result display
3. content field text representation
"""

import base64
import json
import logging

from ..llm_client.client import LLMClient
from ..llm_client.config import ModelSize
from ..nodes import ContentBlock, ContentBlockType
from ..prompts.models import Message

logger = logging.getLogger(__name__)

IMAGE_DESCRIPTION_PROMPT = (
    '请用一段简洁的中文描述这张图片的内容。'
    '如果图片是操作界面截图，请描述界面中的关键元素、操作步骤和重要信息。'
    '如果图片包含文字，请提取关键文字内容。'
    '回复格式为纯 JSON：{"description": "..."}'
)


async def generate_image_descriptions(
    llm_client: LLMClient,
    blocks: list[ContentBlock],
    group_id: str | None = None,
) -> list[ContentBlock]:
    """Generate text descriptions for image-type ContentBlocks.

    Uses the LLM's vision capability to describe each image.
    Populates block.description for blocks that have _raw_bytes.
    Blocks that already have a description are skipped.

    Returns the same list with descriptions populated.
    """
    image_blocks = [
        b for b in blocks
        if b.block_type == ContentBlockType.image
        and b._raw_bytes is not None
        and not b.description  # Skip if already described
    ]

    if not image_blocks:
        return blocks

    logger.info('Generating descriptions for %d image blocks', len(image_blocks))

    for block in image_blocks:
        try:
            b64 = base64.b64encode(block._raw_bytes).decode('utf-8')
            mime = block.mime_type or 'image/png'

            # Build multimodal message with image + text prompt
            # Using OpenAI-compatible vision format (works with Bedrock Mantle)
            messages = [
                Message(role='system', content='你是一个图片描述助手。'),
                Message(
                    role='user',
                    content=[
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:{mime};base64,{b64}',
                            },
                        },
                        {
                            'type': 'text',
                            'text': IMAGE_DESCRIPTION_PROMPT,
                        },
                    ],
                ),
            ]

            try:
                result = await llm_client.generate_response(
                    messages,
                    max_tokens=512,
                    model_size=ModelSize.medium,
                    group_id=group_id,
                    prompt_name='generate_image_description',
                )
                # Result should be {"description": "..."}
                if isinstance(result, dict) and 'description' in result:
                    block.description = result['description']
                elif isinstance(result, dict):
                    # Take the first string value
                    for v in result.values():
                        if isinstance(v, str) and v:
                            block.description = v
                            break
            except NotImplementedError:
                # LLM client doesn't support multimodal — use metadata fallback
                logger.info(
                    'LLM client does not support multimodal messages, '
                    'using metadata fallback for block %d', block.index,
                )
                block.description = _metadata_fallback(block)
            except Exception as e:
                logger.warning(
                    'Vision LLM failed for block %d, using fallback: %s',
                    block.index, e,
                )
                block.description = _metadata_fallback(block)

            if not block.description:
                block.description = _metadata_fallback(block)

            logger.debug(
                'Generated description for block %d: %s',
                block.index, block.description[:80],
            )

        except Exception as e:
            logger.warning(
                'Failed to generate description for block %d: %s',
                block.index, e,
            )
            if not block.description:
                block.description = f'[图片: block_{block.index}]'

    logger.info('Completed %d image descriptions', len(image_blocks))
    return blocks


def _metadata_fallback(block: ContentBlock) -> str:
    """Generate a descriptive placeholder from metadata."""
    fmt = block.metadata.get('format', 'unknown') if block.metadata else 'unknown'
    size = block.metadata.get('size_bytes', 0) if block.metadata else 0
    return f'文档内嵌图片（格式: {fmt}, 大小: {size} bytes）'
