"""
Preprocessing module for multimodal document parsing.

Provides document parsers, S3 asset storage, and image description generation.
"""

from .asset_storage import MultimodalAssetStorage, MultimodalStorageConfig
from .description import generate_image_descriptions
from .parser import DocumentParser, DocumentParserRegistry, default_registry
from .word_parser import WordDocumentParser

__all__ = [
    'DocumentParser',
    'DocumentParserRegistry',
    'MultimodalAssetStorage',
    'MultimodalStorageConfig',
    'WordDocumentParser',
    'default_registry',
    'generate_image_descriptions',
]
