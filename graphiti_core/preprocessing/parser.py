"""
Document parser base class and registry.

New document types are supported by implementing DocumentParser
and registering with DocumentParserRegistry.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ..nodes import ContentBlock


class DocumentParser(ABC):
    """Base class for document parsers that convert files to ordered ContentBlock lists."""

    @abstractmethod
    async def parse(self, file_path: str) -> list[ContentBlock]:
        """Parse a document into ordered content blocks.

        Binary content (images, etc.) is stored in ContentBlock._raw_bytes.
        The caller is responsible for uploading to S3 and populating s3_uri.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions (e.g. ['.docx'])."""
        ...


class DocumentParserRegistry:
    """Registry that routes file extensions to the appropriate parser."""

    def __init__(self):
        self._parsers: dict[str, DocumentParser] = {}

    def register(self, parser: DocumentParser):
        for ext in parser.supported_extensions():
            self._parsers[ext.lower()] = parser

    def get_parser(self, file_path: str) -> DocumentParser | None:
        ext = Path(file_path).suffix.lower()
        return self._parsers.get(ext)


# Lazy-initialized default registry (populated after WordDocumentParser import)
default_registry = DocumentParserRegistry()
