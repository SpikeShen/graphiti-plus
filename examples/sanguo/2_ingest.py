"""
2. Ingest 三国演义 Chapter 1 paragraphs.

Usage:
    cd graphiti
    # Ingest first paragraph only:
    uv run python tests/sanguo/2_ingest.py 1
    # Ingest first 3 paragraphs:
    uv run python tests/sanguo/2_ingest.py 3
    # Ingest all paragraphs (default):
    uv run python tests/sanguo/2_ingest.py
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from common import build_graphiti, load_chapter1_paragraphs, ingest_paragraphs, print_graph_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    paragraphs = load_chapter1_paragraphs(n)
    print(f'Loaded {len(paragraphs)} paragraphs', flush=True)
    for i, p in enumerate(paragraphs):
        print(f'  [{i}] {p[:60]}...', flush=True)

    graphiti = build_graphiti()
    try:
        await ingest_paragraphs(graphiti, paragraphs)
        await print_graph_stats(graphiti)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
