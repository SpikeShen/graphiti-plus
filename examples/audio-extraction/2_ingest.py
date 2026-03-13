"""
2. Load transcript and ingest paragraphs into Graphiti.

Usage:
    cd graphiti
    examples/audio-extraction/run.sh 2        # ingest all paragraphs
    examples/audio-extraction/run.sh 2 2      # ingest first 2 paragraphs only
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from common import build_graphiti, ingest_paragraphs, load_transcript_paragraphs, print_graph_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None

    paragraphs = load_transcript_paragraphs()
    if n:
        paragraphs = paragraphs[:n]

    print(f'Paragraphs to ingest: {len(paragraphs)}', flush=True)
    for i, p in enumerate(paragraphs):
        print(f'  [{i}] ({len(p)} chars) {p[:80]}...', flush=True)

    graphiti = build_graphiti()
    try:
        await ingest_paragraphs(graphiti, paragraphs)
        await print_graph_stats(graphiti)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
