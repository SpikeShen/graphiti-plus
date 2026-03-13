"""
1. Clear Neo4j data and rebuild S3 Vectors indices.

Usage:
    cd graphiti
    examples/audio-extraction/run.sh 1
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from common import build_graphiti, clear_all

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    graphiti = build_graphiti()
    try:
        await clear_all(graphiti)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
