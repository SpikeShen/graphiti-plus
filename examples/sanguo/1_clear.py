"""
1. Clear all data (Neo4j + S3 Vectors).

Usage:
    cd graphiti
    uv run python tests/sanguo/1_clear.py
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
        print('Done.', flush=True)
    finally:
        await graphiti.close()


if __name__ == '__main__':
    asyncio.run(main())
