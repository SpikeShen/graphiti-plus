"""
3. Search tests: standard search + deep search for MAP CCS Tagging content.

Usage:
    cd graphiti
    examples/docx-manual/run.sh 3
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from common import build_graphiti, GROUP_ID
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_RRF,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def print_results(label: str, results: SearchResults):
    print(f'\n--- {label} ---', flush=True)
    print(f'  Edges: {len(results.edges)}', flush=True)
    for edge in results.edges:
        print(f'    {edge.fact}', flush=True)
    if results.nodes:
        print(f'  Nodes: {len(results.nodes)}', flush=True)
        for node in results.nodes:
            print(f'    {node.name}', flush=True)


async def main():
    graphiti = build_graphiti()
    try:
        std_config = EDGE_HYBRID_SEARCH_RRF.model_copy(update={'limit': 10})

        # Standard search queries relevant to the MAP CCS Tagging manual
        for query in [
            'CCS Tagging是什么？',
            '如何激活map-migrated标签？',
            '打标签的前缀规则是什么？',
        ]:
            results = await graphiti.search_(query, config=std_config, group_ids=[GROUP_ID])
            print_results(f'Standard: {query}', results)

        # Deep search comparison
        query = '如何用Tag Editor批量打标签？'
        combined_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={'limit': 10})
        std_results = await graphiti.search_(query, config=combined_config, group_ids=[GROUP_ID])
        deep_results = await graphiti.search_(
            query,
            config=combined_config.model_copy(update={'limit': 20}),
            group_ids=[GROUP_ID],
            deep_search=True,
        )
        print_results(f'Standard (combined): {query}', std_results)
        print_results(f'Deep (combined + deep_search=True): {query}', deep_results)

        # Episode narratives from deep search
        if deep_results.narrative_excerpts:
            print(f'\n--- Episode Narratives (deep search) ---', flush=True)
            print(f'  Total: {len(deep_results.narrative_excerpts)}', flush=True)
            for ue in deep_results.narrative_excerpts:
                print(f'    [score={ue["score"]:.3f}] {ue["excerpt"]}', flush=True)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
