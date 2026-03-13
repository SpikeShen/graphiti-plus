"""
3. Search tests: standard + deep search for night-flight system content.

Usage:
    cd graphiti
    examples/audio-extraction/run.sh 3
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
        excerpt = (edge.source_excerpt or '')[:80]
        print(f'    fact: {edge.fact}', flush=True)
        if excerpt:
            print(f'      excerpt: {excerpt}', flush=True)
    if results.nodes:
        print(f'  Nodes: {len(results.nodes)}', flush=True)
        for node in results.nodes:
            print(f'    {node.name}: {(node.summary or "")[:80]}', flush=True)
    if results.narrative_excerpts:
        print(f'  Narratives: {len(results.narrative_excerpts)}', flush=True)
        for ne in results.narrative_excerpts[:5]:
            print(f'    [score={ne["score"]:.3f}] {ne["excerpt"][:80]}', flush=True)


async def main():
    graphiti = build_graphiti()
    try:
        std_config = EDGE_HYBRID_SEARCH_RRF.model_copy(update={'limit': 10})
        combined_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={'limit': 10})

        # Standard search queries
        queries = [
            '夜航灯有什么特点？',
            '散热系统是怎么设计的？',
            '防水等级是多少？',
            '视角有多宽？',
            '一体化设计包含哪些部分？',
        ]
        for query in queries:
            results = await graphiti.search_(query, config=std_config, group_ids=[GROUP_ID])
            print_results(f'Standard: {query}', results)

        # Deep search comparison
        print('\n' + '=' * 60, flush=True)
        print('Deep Search Comparison', flush=True)
        print('=' * 60, flush=True)

        query = '夜航灯如何实现夜间仿地作业？'
        std_results = await graphiti.search_(query, config=combined_config, group_ids=[GROUP_ID])
        deep_results = await graphiti.search_(
            query,
            config=combined_config.model_copy(update={'limit': 20}),
            group_ids=[GROUP_ID],
            deep_search=True,
        )
        print_results(f'Standard (combined): {query}', std_results)
        print_results(f'Deep (combined + deep_search=True): {query}', deep_results)

    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
