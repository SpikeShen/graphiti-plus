"""
3. Search tests: standard search + deep search comparison.

- '张角是什么人？' and '黄巾起义的原因': standard search only
- '念咒者何人': standard vs deep search comparison (tests source_excerpt + episode narratives)

Usage:
    cd graphiti
    uv run python examples/sanguo/3_search.py
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

        # Standard search (edge only)
        for query in ['张角是什么人？', '黄巾起义的原因']:
            results = await graphiti.search_(query, config=std_config, group_ids=[GROUP_ID])
            print_results(f'Standard: {query}', results)

        # Deep search comparison
        # Use COMBINED config (edge + node) so deep_search can add
        # source_similarity to both — DescribesEdge excerpts are searched
        # via node source_similarity (describes-excerpt-embeddings → EntityNode)
        query = '念咒者何人'
        combined_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={'limit': 10})
        std_results = await graphiti.search_(query, config=combined_config, group_ids=[GROUP_ID])
        deep_results = await graphiti.search_(
            query, config=combined_config.model_copy(update={'limit': 20}),
            group_ids=[GROUP_ID], deep_search=True,
        )
        print_results(f'Standard (combined): {query}', std_results)
        print_results(f'Deep (combined + deep_search=True): {query}', deep_results)

        # For nodes returned by deep search, fetch their DescribesEdges
        # to show the full source tracing chain
        from graphiti_core.edges import DescribesEdge
        if deep_results.nodes:
            print(f'\n--- DescribesEdges for deep search nodes ---', flush=True)
            for node in deep_results.nodes[:5]:
                des = await DescribesEdge.get_by_entity_uuid(graphiti.driver, node.uuid)
                if des:
                    print(f'\n  {node.name}: {len(des)} describes edges', flush=True)
                    for idx, d in enumerate(des):
                        print(f'    [{idx}] → {node.name}', flush=True)
                        print(f'        excerpt: {d.excerpt[:100]}', flush=True)
                        print(f'        fact:    {d.fact[:100]}', flush=True)

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
