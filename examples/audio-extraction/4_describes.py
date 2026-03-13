"""
4. Show DescribesEdge details for key entities.

Usage:
    cd graphiti
    examples/audio-extraction/run.sh 4
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from common import build_graphiti, GROUP_ID

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    graphiti = build_graphiti()
    try:
        # Get all entities
        records, _, _ = await graphiti.driver.execute_query(
            "MATCH (n:Entity) WHERE n.group_id = $gid "
            "RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary "
            "ORDER BY n.name",
            params={'gid': GROUP_ID},
        )
        print(f'--- Entities ({len(records)}) ---', flush=True)
        for r in records:
            print(f'  {r["name"]}: {(r["summary"] or "")[:80]}', flush=True)

        # Get DescribesEdges for each entity
        print(f'\n--- DescribesEdges ---', flush=True)
        records, _, _ = await graphiti.driver.execute_query(
            "MATCH (ep:Episodic)-[d:DESCRIBES]->(en:Entity) "
            "WHERE d.group_id = $gid "
            "RETURN en.name AS entity_name, d.fact AS fact, d.excerpt AS excerpt, "
            "       ep.name AS episode_name "
            "ORDER BY en.name, ep.name",
            params={'gid': GROUP_ID},
        )
        current_entity = None
        for r in records:
            if r['entity_name'] != current_entity:
                current_entity = r['entity_name']
                print(f'\n  Entity: {current_entity}', flush=True)
            print(f'    [{r["episode_name"]}]', flush=True)
            print(f'      fact: {r["fact"]}', flush=True)
            print(f'      excerpt: {(r["excerpt"] or "")[:120]}', flush=True)

        if not records:
            print('  (no DescribesEdges found)', flush=True)

    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
