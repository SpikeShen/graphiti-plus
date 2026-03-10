"""
5. List DescribesEdges: episode → entity description edges with full content.

Shows which original text excerpts were attributed to specific entities,
along with the LLM-generated fact summary.

Usage:
    cd graphiti
    uv run python examples/sanguo/5_describes.py
"""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from common import build_graphiti, GROUP_ID
from graphiti_core.edges import DescribesEdge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _parse_json_list(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return []


async def main():
    graphiti = build_graphiti()
    try:
        print('--- DescribesEdges by Episode ---', flush=True)
        records, _, _ = await graphiti.driver.execute_query(
            "MATCH (e:Episodic) WHERE e.group_id = $gid "
            "RETURN e.name AS name, e.describes_edges AS describes_edges "
            "ORDER BY e.name",
            params={'gid': GROUP_ID},
        )
        total = 0
        for r in records:
            de_uuids = _parse_json_list(r.get('describes_edges'))
            if not de_uuids:
                print(f'\n  {r["name"]}: 0 describes edges', flush=True)
                continue

            edges = await DescribesEdge.get_by_uuids(graphiti.driver, de_uuids)
            total += len(edges)
            print(f'\n  {r["name"]}: {len(edges)} describes edges', flush=True)

            # Build entity name lookup for target nodes
            entity_uuids = list({e.target_node_uuid for e in edges})
            name_records, _, _ = await graphiti.driver.execute_query(
                "MATCH (n:Entity) WHERE n.uuid IN $uuids RETURN n.uuid AS uuid, n.name AS name",
                uuids=entity_uuids,
                routing_='r',
            )
            uuid_to_name = {nr['uuid']: nr['name'] for nr in name_records}

            for idx, edge in enumerate(edges):
                entity_name = uuid_to_name.get(edge.target_node_uuid, '?')
                print(f'    [{idx}] → {entity_name}', flush=True)
                print(f'        excerpt: {edge.excerpt[:100]}', flush=True)
                print(f'        fact:    {edge.fact[:100]}', flush=True)

        print(f'\nTotal: {total} describes edges across {len(records)} episodes', flush=True)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
