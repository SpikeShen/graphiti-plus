"""
5. List episode narratives persisted in Neo4j Episodic nodes.

Usage:
    cd graphiti
    examples/docx-manual/run.sh 5
"""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from common import build_graphiti, GROUP_ID

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
            return [s.strip() for s in raw.split('|') if s.strip()]
    return []


async def main():
    graphiti = build_graphiti()
    try:
        print('--- Episode Narratives in Neo4j ---', flush=True)
        records, _, _ = await graphiti.driver.execute_query(
            "MATCH (e:Episodic) WHERE e.group_id = $gid "
            "RETURN e.name AS name, e.narrative_excerpts AS narrative_excerpts "
            "ORDER BY e.name",
            params={'gid': GROUP_ID},
        )
        total = 0
        for r in records:
            ue_list = _parse_json_list(r.get('narrative_excerpts'))
            total += len(ue_list)
            print(f'\n  {r["name"]}: {len(ue_list)} episode narratives', flush=True)
            for idx, ue in enumerate(ue_list):
                if isinstance(ue, dict):
                    print(f'    [{idx}] {ue.get("excerpt", str(ue))[:120]}', flush=True)
                else:
                    print(f'    [{idx}] {str(ue)[:120]}', flush=True)
        print(f'\nTotal: {total} episode narratives across {len(records)} episodes', flush=True)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
