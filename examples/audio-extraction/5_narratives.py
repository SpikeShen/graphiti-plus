"""
5. Show episode narrative excerpts (text not captured as entities/edges).

Usage:
    cd graphiti
    examples/audio-extraction/run.sh 5
"""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from common import build_graphiti, GROUP_ID

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    graphiti = build_graphiti()
    try:
        records, _, _ = await graphiti.driver.execute_query(
            "MATCH (e:Episodic) WHERE e.group_id = $gid "
            "RETURN e.name AS name, e.narrative_excerpts AS narrative_excerpts "
            "ORDER BY e.name",
            params={'gid': GROUP_ID},
        )
        print(f'--- Episode Narratives ---', flush=True)
        total = 0
        for r in records:
            raw = r.get('narrative_excerpts', '[]')
            try:
                narr_list = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                narr_list = []
            if narr_list:
                print(f'\n  Episode: {r["name"]} ({len(narr_list)} narratives)', flush=True)
                for ne in narr_list:
                    print(f'    • {ne}', flush=True)
                total += len(narr_list)

        print(f'\n  Total narrative excerpts: {total}', flush=True)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
