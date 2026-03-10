"""
6. Verify content_blocks persistence: read back from Neo4j and validate.

This script is specific to docx-manual — verifies that content_blocks
(text + image blocks) were correctly serialized to and deserialized from Neo4j.

Usage:
    cd graphiti
    examples/docx-manual/run.sh 6
"""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from common import build_graphiti, GROUP_ID
from graphiti_core.nodes import ContentBlock, ContentBlockType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    graphiti = build_graphiti()
    try:
        print('--- Verifying content_blocks in Neo4j ---', flush=True)
        records, _, _ = await graphiti.driver.execute_query(
            "MATCH (e:Episodic) WHERE e.group_id = $gid "
            "RETURN e.name AS name, e.source AS source, "
            "e.content AS content, e.content_blocks AS content_blocks "
            "ORDER BY e.name",
            params={'gid': GROUP_ID},
        )

        total_blocks = 0
        total_text = 0
        total_image = 0
        total_table = 0

        for r in records:
            name = r['name']
            source = r['source']
            content = r.get('content', '')
            raw_blocks = r.get('content_blocks', '[]')

            # Parse content_blocks
            try:
                block_dicts = json.loads(raw_blocks) if isinstance(raw_blocks, str) else (raw_blocks or [])
                cb_list = [ContentBlock(**b) for b in block_dicts]
            except Exception as e:
                print(f'\n  {name}: PARSE ERROR: {e}', flush=True)
                continue

            n_text = sum(1 for b in cb_list if b.block_type == ContentBlockType.text)
            n_img = sum(1 for b in cb_list if b.block_type == ContentBlockType.image)
            n_tbl = sum(1 for b in cb_list if b.block_type == ContentBlockType.table)
            total_blocks += len(cb_list)
            total_text += n_text
            total_image += n_img
            total_table += n_tbl

            print(f'\n  {name}:', flush=True)
            print(f'    source: {source}', flush=True)
            print(f'    blocks: {len(cb_list)} ({n_text} text, {n_img} image, {n_tbl} table)', flush=True)
            print(f'    content preview: {content[:100]}...', flush=True)

            # Show each block
            for b in cb_list:
                if b.block_type == ContentBlockType.text:
                    print(f'      [{b.index}] text/{b.semantic_role.value}: {(b.text or "")[:80]}', flush=True)
                elif b.block_type == ContentBlockType.image:
                    s3 = f', s3={b.s3_uri}' if b.s3_uri else ', s3=MISSING'
                    print(f'      [{b.index}] image: mime={b.mime_type}{s3}, desc={b.description}', flush=True)
                elif b.block_type == ContentBlockType.table:
                    print(f'      [{b.index}] table: {b.description}, rows in text={len((b.text or "").splitlines())}', flush=True)

        print(f'\n--- Summary ---', flush=True)
        print(f'  Episodes: {len(records)}', flush=True)
        print(f'  Total blocks: {total_blocks} ({total_text} text, {total_image} image, {total_table} table)', flush=True)

        # Validation checks
        errors = []
        for r in records:
            raw_blocks = r.get('content_blocks', '[]')
            try:
                block_dicts = json.loads(raw_blocks) if isinstance(raw_blocks, str) else (raw_blocks or [])
                cb_list = [ContentBlock(**b) for b in block_dicts]
            except Exception:
                continue

            if r['source'] == 'document' and not cb_list:
                errors.append(f'{r["name"]}: document episode has empty content_blocks')
            for b in cb_list:
                if b.block_type == ContentBlockType.text and not b.text:
                    errors.append(f'{r["name"]}[{b.index}]: text block has no text')
                if b.block_type == ContentBlockType.image and not b.description:
                    errors.append(f'{r["name"]}[{b.index}]: image block has no description')
                if b.block_type == ContentBlockType.image and not b.s3_uri:
                    errors.append(f'{r["name"]}[{b.index}]: image block has no s3_uri')

        if errors:
            print(f'\n  ERRORS ({len(errors)}):', flush=True)
            for e in errors:
                print(f'    ✗ {e}', flush=True)
        else:
            print(f'\n  ✓ All validations passed', flush=True)

    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
