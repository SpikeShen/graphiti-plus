"""
2. Parse docx and ingest sections as document-type episodes with content_blocks.

Usage:
    cd graphiti
    examples/docx-manual/run.sh 2        # ingest all sections
    examples/docx-manual/run.sh 2 3      # ingest first 3 sections only
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from common import (
    DOCX_PATH,
    build_graphiti,
    ingest_sections,
    parse_docx,
    print_graph_stats,
    split_blocks_by_section,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print(f'Parsing: {DOCX_PATH}', flush=True)
    blocks = parse_docx(DOCX_PATH)
    print(f'  Total blocks: {len(blocks)}', flush=True)
    for b in blocks:
        tag = f'[{b.block_type.value}:{b.semantic_role.value}]'
        preview = (b.text or '')[:60] if b.block_type.value == 'text' else (b.description or '')
        print(f'    {b.index:3d} {tag:25s} {preview}', flush=True)

    sections = split_blocks_by_section(blocks)
    if n:
        sections = sections[:n]
    print(f'\nSections to ingest: {len(sections)}', flush=True)
    for i, (name, sblocks) in enumerate(sections):
        n_text = sum(1 for b in sblocks if b.block_type.value == 'text')
        n_img = sum(1 for b in sblocks if b.block_type.value == 'image')
        print(f'  [{i}] {name[:50]} ({n_text} text, {n_img} img)', flush=True)

    graphiti = build_graphiti()
    try:
        await ingest_sections(graphiti, sections)
        await print_graph_stats(graphiti)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
