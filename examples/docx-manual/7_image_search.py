"""
7. Image-based search: use an image as query to find related content via cross-modal
   vector search (Nova MME image embedding → cosine similarity against text/image vectors).

Search target:
    test-image-query.png — a partial screenshot cropped from the Cost Explorer "按标签分组"
    page in section 7 of the docx manual. The original full image is block_005.png
    (文档第25个内容块), which shows "点击Cost Explorer, 根据标签分组，选择摊销成本".

Expected results:
    - describes-excerpt-embeddings should rank the original image (block_005.png) highest
      because it's the same visual content. This image is attached to a DescribesEdge
      pointing to the "Cost Explorer" entity.
    - Other Cost Explorer screenshots (block_003.png, block_007.png) from the same section
      should also score high since they share similar AWS console UI elements.
    - The deep search should return "Cost Explorer" in the nodes list (via
      s3_vectors_node_source_similarity_search → describes-excerpt-embeddings).
    - Narrative excerpts with image references should appear with higher scores than
      pure-text narratives, confirming cross-modal embedding works.

Indices searched:
    Standard search:
      - edge-fact-embeddings (cosine_similarity) → RELATES_TO edges
      - entity-name-embeddings (cosine_similarity) → Entity nodes
    Deep search adds:
      - edge-source-embeddings (source_similarity) → RELATES_TO edges via source_excerpt
      - describes-excerpt-embeddings (source_similarity) → Entity nodes via DescribesEdge
      - episode-narrative-embeddings → narrative excerpts

Usage:
    cd graphiti
    examples/docx-manual/run.sh 7 [/path/to/image.png]
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from common import build_graphiti, GROUP_ID
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), 'test-image-query.png')

# Vector-only config (no BM25/fulltext — empty query can't drive Lucene fulltext)
IMAGE_SEARCH_CONFIG = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.mmr,
        mmr_lambda=0.9,
        sim_min_score=0.0,
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.mmr,
        mmr_lambda=0.9,
        sim_min_score=0.0,
    ),
)


def print_results(label: str, results: SearchResults):
    print(f'\n--- {label} ---', flush=True)
    if results.edges:
        print(f'  Edges: {len(results.edges)}', flush=True)
        for edge in results.edges:
            has_img = '[image:' in (edge.source_excerpt or '')
            tag = ' 📷' if has_img else ''
            print(f'    [fact] {edge.fact}{tag}', flush=True)
            if edge.source_excerpt:
                preview = edge.source_excerpt[:120].replace('\n', ' ')
                print(f'    [excerpt] {preview}', flush=True)
    if results.nodes:
        print(f'  Nodes: {len(results.nodes)}', flush=True)
        for node in results.nodes:
            print(f'    {node.name}', flush=True)
    if results.narrative_excerpts:
        print(f'  Narratives: {len(results.narrative_excerpts)}', flush=True)
        for ne in results.narrative_excerpts:
            has_img = '[image:' in ne['excerpt']
            tag = ' 📷' if has_img else ''
            preview = ne['excerpt'][:120].replace('\n', ' ')
            print(f'    [score={ne["score"]:.3f}] {preview}{tag}', flush=True)


def print_raw_index(label, results):
    """Print raw S3 Vectors query results."""
    print(f'\n--- Raw: {label} (top 5) ---', flush=True)
    if not results:
        print('  (empty)', flush=True)
        return
    for r in results:
        meta = r.metadata or {}
        preview = (
            meta.get('fact', '') or
            meta.get('source_excerpt_preview', '') or
            meta.get('name', '') or
            meta.get('excerpt', '') or
            meta.get('content_preview', '') or
            ''
        )[:100]
        print(f'  score={r.score:.4f} | {preview}', flush=True)


async def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not os.path.isfile(image_path):
        print(f'Error: image not found: {image_path}', flush=True)
        sys.exit(1)

    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    fmt_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}
    img_format = fmt_map.get(ext, 'jpeg')

    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    print(f'Image: {image_path} ({len(image_bytes)} bytes, format={img_format})', flush=True)

    graphiti = build_graphiti()
    try:
        # --- Phase 1: Raw index probing (see scores across all indices) ---
        sv = graphiti.s3_vectors
        if sv:
            img_vector = await graphiti.embedder.create_image(image_bytes, img_format)
            print(f'Image vector dim: {len(img_vector)}')

            print_raw_index('edge-fact-embeddings',
                sv.query_edge_vectors(query_vector=img_vector, group_ids=[GROUP_ID], top_k=5, min_score=0.0))
            print_raw_index('edge-source-embeddings',
                sv.query_edge_source_vectors(query_vector=img_vector, group_ids=[GROUP_ID], top_k=5, min_score=0.0))
            print_raw_index('entity-name-embeddings',
                sv.query_entity_vectors(query_vector=img_vector, group_ids=[GROUP_ID], top_k=5, min_score=0.0))
            print_raw_index('describes-fact-embeddings',
                sv.query_describes_fact_vectors(query_vector=img_vector, group_ids=[GROUP_ID], top_k=5, min_score=0.0))
            print_raw_index('describes-excerpt-embeddings',
                sv.query_describes_excerpt_vectors(query_vector=img_vector, group_ids=[GROUP_ID], top_k=5, min_score=0.0))
            print_raw_index('episode-narrative-embeddings',
                sv.query_narrative_vectors(query_vector=img_vector, group_ids=[GROUP_ID], top_k=5, min_score=0.0))

        # --- Phase 2: Standard image search ---
        config = IMAGE_SEARCH_CONFIG.model_copy(update={'limit': 10})
        results = await graphiti.search_(
            query='',
            query_image=image_bytes,
            query_image_format=img_format,
            config=config,
            group_ids=[GROUP_ID],
        )
        print_results('Image search (standard)', results)

        # --- Phase 3: Deep image search ---
        deep_results = await graphiti.search_(
            query='',
            query_image=image_bytes,
            query_image_format=img_format,
            config=config.model_copy(update={'limit': 20}),
            group_ids=[GROUP_ID],
            deep_search=True,
        )
        print_results('Image search (deep)', deep_results)

        # --- Phase 4: Verification ---
        print('\n--- Verification ---', flush=True)
        deep_node_names = [n.name for n in (deep_results.nodes or [])]
        img_edges = [
            e for e in (deep_results.edges or [])
            if '[image:' in (e.source_excerpt or '')
        ]
        checks = [
            ('Cost Explorer in deep nodes', 'Cost Explorer' in deep_node_names),
            ('Image-sourced edges found', len(img_edges) > 0),
        ]
        all_pass = True
        for desc, ok in checks:
            status = '✓' if ok else '✗'
            print(f'  {status} {desc}', flush=True)
            if not ok:
                all_pass = False
        if all_pass:
            print('  All checks passed', flush=True)
    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
