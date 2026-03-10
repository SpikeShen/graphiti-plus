"""
8. Test entity_types with Person model — verify LLM attribute extraction.

Ingests a short text episode about people, using entity_types={'Person': Person}
to guide the LLM to extract structured attributes (name, role, etc.).

Then demonstrates programmatic update of pipeline-fillable fields (photo_s3_uri).

Usage:
    cd graphiti
    examples/docx-manual/run.sh 8
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from common import GROUP_ID, build_graphiti

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Entity type definitions
# ---------------------------------------------------------------------------

class Person(BaseModel):
    """A human person mentioned in the text."""

    role: str | None = Field(None, description='职务或角色，如项目经理、架构师')
    organization: str | None = Field(None, description='所属组织或公司')
    expertise: list[str] = Field(default_factory=list, description='专业领域，如云计算、数据库')


class AWSService(BaseModel):
    """An AWS cloud service or product."""

    service_category: str | None = Field(None, description='服务类别，如计算、存储、数据库')
    is_managed: bool | None = Field(None, description='是否为托管服务')


ENTITY_TYPES = {
    'Person': Person,
    'AWSService': AWSService,
}

# ---------------------------------------------------------------------------
# Test episode content
# ---------------------------------------------------------------------------

EPISODE_TEXT = """
项目周会纪要（2026-03-01）

参会人员：张伟（项目经理，来自云智科技）、李明（架构师，来自云智科技）、王芳（DBA，来自数据无限公司）

讨论要点：
1. 张伟汇报了MAP 2.0迁移项目的整体进度，目前已完成60%的资源标签打标工作。
2. 李明提出将核心数据库从自建MySQL迁移到Amazon Aurora，预计可以降低30%的运维成本。
   他同时建议使用Amazon S3存储历史归档数据，配合S3 Intelligent-Tiering自动优化存储成本。
3. 王芳负责数据库迁移的具体执行，她提到需要使用AWS DMS（Database Migration Service）
   来实现在线迁移，减少停机时间。
4. 团队决定下周进行Aurora的POC测试，由李明和王芳共同负责。

行动项：
- 张伟：更新项目甘特图，同步给客户
- 李明：准备Aurora架构方案文档
- 王芳：搭建DMS测试环境
"""


async def main():
    print('=== Entity Types Test ===\n', flush=True)

    graphiti = build_graphiti()
    try:
        # Ingest with entity_types
        print('Ingesting episode with entity_types...', flush=True)
        result = await graphiti.add_episode(
            name='project-meeting-20260301',
            episode_body=EPISODE_TEXT,
            source_description='项目周会纪要',
            reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
            group_id=GROUP_ID,
            entity_types=ENTITY_TYPES,
        )

        print(f'\nExtracted {len(result.nodes)} nodes, {len(result.edges)} edges\n', flush=True)

        # Show extracted entities with their types and attributes
        print('--- Extracted Entities ---', flush=True)
        for node in sorted(result.nodes, key=lambda n: n.name):
            labels = [l for l in node.labels if l != 'Entity']
            label_str = ', '.join(labels) if labels else 'Entity'
            print(f'\n  [{label_str}] {node.name}', flush=True)
            print(f'    summary: {node.summary[:80]}...' if len(node.summary) > 80
                  else f'    summary: {node.summary}', flush=True)
            if node.attributes:
                print(f'    attributes: {json.dumps(node.attributes, ensure_ascii=False, indent=6)}',
                      flush=True)
            else:
                print('    attributes: (empty)', flush=True)

        # Show edges
        print('\n--- Extracted Edges ---', flush=True)
        for edge in result.edges:
            print(f'  {edge.source_node_uuid[:8]}→{edge.target_node_uuid[:8]}: {edge.fact}',
              flush=True)

        # Demonstrate programmatic attribute update
        print('\n--- Programmatic Attribute Update ---', flush=True)
        person_nodes = [n for n in result.nodes if 'Person' in n.labels]
        if person_nodes:
            node = person_nodes[0]
            print(f'  Updating {node.name} with pipeline-generated fields...', flush=True)

            # Simulate adding photo and face embedding (pipeline-fillable, not LLM-fillable)
            node.attributes['photo_s3_uri'] = f's3://graphiti-multimodal-assets-poc/faces/{node.uuid}/profile.jpg'
            node.attributes['face_embedding_dim'] = 512
            node.attributes['face_confidence'] = 0.95

            await node.save(graphiti.driver)
            print(f'  Saved. attributes now: {json.dumps(node.attributes, ensure_ascii=False)}',
                  flush=True)

            # Verify by re-reading from DB
            from graphiti_core.nodes import EntityNode
            reloaded = await EntityNode.get_by_uuid(graphiti.driver, node.uuid)
            print(f'  Reloaded from DB: {json.dumps(reloaded.attributes, ensure_ascii=False)}',
                  flush=True)

            # Check round-trip
            assert reloaded.attributes.get('photo_s3_uri') == node.attributes['photo_s3_uri'], \
                'photo_s3_uri round-trip failed!'
            assert reloaded.attributes.get('face_confidence') == 0.95, \
                'face_confidence round-trip failed!'
            print('  ✓ Round-trip verification passed', flush=True)

    finally:
        await graphiti.close()
        print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
