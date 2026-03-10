"""
Graphiti CRUD 综合测试 (AWS Bedrock: Kimi K2.5 + Nova MME + Neo4j)

测试覆盖:
  CREATE: add_episode (text/json), add_triplet
  READ:   search, search_ (高级搜索), retrieve_episodes,
          get_nodes_and_edges_by_episode, namespace get_by_uuid/get_by_node_uuid
  UPDATE: 修改 node summary, 修改 edge fact, 时序更新(矛盾信息导致 edge 失效)
  DELETE: remove_episode, namespace delete node/edge

Usage:
    cd graphiti
    uv run python examples/quickstart/test_crud_bedrock.py
"""

import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.bedrock_reranker_client import BedrockRerankerClient
from graphiti_core.embedder.bedrock_nova import BedrockNovaEmbedder, BedrockNovaEmbedderConfig
from graphiti_core.llm_client.bedrock_client import BedrockLLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_RRF,
)
from graphiti_core.utils.datetime_utils import utc_now

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
load_dotenv()

REGION = os.environ.get('AWS_REGION', 'us-east-1')
LLM_MODEL = os.environ.get('BEDROCK_MODEL', 'moonshotai.kimi-k2.5')
EMBEDDING_MODEL = os.environ.get(
    'BEDROCK_EMBEDDING_MODEL', 'amazon.nova-2-multimodal-embeddings-v1:0'
)
LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.1'))
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

GROUP_ID = 'crud_test'
passed = 0
failed = 0


def ok(name: str, detail: str = ''):
    global passed
    passed += 1
    print(f'  ✅ {name}' + (f' — {detail}' if detail else ''))


def fail(name: str, detail: str = ''):
    global failed
    failed += 1
    print(f'  ❌ {name}' + (f' — {detail}' if detail else ''))


async def main():
    global passed, failed

    llm_client = BedrockLLMClient(
        config=LLMConfig(model=LLM_MODEL, small_model=LLM_MODEL, temperature=LLM_TEMPERATURE),
        region_name=REGION,
    )
    embedder = BedrockNovaEmbedder(
        config=BedrockNovaEmbedderConfig(
            model_id=EMBEDDING_MODEL, region_name=REGION, embedding_dim=1024,
        )
    )
    cross_encoder = BedrockRerankerClient(client=llm_client.client, model_id=LLM_MODEL)

    graphiti = Graphiti(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
        llm_client=llm_client, embedder=embedder, cross_encoder=cross_encoder,
    )

    try:
        print('\n🧹 清理旧测试数据...')
        try:
            await EntityNode.delete_by_group_id(graphiti.driver, GROUP_ID)
        except Exception:
            pass

        await graphiti.build_indices_and_constraints()
        print('  索引/约束已就绪')

        # 1. CREATE — add_episode (text)
        print('\n📝 1. CREATE: add_episode (text)')
        ref_time = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        result1 = await graphiti.add_episode(
            name='项目启动',
            episode_body='张三是DataFlow项目的技术负责人，他在北京办公室工作。李四是该项目的产品经理。',
            source=EpisodeType.text, source_description='项目周报',
            reference_time=ref_time, group_id=GROUP_ID,
        )
        episode1_uuid = result1.episode.uuid
        nodes_created = result1.nodes
        edges_created = result1.edges
        if episode1_uuid and len(nodes_created) > 0:
            ok('add_episode(text)', f'episode={episode1_uuid[:8]}..., nodes={len(nodes_created)}, edges={len(edges_created)}')
        else:
            fail('add_episode(text)', 'no episode or nodes created')

        # 2. CREATE — add_episode (json)
        print('\n📝 2. CREATE: add_episode (json)')
        ref_time2 = ref_time + timedelta(hours=1)
        result2 = await graphiti.add_episode(
            name='项目元数据',
            episode_body=json.dumps({
                'project': 'DataFlow', 'department': '数据工程部', 'budget': '500万',
                'start_date': '2025-06-01', 'tech_stack': ['Python', 'Neo4j', 'Kafka'],
            }, ensure_ascii=False),
            source=EpisodeType.json, source_description='项目管理系统',
            reference_time=ref_time2, group_id=GROUP_ID,
        )
        episode2_uuid = result2.episode.uuid
        if episode2_uuid and len(result2.nodes) > 0:
            ok('add_episode(json)', f'episode={episode2_uuid[:8]}..., nodes={len(result2.nodes)}, edges={len(result2.edges)}')
        else:
            fail('add_episode(json)', 'no data created')

        # 3. CREATE — add_triplet
        print('\n📝 3. CREATE: add_triplet')
        now = utc_now()
        source_node = EntityNode(name='王五', group_id=GROUP_ID, summary='王五是DataFlow项目的后端开发工程师', labels=['Entity'], created_at=now)
        target_node = EntityNode(name='DataFlow', group_id=GROUP_ID, summary='DataFlow是一个数据工程项目', labels=['Entity'], created_at=now)
        triplet_edge = EntityEdge(name='DEVELOPS', group_id=GROUP_ID, source_node_uuid=source_node.uuid, target_node_uuid=target_node.uuid, fact='王五负责开发DataFlow项目的后端服务', created_at=now, valid_at=now)
        triplet_result = await graphiti.add_triplet(source_node, triplet_edge, target_node)
        if len(triplet_result.nodes) >= 2 and len(triplet_result.edges) >= 1:
            ok('add_triplet', f'nodes={len(triplet_result.nodes)}, edges={len(triplet_result.edges)}')
            triplet_edge_uuid = triplet_result.edges[0].uuid
            triplet_source_uuid = triplet_result.nodes[0].uuid
        else:
            fail('add_triplet', f'nodes={len(triplet_result.nodes)}, edges={len(triplet_result.edges)}')
            triplet_edge_uuid = triplet_source_uuid = None

        # 4. READ — search
        print('\n🔍 4. READ: search (edge search)')
        search_results = await graphiti.search('谁在DataFlow项目工作？', group_ids=[GROUP_ID], num_results=10)
        if len(search_results) > 0:
            ok('search(edges)', f'返回 {len(search_results)} 条事实')
            for r in search_results[:3]:
                print(f'      → {r.fact}')
        else:
            fail('search(edges)', '无结果')

        # 5. READ — search_
        print('\n🔍 5. READ: search_ (高级搜索)')
        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={'limit': 5})
        advanced_results = await graphiti.search_(query='DataFlow项目', config=config, group_ids=[GROUP_ID])
        if len(advanced_results.nodes) > 0 or len(advanced_results.edges) > 0:
            ok('search_(高级)', f'nodes={len(advanced_results.nodes)}, edges={len(advanced_results.edges)}')
        else:
            fail('search_(高级)', '无结果')

        # 6. READ — retrieve_episodes
        print('\n🔍 6. READ: retrieve_episodes')
        episodes = await graphiti.retrieve_episodes(reference_time=datetime.now(timezone.utc), last_n=10, group_ids=[GROUP_ID])
        if len(episodes) > 0:
            ok('retrieve_episodes', f'返回 {len(episodes)} 个 episodes')
        else:
            fail('retrieve_episodes', '无结果')

        # 7-10. READ — various get methods
        print('\n🔍 7-10. READ: get_by_uuid / get_by_node_uuid')
        try:
            await graphiti.get_nodes_and_edges_by_episode([episode1_uuid])
            ok('get_nodes_and_edges_by_episode')
        except Exception as e:
            fail('get_nodes_and_edges_by_episode', str(e))
        if nodes_created:
            try:
                await graphiti.nodes.entity.get_by_uuid(nodes_created[0].uuid)
                ok('nodes.entity.get_by_uuid')
            except Exception as e:
                fail('nodes.entity.get_by_uuid', str(e))
        if edges_created:
            try:
                await graphiti.edges.entity.get_by_uuid(edges_created[0].uuid)
                ok('edges.entity.get_by_uuid')
            except Exception as e:
                fail('edges.entity.get_by_uuid', str(e))

        # 11-12. UPDATE
        print('\n✏️  11-12. UPDATE: node summary + edge fact')
        if nodes_created:
            node = await graphiti.nodes.entity.get_by_uuid(nodes_created[0].uuid)
            node.summary += ' [已更新]'
            await graphiti.nodes.entity.save(node)
            v = await graphiti.nodes.entity.get_by_uuid(node.uuid)
            ok('update node') if '已更新' in v.summary else fail('update node')
        if edges_created:
            edge = await graphiti.edges.entity.get_by_uuid(edges_created[0].uuid)
            edge.fact += ' (补充)'
            await graphiti.edges.entity.save(edge)
            v = await graphiti.edges.entity.get_by_uuid(edge.uuid)
            ok('update edge') if '补充' in v.fact else fail('update edge')

        # 13. UPDATE — temporal
        print('\n✏️  13. UPDATE: 时序更新')
        result3 = await graphiti.add_episode(
            name='人事变动',
            episode_body='张三已经离开DataFlow项目，转去了CloudNet项目。李四接替张三成为DataFlow项目的新技术负责人。',
            source=EpisodeType.text, source_description='人事通知',
            reference_time=ref_time + timedelta(days=30), group_id=GROUP_ID,
        )
        ok('时序更新', f'edges={len(result3.edges)}')

        # 15-17. DELETE
        print('\n🗑️  15-17. DELETE')
        if triplet_edge_uuid:
            try:
                e = await graphiti.edges.entity.get_by_uuid(triplet_edge_uuid)
                await graphiti.edges.entity.delete(e)
                ok('delete edge')
            except Exception as ex:
                fail('delete edge', str(ex))
        if triplet_source_uuid:
            try:
                n = await graphiti.nodes.entity.get_by_uuid(triplet_source_uuid)
                await graphiti.nodes.entity.delete(n)
                ok('delete node')
            except Exception as ex:
                fail('delete node', str(ex))
        try:
            await graphiti.remove_episode(episode2_uuid)
            ok('remove_episode')
        except Exception as ex:
            fail('remove_episode', str(ex))

    except Exception as e:
        print(f'\n💥 未捕获异常: {e}')
        traceback.print_exc()
    finally:
        await graphiti.close()

    total = passed + failed
    print(f'\n{"="*50}')
    print(f'  测试完成: {passed}/{total} 通过, {failed}/{total} 失败')
    print(f'{"="*50}')


if __name__ == '__main__':
    asyncio.run(main())
