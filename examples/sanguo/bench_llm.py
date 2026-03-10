"""
Benchmark Kimi K2.5 via Bedrock Mantle — measure latency vs input/output size.

Usage:
    uv run python tests/sanguo/bench_llm.py

Tests:
  1. Fixed input, varying requested output length
  2. Varying input size, fixed output
  3. Simulate extract_edges prompt structure with different context sizes
"""

import asyncio
import time
import json
from dotenv import load_dotenv
import os

load_dotenv()

REGION = os.environ.get('AWS_REGION', 'us-east-1')
MODEL = os.environ.get('BEDROCK_MODEL', 'moonshotai.kimi-k2.5')


def get_client():
    from openai import AsyncOpenAI
    from aws_bedrock_token_generator import BedrockTokenGenerator
    import boto3

    session = boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()
    token = BedrockTokenGenerator().get_token(credentials=creds, region=REGION)
    return AsyncOpenAI(
        api_key=token,
        base_url=f'https://bedrock-mantle.{REGION}.api.aws/v1',
    )


EXTRACTED_EDGES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ExtractedEdges",
        "schema": {
            "properties": {
                "edges": {
                    "items": {
                        "properties": {
                            "source_entity_name": {"type": "string"},
                            "target_entity_name": {"type": "string"},
                            "relation_type": {"type": "string"},
                            "fact": {"type": "string"},
                            "source_excerpt": {"default": "", "type": "string"},
                            "valid_at": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                            "invalid_at": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                        },
                        "required": ["source_entity_name", "target_entity_name", "relation_type", "fact"],
                        "type": "object",
                    },
                    "type": "array",
                },
                "narrative_excerpts": {
                    "default": [],
                    "items": {"type": "string"},
                    "type": "array",
                },
            },
            "required": ["edges"],
            "type": "object",
        },
    },
}


async def call_llm(client, system: str, user: str, max_tokens: int = 4096,
                   response_format: dict | None = None) -> dict:
    """Single LLM call, return timing + token info."""
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            response_format=response_format or {'type': 'json_object'},
        )
        latency = time.time() - t0
        content = resp.choices[0].message.content or ''
        return {
            'input_tokens': resp.usage.prompt_tokens if resp.usage else 0,
            'output_tokens': resp.usage.completion_tokens if resp.usage else 0,
            'latency_s': round(latency, 1),
            'output_len': len(content),
            'ok': True,
        }
    except Exception as e:
        return {
            'latency_s': round(time.time() - t0, 1),
            'error': str(e)[:200],
            'ok': False,
        }


# --- Test prompts ---

SANGUO_PARA_0 = """话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。及秦灭之后，楚、汉分争，又并入于汉。汉朝自高祖斩白蛇而起义，一统天下，后来光武中兴，传至献帝，遂分为三国。推其致乱之由，殆始于桓、灵二帝。桓帝禁锢善类，崇信宦官。及桓帝崩，灵帝即位，大将军窦武、太傅陈蕃共相辅佐。时有宦官曹节等弄权，窦武、陈蕃谋诛之，机事不密，反为所害，中涓自此愈横。"""

SANGUO_PARA_1 = """建宁二年四月望日，帝御温德殿。方升座，殿角狂风骤起。只见一条大青蛇，从梁上飞将下来，蟠于椅上。帝惊倒，左右急救入宫，百官俱奔避。须臾，蛇不见了。忽然大雷大雨，加以冰雹，落到半夜方止，坏却房屋无数。建宁四年二月，洛阳地震；又海水泛溢，沿海居民，尽被大浪卷入海中。光和元年，雌鸡化雄。六月朔，黑气十余丈，飞入温德殿中。秋七月，有虹现于玉堂；五原山岸，尽皆崩裂。种种不祥，非止一端。帝下诏问群臣以灾异之由，议郎蔡邕上疏，以为蜺堕鸡化，乃妇寺干政之所致，言颇切直。帝览奏叹息，因黄门传诏。曹节在后窃视，悉宣告左右；遂以他事陷邕于罪，放归田里。后张让、赵忠、封谞、段珪、曹节、侯览、蹇硕、程旷、夏恽、郭胜十人朋比为奸，号为"十常侍"。帝尊信张让，呼为"阿父"。朝政日非，以致天下人心思乱，盗贼蜂起。"""

SANGUO_PARA_2 = """时巨鹿郡有兄弟三人，一名张角，一名张宝，一名张梁。那张角本是个不第秀才，因入山采药，遇一老人，碧眼童颜，手执藜杖，唤角至一洞中，以天书三卷授之，曰："此名《太平要术》，汝得之，当代天宣化，普救世人；若萌异心，必获恶报。"角拜问姓名。老人曰："吾乃南华老仙也。"言讫，化阵清风而去。角得此书，晓夜攻习，能呼风唤雨，号为"太平道人"。中平元年正月内，疫气流行，张角散施符水，为人治病，自称"大贤良师"。角有徒弟五百余人，云游四方，皆能书符念咒。次后徒众日多，角乃立三十六方，大方万余人，小方六七千，各立渠帅，称为将军；讹言"苍天已死，黄天当立"，又云"岁在甲子，天下大吉"。令人各以白土书"甲子"二字于家中大门上。青、幽、徐、冀、荆、扬、兖、豫八州之人，家家侍奉大贤良师。角遣其党马元义，暗赍金帛，结交中涓封谞，以为内应。"""

ENTITIES_SMALL = json.dumps([
    {"name": "汉朝", "type": "Entity"},
    {"name": "桓帝", "type": "Entity"},
    {"name": "灵帝", "type": "Entity"},
    {"name": "窦武", "type": "Entity"},
    {"name": "陈蕃", "type": "Entity"},
], ensure_ascii=False)

ENTITIES_LARGE = json.dumps([
    {"name": "汉朝", "type": "Entity"}, {"name": "桓帝", "type": "Entity"},
    {"name": "灵帝", "type": "Entity"}, {"name": "窦武", "type": "Entity"},
    {"name": "陈蕃", "type": "Entity"}, {"name": "曹节", "type": "Entity"},
    {"name": "张角", "type": "Entity"}, {"name": "张宝", "type": "Entity"},
    {"name": "张梁", "type": "Entity"}, {"name": "南华老仙", "type": "Entity"},
    {"name": "太平道", "type": "Entity"}, {"name": "马元义", "type": "Entity"},
    {"name": "封谞", "type": "Entity"}, {"name": "蔡邕", "type": "Entity"},
    {"name": "张让", "type": "Entity"}, {"name": "赵忠", "type": "Entity"},
    {"name": "段珪", "type": "Entity"}, {"name": "蹇硕", "type": "Entity"},
    {"name": "十常侍", "type": "Entity"}, {"name": "黄巾军", "type": "Entity"},
    {"name": "巨鹿郡", "type": "Entity"}, {"name": "温德殿", "type": "Entity"},
    {"name": "洛阳", "type": "Entity"}, {"name": "高祖", "type": "Entity"},
    {"name": "光武帝", "type": "Entity"}, {"name": "献帝", "type": "Entity"},
], ensure_ascii=False)

SYSTEM_PROMPT = "You are an expert fact extractor. Extract relationships as JSON."

def build_extract_edges_prompt(previous: list[str], current: str, entities: str) -> str:
    prev_json = json.dumps(previous, ensure_ascii=False)
    return f"""<PREVIOUS_MESSAGES>
{prev_json}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{current}
</CURRENT_MESSAGE>

<ENTITIES>
{entities}
</ENTITIES>

Extract all factual relationships between the ENTITIES based on the CURRENT MESSAGE.
Return JSON with "edges" array, each edge has: source_entity_name, target_entity_name, relation_type, fact, source_excerpt.
Also return "narrative_excerpts" array for text not covered by any edge."""


async def main():
    client = get_client()

    print("=" * 70)
    print("Benchmark: Kimi K2.5 via Bedrock Mantle")
    print(f"Model: {MODEL}, Region: {REGION}")
    print("=" * 70)

    # --- Test 1: Simple echo with varying output ---
    print("\n--- Test 1: Fixed input, varying requested output ---")
    for n_items in [5, 20, 50]:
        prompt = f"Generate a JSON array of exactly {n_items} fictional Chinese historical characters. Each item: {{\"name\": \"...\", \"title\": \"...\", \"era\": \"...\"}}. Return JSON only."
        r = await call_llm(client, "You are a helpful assistant. Return JSON only.", prompt)
        status = f"in={r.get('input_tokens',0)} out={r.get('output_tokens',0)}" if r['ok'] else r.get('error','')[:80]
        print(f"  items={n_items:3d}  latency={r['latency_s']:6.1f}s  {status}")

    # --- Test 2: Simulate extract_edges with increasing context ---
    print("\n--- Test 2: extract_edges prompt, increasing PREVIOUS_MESSAGES ---")
    test_cases = [
        ("0 prev", [], SANGUO_PARA_0, ENTITIES_SMALL),
        ("1 prev", [SANGUO_PARA_0], SANGUO_PARA_1, ENTITIES_SMALL),
        ("2 prev", [SANGUO_PARA_0, SANGUO_PARA_1], SANGUO_PARA_2, ENTITIES_LARGE),
    ]
    for label, prev, current, entities in test_cases:
        prompt = build_extract_edges_prompt(prev, current, entities)
        r = await call_llm(client, SYSTEM_PROMPT, prompt)
        status = f"in={r.get('input_tokens',0)} out={r.get('output_tokens',0)}" if r['ok'] else r.get('error','')[:80]
        print(f"  {label:8s}  latency={r['latency_s']:6.1f}s  {status}")

    # --- Test 3: Same prompt repeated 3 times to check variance ---
    print("\n--- Test 3: Same prompt x3 (variance check) ---")
    prompt = build_extract_edges_prompt(
        [SANGUO_PARA_0, SANGUO_PARA_1], SANGUO_PARA_2, ENTITIES_LARGE
    )
    for i in range(3):
        r = await call_llm(client, SYSTEM_PROMPT, prompt)
        status = f"in={r.get('input_tokens',0)} out={r.get('output_tokens',0)}" if r['ok'] else r.get('error','')[:80]
        print(f"  run {i+1}:  latency={r['latency_s']:6.1f}s  {status}")

    # --- Test 4: Concurrent requests (simulate resolve_extracted_edges burst) ---
    print("\n--- Test 4: Concurrent requests (simulating SEMAPHORE_LIMIT burst) ---")
    simple_prompt = (
        "Given the fact: '桓帝禁锢善类，崇信宦官', and the entities 桓帝 and 宦官, "
        "determine if this is a duplicate of: '桓帝信任宦官'. "
        "Return JSON: {\"is_duplicate\": true/false, \"reason\": \"...\"}"
    )
    for concurrency in [1, 5, 10, 20]:
        tasks = []
        t0 = time.time()
        for _ in range(concurrency):
            tasks.append(call_llm(client, "You are a helpful assistant. Return JSON only.", simple_prompt, max_tokens=256))
        results = await asyncio.gather(*tasks)
        wall_time = time.time() - t0
        ok_count = sum(1 for r in results if r['ok'])
        latencies = [r['latency_s'] for r in results if r['ok']]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        max_lat = max(latencies) if latencies else 0
        errors = [r.get('error', '')[:60] for r in results if not r['ok']]
        err_str = f"  errors: {errors}" if errors else ""
        print(f"  concurrency={concurrency:2d}  wall={wall_time:.1f}s  ok={ok_count}/{concurrency}  avg_lat={avg_lat:.1f}s  max_lat={max_lat:.1f}s{err_str}")

    # --- Test 5: json_schema vs json_object format comparison ---
    print("\n--- Test 5: json_schema (structured output) vs json_object ---")
    prompt = build_extract_edges_prompt(
        [SANGUO_PARA_0, SANGUO_PARA_1], SANGUO_PARA_2, ENTITIES_LARGE
    )
    for label, fmt in [("json_object", None), ("json_schema", EXTRACTED_EDGES_SCHEMA)]:
        r = await call_llm(client, SYSTEM_PROMPT, prompt, max_tokens=16384, response_format=fmt)
        status = f"in={r.get('input_tokens',0)} out={r.get('output_tokens',0)}" if r['ok'] else r.get('error','')[:120]
        print(f"  {label:12s}  latency={r['latency_s']:6.1f}s  {status}")

    print("\nDone.")


if __name__ == '__main__':
    asyncio.run(main())
