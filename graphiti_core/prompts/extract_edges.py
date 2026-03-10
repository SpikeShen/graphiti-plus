"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import base64
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field, model_validator

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json


class Edge(BaseModel):
    source_entity_name: str = Field(
        ..., description='The name of the source entity from the ENTITIES list'
    )
    target_entity_name: str = Field(
        ..., description='The name of the target entity from the ENTITIES list'
    )
    relation_type: str = Field(
        ...,
        description='The type of relationship between the entities, in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)',
    )
    fact: str = Field(
        ...,
        description='A natural language description of the relationship between the entities, paraphrased from the source text',
    )
    source_excerpt: str = Field(
        default='',
        description='The exact sentence(s) from the CURRENT MESSAGE that support this fact. Copy verbatim from the source text, do not paraphrase.',
    )
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )


class NarrativeExcerpt(BaseModel):
    excerpt: str = Field(
        ...,
        description='The exact sentence(s) copied verbatim from the CURRENT MESSAGE.',
    )
    related_entity: str | None = Field(
        default=None,
        description='If this excerpt primarily describes a specific entity from the ENTITIES list '
        '(e.g., abilities, characteristics, background), put the entity name here. '
        'None if the excerpt is pure narrative/setting.',
    )
    fact: str | None = Field(
        default=None,
        description='If related_entity is set, a concise summary of what this excerpt '
        'describes about the entity. None if the excerpt is pure narrative/setting.',
    )


# Backward compatibility alias
UncoveredExcerpt = NarrativeExcerpt


class ExtractedEdges(BaseModel):
    edges: list[Edge]
    narrative_excerpts: list[NarrativeExcerpt] = Field(
        default_factory=list,
        description='Narrative sentences or clauses from the CURRENT MESSAGE that cannot be represented '
        'as a relationship between two named entities. These include descriptive passages about abilities, '
        'characteristics, anonymous groups, settings, or background information. '
        'For each excerpt, determine if it primarily describes a specific entity from the ENTITIES list.',
    )

    @model_validator(mode='before')
    @classmethod
    def _coerce_narrative_excerpts(cls, data: dict) -> dict:
        """Accept plain strings and legacy 'uncovered_excerpts' key for backward compatibility."""
        if isinstance(data, dict):
            # Accept legacy key
            if 'uncovered_excerpts' in data and 'narrative_excerpts' not in data:
                data['narrative_excerpts'] = data.pop('uncovered_excerpts')
            raw = data.get('narrative_excerpts', [])
            if isinstance(raw, list):
                coerced = []
                for item in raw:
                    if isinstance(item, str):
                        coerced.append({'excerpt': item})
                    else:
                        coerced.append(item)
                data['narrative_excerpts'] = coerced
        return data


class Prompt(Protocol):
    edge: PromptVersion
    edge_document: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    edge_document: PromptFunction
    extract_attributes: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    edge_types_section = ''
    if context.get('edge_types'):
        edge_types_section = f"""
<FACT_TYPES>
{to_prompt_json(context['edge_types'])}
</FACT_TYPES>
"""

    return [
        Message(
            role='system',
            content='You are an expert fact extractor that extracts fact triples from text. '
            '1. Extracted fact triples should also be extracted with relevant date information.'
            '2. Treat the CURRENT TIME as the time the CURRENT MESSAGE was sent. All temporal information should be extracted relative to this time.',
        ),
        Message(
            role='user',
            content=f"""
<PREVIOUS_MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>

<ENTITIES>
{to_prompt_json(context['nodes'])}
</ENTITIES>

<REFERENCE_TIME>
{context['reference_time']}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>
{edge_types_section}
# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


{context['custom_extraction_instructions']}

# EXTRACTION RULES

1. **Entity Name Validation**: `source_entity_name` and `target_entity_name` must use only the `name` values from the ENTITIES list provided above.
   - **CRITICAL**: Using names not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities.
3. Do not emit duplicate or semantically redundant facts.
4. The `fact` should closely paraphrase the original source sentence(s). Do not verbatim quote the original text.
5. The `source_excerpt` must be the exact sentence(s) copied verbatim from the CURRENT MESSAGE that support the extracted fact. Do not paraphrase or modify.
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week").
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# RELATION TYPE RULES

- If FACT_TYPES are provided and the relationship matches one of the types (considering the entity type signature), use that fact_type_name as the `relation_type`.
- Otherwise, derive a `relation_type` from the relationship predicate in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.

# NARRATIVE EXCERPTS RULES

After extracting all edges, review the CURRENT MESSAGE and identify any sentences or clauses that are NOT covered by any extracted edge's `source_excerpt`.
These are narrative passages that cannot be represented as a named-entity-to-named-entity relationship.
- Include descriptive text about abilities, characteristics, anonymous groups, settings, or background information.
- Copy each narrative sentence/clause verbatim from the CURRENT MESSAGE.
- Do NOT include text that is already captured in an edge's `source_excerpt`.
- If all text is covered by edges, return an empty list.
- For each narrative excerpt, determine if it primarily describes a specific entity from the ENTITIES list (e.g., abilities, characteristics, background of that entity). If so, set `related_entity` to that entity's name and provide a concise `fact` summarizing what the excerpt describes about the entity. If the excerpt is pure narrative, setting description, or cannot be attributed to a single entity, set both `related_entity` and `fact` to null.
        """,
        ),
    ]


def edge_document(context: dict[str, Any]) -> list[Message]:
    """Build a multimodal edge extraction prompt for document-type episodes.

    When ``content_blocks`` contains image blocks with ``_raw_bytes``, the user
    message is a content-array so a vision LLM can directly understand images.
    For images already uploaded (no ``_raw_bytes``), the ``s3_uri`` reference
    format ``[image:s3://...]`` is used so the LLM can reference it in
    ``source_excerpt``.
    """
    from graphiti_core.nodes import ContentBlockType

    content_blocks = context.get('content_blocks', [])

    edge_types_section = ''
    if context.get('edge_types'):
        edge_types_section = f"""
<FACT_TYPES>
{to_prompt_json(context['edge_types'])}
</FACT_TYPES>
"""

    sys_content = (
        'You are an expert fact extractor that extracts fact triples from documents. '
        'The document may contain text and images. '
        '1. Extracted fact triples should also be extracted with relevant date information. '
        '2. Treat the CURRENT TIME as the time the document was ingested. '
        'All temporal information should be extracted relative to this time.'
    )

    # --- Build multimodal user content parts ---
    user_parts: list[dict[str, Any]] = []

    user_parts.append({
        'type': 'text',
        'text': f"""
<PREVIOUS_MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
""",
    })

    # Insert content blocks in order
    for block in sorted(content_blocks, key=lambda b: b.index):
        if block.parent_index is not None:
            continue
        if block.block_type == ContentBlockType.text:
            if block.text:
                user_parts.append({'type': 'text', 'text': block.text})
        elif block.block_type == ContentBlockType.image:
            if block._raw_bytes:
                b64 = base64.b64encode(block._raw_bytes).decode('utf-8')
                mime = block.mime_type or 'image/jpeg'
                user_parts.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:{mime};base64,{b64}'},
                })
                # Add a text hint so LLM knows how to reference this image in source_excerpt
                if block.s3_uri:
                    user_parts.append({
                        'type': 'text',
                        'text': f'(The above image can be referenced as: [image:{block.s3_uri}])',
                    })
            elif block.s3_uri:
                # Already uploaded, use description + reference marker
                desc = block.description or ''
                user_parts.append({
                    'type': 'text',
                    'text': f'[image:{block.s3_uri}] {desc}',
                })
        else:
            user_parts.append({'type': 'text', 'text': block.text_representation})

    user_parts.append({
        'type': 'text',
        'text': f"""
</CURRENT_MESSAGE>

<ENTITIES>
{to_prompt_json(context['nodes'])}
</ENTITIES>

<REFERENCE_TIME>
{context['reference_time']}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>
{edge_types_section}
# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
The document may contain images — extract facts from both text and images.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE (text or images),
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


{context['custom_extraction_instructions']}

# EXTRACTION RULES

1. **Entity Name Validation**: `source_entity_name` and `target_entity_name` must use only the `name` values from the ENTITIES list provided above.
   - **CRITICAL**: Using names not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities.
3. Do not emit duplicate or semantically redundant facts.
4. The `fact` should closely paraphrase the original source sentence(s) or describe what is depicted in the image. Do not verbatim quote the original text.
5. **source_excerpt rules**:
   - For facts from text: copy the exact sentence(s) verbatim from the CURRENT MESSAGE.
   - For facts from images: use the format `[image:s3://...] description` where the s3 URI is the image reference shown in the document, followed by a brief description of what the image depicts that supports the fact.
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week").
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# RELATION TYPE RULES

- If FACT_TYPES are provided and the relationship matches one of the types (considering the entity type signature), use that fact_type_name as the `relation_type`.
- Otherwise, derive a `relation_type` from the relationship predicate in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.

# NARRATIVE EXCERPTS RULES

After extracting all edges, review the CURRENT MESSAGE and identify any sentences or clauses that are NOT covered by any extracted edge's `source_excerpt`.
These are narrative passages that cannot be represented as a named-entity-to-named-entity relationship.
- Include descriptive text about abilities, characteristics, anonymous groups, settings, or background information.
- For text: copy each narrative sentence/clause verbatim from the CURRENT MESSAGE.
- For images: use the format `[image:s3://...] description` as the excerpt.
- Do NOT include text that is already captured in an edge's `source_excerpt`.
- If all content is covered by edges, return an empty list.
- For each narrative excerpt, determine if it primarily describes a specific entity from the ENTITIES list. If so, set `related_entity` to that entity's name and provide a concise `fact` summarizing what the excerpt describes about the entity. If the excerpt is pure narrative, setting description, or cannot be attributed to a single entity, set both `related_entity` and `fact` to null.
""",
    })

    return [
        Message(role='system', content=sys_content),
        Message(role='user', content=user_parts),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts fact properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following FACT, its REFERENCE TIME, and any EXISTING ATTRIBUTES, extract or update
        attributes based on the information explicitly stated in the fact. Use the provided attribute
        descriptions to understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate attribute values if they cannot be found explicitly in the fact.
        2. Only use information stated in the FACT to set attribute values.
        3. Use REFERENCE TIME to resolve any relative temporal expressions in the fact.
        4. Preserve existing attribute values unless the fact explicitly provides new information.

        <FACT>
        {context['fact']}
        </FACT>

        <REFERENCE TIME>
        {context['reference_time']}
        </REFERENCE TIME>

        <EXISTING ATTRIBUTES>
        {to_prompt_json(context['existing_attributes'])}
        </EXISTING ATTRIBUTES>
        """,
        ),
    ]


versions: Versions = {
    'edge': edge,
    'edge_document': edge_document,
    'extract_attributes': extract_attributes,
}
