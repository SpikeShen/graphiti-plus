"""
Bedrock LLM Client for Graphiti.

Uses OpenAI SDK with Bedrock Mantle endpoint for models like Kimi K2.5.
Authenticates via short-term token generated from IAM credentials
using aws-bedrock-token-generator.

Requirements:
    pip install aws-bedrock-token-generator openai
"""

import asyncio
import json
import logging
import os
import typing
from time import time
from typing import Any, ClassVar

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'moonshotai.kimi-k2.5'

# Timeout configuration by prompt type (in seconds).
# LLM timeout configuration (seconds).
# Configurable via environment variables — code defaults are fallback values.
#
#   LLM_TIMEOUT_DEFAULT          → all other prompts               (default: 15)
#   LLM_TIMEOUT_NODE_EXTRACT_TXT → text node extraction            (default: 15)
#   LLM_TIMEOUT_NODE_EXTRACT_MM  → multimodal node extraction      (default: 30)
#   LLM_TIMEOUT_EDGE_EXTRACT_TXT → text edge extraction            (default: 60)
#   LLM_TIMEOUT_EDGE_EXTRACT_MM  → multimodal edge extraction      (default: 90)
#
# Design rationale:
#   - Text node extraction: short input, short output → 15s (same as default for now)
#   - Multimodal node extraction: large input (base64) but short output → 30s
#   - Text edge extraction: moderate input, large output (many edges) → 60s
#   - Multimodal edge extraction: large input + large output → 90s
#   - Overly long timeouts make ingest time unpredictable; prefer fast-fail + retry.

def _get_timeout_float(env_name: str, default: float) -> float:
    val = os.environ.get(env_name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning('Invalid value for %s=%r, using default %.1f', env_name, val, default)
        return default


def _load_prompt_timeouts() -> dict[str, float]:
    """Build PROMPT_TIMEOUTS dict from env vars with sensible defaults."""
    t_default = _get_timeout_float('LLM_TIMEOUT_DEFAULT', 15.0)
    t_node_txt = _get_timeout_float('LLM_TIMEOUT_NODE_EXTRACT_TXT', 15.0)
    t_node_mm = _get_timeout_float('LLM_TIMEOUT_NODE_EXTRACT_MM', 30.0)
    t_edge_txt = _get_timeout_float('LLM_TIMEOUT_EDGE_EXTRACT_TXT', 60.0)
    t_edge_mm = _get_timeout_float('LLM_TIMEOUT_EDGE_EXTRACT_MM', 90.0)
    return {
        'default': t_default,
        # Node extraction
        'extract_nodes.extract_text': t_node_txt,
        'extract_nodes.extract_message': t_node_txt,
        'extract_nodes.extract_json': t_node_txt,
        'extract_nodes.extract_document': t_node_mm,
        # Edge extraction
        'extract_edges.edge': t_edge_txt,
        'extract_edges.edge_document': t_edge_mm,
    }


PROMPT_TIMEOUTS = _load_prompt_timeouts()

# Enable LLM trace logging via environment variable (checked at runtime)
def _is_llm_trace_enabled() -> bool:
    return os.environ.get('GRAPHITI_LLM_TRACE', 'false').lower() in ('true', '1', 'yes')


def _generate_bedrock_token(region_name: str = 'us-east-1') -> str:
    """Generate a short-term Bedrock API token from IAM credentials."""
    try:
        from aws_bedrock_token_generator import BedrockTokenGenerator
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials().get_frozen_credentials()
        generator = BedrockTokenGenerator()
        token = generator.get_token(credentials=credentials, region=region_name)
        return token
    except ImportError:
        raise ImportError(
            'aws-bedrock-token-generator is required for BedrockLLMClient. '
            'Install it with: pip install aws-bedrock-token-generator'
        ) from None


class BedrockLLMClient(LLMClient):
    """
    LLM client using OpenAI SDK with Bedrock Mantle (OpenAI-compatible) endpoint.

    Supports Mantle-powered models like Kimi K2.5, DeepSeek V3.2, etc.
    Uses IAM credentials (boto3 default chain) to generate short-term tokens.
    """

    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        region_name: str = 'us-east-1',
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if config is None:
            config = LLMConfig()
        if config.model is None:
            config.model = DEFAULT_MODEL

        super().__init__(config, cache)
        self.max_tokens = max_tokens
        self.region_name = region_name

        # Generate token from IAM credentials
        token = _generate_bedrock_token(region_name)
        base_url = f'https://bedrock-mantle.{region_name}.api.aws/v1'

        self.client = AsyncOpenAI(api_key=token, base_url=base_url)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        prompt_name: str | None = None,
        attempt: int = 0,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if isinstance(m.content, list):
                # Multimodal content array (OpenAI vision format)
                if m.role == 'user':
                    openai_messages.append({'role': 'user', 'content': m.content})
                elif m.role == 'system':
                    # System messages don't support content arrays in most APIs;
                    # extract text parts only
                    text_parts = ' '.join(
                        p['text'] for p in m.content
                        if isinstance(p, dict) and p.get('type') == 'text'
                    )
                    openai_messages.append({'role': 'system', 'content': text_parts})
            else:
                if m.role == 'user':
                    openai_messages.append({'role': 'user', 'content': m.content})
                elif m.role == 'system':
                    openai_messages.append({'role': 'system', 'content': m.content})

        model = self.model or DEFAULT_MODEL
        if model_size == ModelSize.small and self.small_model:
            model = self.small_model

        fmt_type = 'json_object'
        
        # Determine timeout based on prompt name
        timeout_seconds = PROMPT_TIMEOUTS.get(prompt_name or '', PROMPT_TIMEOUTS['default'])
        
        try:
            response_format: dict[str, Any] = {'type': 'json_object'}
            if response_model is not None:
                schema_name = getattr(response_model, '__name__', 'structured_response')
                json_schema = response_model.model_json_schema()
                response_format = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': schema_name,
                        'schema': json_schema,
                    },
                }
                fmt_type = f'json_schema({schema_name})'

            _t0 = time()
            n_msgs = len(openai_messages)
            total_chars = sum(
                len(m.get('content', '')) if isinstance(m.get('content'), str)
                else sum(len(p.get('text', '')) for p in m.get('content', []) if isinstance(p, dict) and p.get('type') == 'text')
                for m in openai_messages
            )
            if _is_llm_trace_enabled():
                logger.info(
                    '[LLM_TRACE] >>> SEND  prompt=%s  attempt=%d  model=%s  fmt=%s  '
                    'msgs=%d  chars=%d  max_tokens=%d  timeout=%.1fs',
                    prompt_name, attempt, model, fmt_type, n_msgs, total_chars, 
                    min(max_tokens, self.max_tokens), timeout_seconds,
                )

            # Wrap the API call with asyncio.timeout
            try:
                async with asyncio.timeout(timeout_seconds):
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=openai_messages,
                        temperature=self.temperature,
                        max_tokens=min(max_tokens, self.max_tokens),
                        response_format=response_format,  # type: ignore[arg-type]
                    )
            except asyncio.TimeoutError:
                _elapsed = time() - _t0
                if _is_llm_trace_enabled():
                    logger.warning(
                        '[LLM_TRACE] !!! TIMEOUT  prompt=%s  attempt=%d  elapsed=%.1fs  '
                        'timeout_limit=%.1fs',
                        prompt_name, attempt, _elapsed, timeout_seconds,
                    )
                raise TimeoutError(
                    f'LLM call timed out after {timeout_seconds}s (prompt={prompt_name}, attempt={attempt})'
                )
            
            _elapsed = time() - _t0
            result = response.choices[0].message.content or ''
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason if response.choices else 'unknown'

            if _is_llm_trace_enabled():
                logger.info(
                    '[LLM_TRACE] <<< RECV  prompt=%s  attempt=%d  elapsed=%.1fs  '
                    'in_tok=%d  out_tok=%d  finish=%s  result_len=%d',
                    prompt_name, attempt, _elapsed, input_tokens, output_tokens,
                    finish_reason, len(result),
                )

            parsed = json.loads(result)
            return parsed, input_tokens, output_tokens
        except json.JSONDecodeError as e:
            _elapsed = time() - _t0
            if _is_llm_trace_enabled():
                logger.warning(
                    '[LLM_TRACE] !!! JSON_ERR  prompt=%s  attempt=%d  elapsed=%.1fs  '
                    'error=%s  result_preview=%.200s',
                    prompt_name, attempt, _elapsed, str(e)[:100],
                    result[:200] if 'result' in dir() else '(no result)',
                )
            raise
        except TimeoutError:
            # Re-raise TimeoutError to trigger retry
            raise
        except Exception as e:
            _elapsed = time() - _t0 if '_t0' in dir() else 0
            error_msg = str(e)
            if _is_llm_trace_enabled():
                logger.warning(
                    '[LLM_TRACE] !!! ERROR  prompt=%s  attempt=%d  elapsed=%.1fs  '
                    'type=%s  error=%.200s',
                    prompt_name, attempt, _elapsed, type(e).__name__, error_msg[:200],
                )
            if '429' in error_msg or 'rate' in error_msg.lower():
                raise RateLimitError(error_msg) from e
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        lang_instruction = get_extraction_language_instruction(group_id)
        if isinstance(messages[0].content, str):
            messages[0].content += lang_instruction
        else:
            messages[0].content.append({'type': 'text', 'text': lang_instruction})

        for message in messages:
            message.content = self._clean_input(message.content)

        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'bedrock',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            retry_count = 0
            last_error = None
            total_input_tokens = 0
            total_output_tokens = 0
            _start_time = time()

            if _is_llm_trace_enabled():
                logger.info(
                    '[LLM_TRACE] === BEGIN  prompt=%s  max_retries=%d',
                    prompt_name, self.MAX_RETRIES,
                )

            while retry_count <= self.MAX_RETRIES:
                try:
                    result, input_tokens, output_tokens = await self._generate_response(
                        messages, response_model, max_tokens, model_size,
                        prompt_name=prompt_name, attempt=retry_count,
                    )
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                    _total_elapsed = time() - _start_time
                    if _is_llm_trace_enabled():
                        logger.info(
                            '[LLM_TRACE] === END    prompt=%s  total_elapsed=%.1fs  '
                            'retries=%d  total_in=%d  total_out=%d',
                            prompt_name, _total_elapsed, retry_count,
                            total_input_tokens, total_output_tokens,
                        )

                    # Record token usage
                    self.token_tracker.record(prompt_name, total_input_tokens, total_output_tokens)

                    # Record to S3 logger
                    if self.s3_logger:
                        model = self.model or DEFAULT_MODEL
                        if model_size == ModelSize.small and self.small_model:
                            model = self.small_model
                        self.s3_logger.record(
                            operation='llm.generate',
                            model_id=model,
                            prompt_name=prompt_name,
                            group_id=group_id,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            latency_ms=(time() - _start_time) * 1000,
                            input_preview=(
                                messages[-1].content if messages and isinstance(messages[-1].content, str)
                                else '[multimodal]' if messages else None
                            ),
                        )

                    return result
                except RateLimitError:
                    span.set_status('error', str(last_error))
                    raise
                except TimeoutError as e:
                    # Timeout should trigger retry
                    last_error = e
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(
                            f'Max retries ({self.MAX_RETRIES}) exceeded after timeout. Last error: {e}'
                        )
                        span.set_status('error', str(e))
                        span.record_exception(e)
                        raise

                    retry_count += 1
                    logger.warning(
                        f'Retrying after timeout (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                    )
                    # Don't append error context for timeout - just retry immediately
                    continue
                except Exception as e:
                    last_error = e
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(
                            f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}'
                        )
                        span.set_status('error', str(e))
                        span.record_exception(e)
                        raise

                    retry_count += 1
                    error_context = (
                        f'The previous response was invalid. '
                        f'Error: {e.__class__.__name__}: {e}. '
                        f'Please try again with a valid JSON response.'
                    )
                    messages.append(Message(role='user', content=error_context))
                    logger.warning(
                        f'Retrying after error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                    )

            span.set_status('error', str(last_error))
            raise last_error or Exception('Max retries exceeded')
