"""
Unit tests for BedrockLLMClient timeout / retry mechanism (Chapter 7).

All tests use mock mode — no real AWS/LLM calls.
"""

import asyncio
import json
import os
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.llm_client.bedrock_client import (
    PROMPT_TIMEOUTS,
    BedrockLLMClient,
    _is_llm_trace_enabled,
    _load_prompt_timeouts,
)
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message

pytest_plugins = ('pytest_asyncio',)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_response(content: str = '{"result": "ok"}',
                        input_tokens: int = 10,
                        output_tokens: int = 5,
                        finish_reason: str = 'stop'):
    """Build a fake OpenAI ChatCompletion response object."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def bedrock_client():
    """Create a BedrockLLMClient with mocked token generation and AsyncOpenAI."""
    with patch('graphiti_core.llm_client.bedrock_client._generate_bedrock_token', return_value='fake-token'):
        client = BedrockLLMClient(
            config=LLMConfig(model='test-model'),
            region_name='us-east-1',
        )
    # Replace the async openai client with a mock
    client.client = MagicMock()
    client.client.chat = MagicMock()
    client.client.chat.completions = MagicMock()
    client.client.chat.completions.create = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# PROMPT_TIMEOUTS config
# ---------------------------------------------------------------------------

class TestPromptTimeouts:
    """Verify default values and env-var override for PROMPT_TIMEOUTS."""

    # --- defaults (no env override) ---
    def test_default_timeout(self):
        assert PROMPT_TIMEOUTS['default'] == 15.0

    def test_node_extract_txt(self):
        assert PROMPT_TIMEOUTS['extract_nodes.extract_text'] == 15.0
        assert PROMPT_TIMEOUTS['extract_nodes.extract_message'] == 15.0
        assert PROMPT_TIMEOUTS['extract_nodes.extract_json'] == 15.0

    def test_node_extract_mm(self):
        assert PROMPT_TIMEOUTS['extract_nodes.extract_document'] == 30.0

    def test_edge_extract_txt(self):
        assert PROMPT_TIMEOUTS['extract_edges.edge'] == 60.0

    def test_edge_extract_mm(self):
        assert PROMPT_TIMEOUTS['extract_edges.edge_document'] == 90.0

    def test_unknown_prompt_gets_default(self):
        val = PROMPT_TIMEOUTS.get('nonexistent', PROMPT_TIMEOUTS['default'])
        assert val == 15.0

    # --- env-var overrides ---
    def test_env_override_default(self):
        with patch.dict(os.environ, {'LLM_TIMEOUT_DEFAULT': '20'}):
            t = _load_prompt_timeouts()
        assert t['default'] == 20.0

    def test_env_override_node_extract_txt(self):
        with patch.dict(os.environ, {'LLM_TIMEOUT_NODE_EXTRACT_TXT': '25'}):
            t = _load_prompt_timeouts()
        assert t['extract_nodes.extract_text'] == 25.0
        assert t['extract_nodes.extract_message'] == 25.0
        assert t['extract_nodes.extract_json'] == 25.0

    def test_env_override_node_extract_mm(self):
        with patch.dict(os.environ, {'LLM_TIMEOUT_NODE_EXTRACT_MM': '45'}):
            t = _load_prompt_timeouts()
        assert t['extract_nodes.extract_document'] == 45.0

    def test_env_override_edge_extract_txt(self):
        with patch.dict(os.environ, {'LLM_TIMEOUT_EDGE_EXTRACT_TXT': '120'}):
            t = _load_prompt_timeouts()
        assert t['extract_edges.edge'] == 120.0

    def test_env_override_edge_extract_mm(self):
        with patch.dict(os.environ, {'LLM_TIMEOUT_EDGE_EXTRACT_MM': '180'}):
            t = _load_prompt_timeouts()
        assert t['extract_edges.edge_document'] == 180.0

    def test_env_invalid_value_uses_fallback(self):
        with patch.dict(os.environ, {'LLM_TIMEOUT_DEFAULT': 'abc'}):
            t = _load_prompt_timeouts()
        assert t['default'] == 15.0


# ---------------------------------------------------------------------------
# _is_llm_trace_enabled
# ---------------------------------------------------------------------------

class TestLLMTraceEnabled:
    def test_default_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GRAPHITI_LLM_TRACE', None)
            assert _is_llm_trace_enabled() is False

    def test_enabled_true(self):
        with patch.dict(os.environ, {'GRAPHITI_LLM_TRACE': 'true'}):
            assert _is_llm_trace_enabled() is True

    def test_enabled_1(self):
        with patch.dict(os.environ, {'GRAPHITI_LLM_TRACE': '1'}):
            assert _is_llm_trace_enabled() is True

    def test_disabled_false(self):
        with patch.dict(os.environ, {'GRAPHITI_LLM_TRACE': 'false'}):
            assert _is_llm_trace_enabled() is False


# ---------------------------------------------------------------------------
# _generate_response
# ---------------------------------------------------------------------------

class TestGenerateResponseInternal:
    @pytest.mark.asyncio
    async def test_success_returns_tuple(self, bedrock_client):
        bedrock_client.client.chat.completions.create.return_value = _make_chat_response(
            '{"edges": []}', input_tokens=20, output_tokens=10
        )
        msgs = [Message(role='system', content='sys'), Message(role='user', content='hi')]
        result, in_tok, out_tok = await bedrock_client._generate_response(
            msgs, prompt_name='test_prompt'
        )
        assert result == {'edges': []}
        assert in_tok == 20
        assert out_tok == 10

    @pytest.mark.asyncio
    async def test_json_decode_error_raises(self, bedrock_client):
        bedrock_client.client.chat.completions.create.return_value = _make_chat_response(
            'not json at all'
        )
        msgs = [Message(role='system', content='sys'), Message(role='user', content='hi')]
        with pytest.raises(json.JSONDecodeError):
            await bedrock_client._generate_response(msgs, prompt_name='test')

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self, bedrock_client):
        """Simulate asyncio.timeout triggering by making the API call hang."""
        async def slow_call(**kwargs):
            await asyncio.sleep(100)  # will be cancelled by timeout

        bedrock_client.client.chat.completions.create = slow_call
        msgs = [Message(role='system', content='sys'), Message(role='user', content='hi')]

        # Use a very short timeout to trigger quickly
        with patch.dict(
            'graphiti_core.llm_client.bedrock_client.PROMPT_TIMEOUTS',
            {'default': 0.05},
        ):
            with pytest.raises(TimeoutError, match='timed out'):
                await bedrock_client._generate_response(msgs, prompt_name='fast_test')

    @pytest.mark.asyncio
    async def test_rate_limit_error_converted(self, bedrock_client):
        bedrock_client.client.chat.completions.create.side_effect = Exception(
            'ThrottlingException: 429 rate limit'
        )
        msgs = [Message(role='system', content='sys'), Message(role='user', content='hi')]
        with pytest.raises(RateLimitError):
            await bedrock_client._generate_response(msgs, prompt_name='test')


# ---------------------------------------------------------------------------
# generate_response (outer retry loop)
# ---------------------------------------------------------------------------

class TestGenerateResponseOuter:
    @pytest.mark.asyncio
    async def test_success_first_try(self, bedrock_client):
        bedrock_client.client.chat.completions.create.return_value = _make_chat_response(
            '{"ok": true}', input_tokens=15, output_tokens=8
        )
        msgs = [Message(role='system', content='sys'), Message(role='user', content='q')]
        result = await bedrock_client.generate_response(msgs, prompt_name='test')
        assert result == {'ok': True}

    @pytest.mark.asyncio
    async def test_retry_on_json_error(self, bedrock_client):
        """First call returns bad JSON, second call succeeds."""
        bedrock_client.client.chat.completions.create.side_effect = [
            _make_chat_response('bad json'),
            _make_chat_response('{"fixed": true}', input_tokens=5, output_tokens=3),
        ]
        msgs = [Message(role='system', content='sys'), Message(role='user', content='q')]
        result = await bedrock_client.generate_response(msgs, prompt_name='test')
        assert result == {'fixed': True}
        # Should have appended error context message for retry
        assert len(msgs) == 3  # original 2 + 1 error context

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, bedrock_client):
        """First call times out, second call succeeds."""
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(100)  # will timeout
            return _make_chat_response('{"retry": true}')

        bedrock_client.client.chat.completions.create = side_effect

        with patch.dict(
            'graphiti_core.llm_client.bedrock_client.PROMPT_TIMEOUTS',
            {'default': 0.05},
        ):
            msgs = [Message(role='system', content='sys'), Message(role='user', content='q')]
            result = await bedrock_client.generate_response(msgs, prompt_name='test')
            assert result == {'retry': True}
            # Timeout retry should NOT append error context message
            assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises(self, bedrock_client):
        """All attempts fail with JSON error → should raise after MAX_RETRIES."""
        bedrock_client.client.chat.completions.create.return_value = _make_chat_response(
            'always bad'
        )
        msgs = [Message(role='system', content='sys'), Message(role='user', content='q')]
        with pytest.raises(json.JSONDecodeError):
            await bedrock_client.generate_response(msgs, prompt_name='test')
        # Should have tried 1 + MAX_RETRIES times
        assert bedrock_client.client.chat.completions.create.call_count == 1 + BedrockLLMClient.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_rate_limit_not_retried(self, bedrock_client):
        """RateLimitError should propagate immediately without retry."""
        bedrock_client.client.chat.completions.create.side_effect = Exception(
            '429 rate limit exceeded'
        )
        msgs = [Message(role='system', content='sys'), Message(role='user', content='q')]
        with pytest.raises(RateLimitError):
            await bedrock_client.generate_response(msgs, prompt_name='test')

    @pytest.mark.asyncio
    async def test_token_accumulation_across_retries(self, bedrock_client):
        """Failed JSON parse raises before unpacking tokens, so only successful attempt counts."""
        bedrock_client.client.chat.completions.create.side_effect = [
            _make_chat_response('bad json', input_tokens=10, output_tokens=5),
            _make_chat_response('{"ok": 1}', input_tokens=12, output_tokens=8),
        ]
        msgs = [Message(role='system', content='sys'), Message(role='user', content='q')]
        await bedrock_client.generate_response(msgs, prompt_name='test')
        # JSONDecodeError is raised inside _generate_response after API call but before
        # returning the tuple, so the failed attempt's tokens are NOT accumulated.
        total = bedrock_client.token_tracker.get_total_usage()
        assert total.input_tokens == 12
        assert total.output_tokens == 8

    @pytest.mark.asyncio
    async def test_s3_logger_called_on_success(self, bedrock_client):
        mock_logger = MagicMock()
        bedrock_client.s3_logger = mock_logger
        bedrock_client.client.chat.completions.create.return_value = _make_chat_response(
            '{"ok": 1}', input_tokens=10, output_tokens=5
        )
        msgs = [Message(role='system', content='sys'), Message(role='user', content='q')]
        await bedrock_client.generate_response(msgs, prompt_name='my_prompt', group_id='g1')

        mock_logger.record.assert_called_once()
        call_kwargs = mock_logger.record.call_args.kwargs
        assert call_kwargs['operation'] == 'llm.generate'
        assert call_kwargs['prompt_name'] == 'my_prompt'
        assert call_kwargs['group_id'] == 'g1'
        assert call_kwargs['input_tokens'] == 10
        assert call_kwargs['output_tokens'] == 5
        assert call_kwargs['latency_ms'] > 0

    @pytest.mark.asyncio
    async def test_s3_logger_not_called_on_failure(self, bedrock_client):
        mock_logger = MagicMock()
        bedrock_client.s3_logger = mock_logger
        bedrock_client.client.chat.completions.create.return_value = _make_chat_response('bad')
        msgs = [Message(role='system', content='sys'), Message(role='user', content='q')]
        with pytest.raises(json.JSONDecodeError):
            await bedrock_client.generate_response(msgs, prompt_name='test')
        mock_logger.record.assert_not_called()
