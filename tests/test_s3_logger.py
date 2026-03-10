"""
Unit tests for S3InvocationLogger (Chapter 7 — S3 调用日志系统).

All tests use mock mode — no real AWS calls.
"""

import gzip
import json
import os
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.logging.s3_logger import LogRecord, S3InvocationLogger, create_s3_logger


# ---------------------------------------------------------------------------
# LogRecord
# ---------------------------------------------------------------------------

class TestLogRecord:
    def test_defaults(self):
        rec = LogRecord(timestamp='2026-01-01T00:00:00Z', operation='llm.generate', model_id='m1')
        assert rec.status == 'ok'
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0
        assert rec.metadata == {}
        assert rec.error_message is None

    def test_serialization_round_trip(self):
        rec = LogRecord(
            timestamp='2026-01-01T00:00:00Z',
            operation='embedding.create',
            model_id='nova-embed',
            prompt_name='get_entities',
            input_tokens=100,
            output_tokens=50,
            latency_ms=123.4,
        )
        d = asdict(rec)
        assert d['operation'] == 'embedding.create'
        assert d['latency_ms'] == 123.4
        # JSON round-trip
        assert json.loads(json.dumps(d)) == d


# ---------------------------------------------------------------------------
# S3InvocationLogger
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_s3():
    with patch('graphiti_core.logging.s3_logger.boto3') as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        yield mock_client


class TestS3InvocationLogger:
    def _make_logger(self, mock_s3, **kwargs):
        """Helper to create a logger with mocked S3 client."""
        defaults = dict(bucket='test-bucket', prefix='TestLogs', region_name='us-east-1',
                        flush_threshold=5, input_preview_chars=50)
        defaults.update(kwargs)
        lgr = S3InvocationLogger(**defaults)
        lgr._s3 = mock_s3  # replace real client with mock
        return lgr

    def test_init_requires_bucket(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='bucket is required'):
                S3InvocationLogger(bucket='')

    def test_init_from_env(self, mock_s3):
        env = {'S3_LOG_BUCKET': 'env-bucket', 'S3_LOG_PREFIX': 'EnvPrefix',
               'S3_LOG_FLUSH_THRESHOLD': '10', 'S3_LOG_PREVIEW_CHARS': '100'}
        with patch.dict(os.environ, env, clear=False):
            lgr = S3InvocationLogger()
            lgr._s3 = mock_s3
            assert lgr.bucket == 'env-bucket'
            assert lgr.prefix == 'EnvPrefix'
            assert lgr.flush_threshold == 10
            assert lgr.input_preview_chars == 100

    def test_record_buffers(self, mock_s3):
        lgr = self._make_logger(mock_s3, flush_threshold=10)
        lgr.record(operation='llm.generate', model_id='m1', input_tokens=5)
        assert len(lgr._buffer) == 1
        mock_s3.put_object.assert_not_called()

    def test_input_preview_truncation(self, mock_s3):
        lgr = self._make_logger(mock_s3, input_preview_chars=10)
        lgr.record(operation='llm.generate', model_id='m1',
                   input_preview='a' * 100)
        assert len(lgr._buffer[0].input_preview) == 10

    def test_input_preview_disabled(self, mock_s3):
        lgr = self._make_logger(mock_s3, input_preview_chars=0)
        lgr.record(operation='llm.generate', model_id='m1',
                   input_preview='hello')
        assert lgr._buffer[0].input_preview is None

    def test_auto_flush_on_threshold(self, mock_s3):
        lgr = self._make_logger(mock_s3, flush_threshold=3)
        for i in range(3):
            lgr.record(operation='llm.generate', model_id='m1', input_tokens=i)
        # Buffer should be empty after auto-flush
        assert len(lgr._buffer) == 0
        mock_s3.put_object.assert_called_once()

    def test_flush_writes_gzip_jsonl(self, mock_s3):
        lgr = self._make_logger(mock_s3, flush_threshold=100)
        lgr.record(operation='llm.generate', model_id='m1', input_tokens=10, output_tokens=5)
        lgr.record(operation='embedding.create', model_id='nova', input_tokens=20)
        lgr.flush()

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs['Bucket'] == 'test-bucket'
        assert call_kwargs['ContentEncoding'] == 'gzip'
        # Key follows Hive partitioning
        key = call_kwargs['Key']
        assert key.startswith('TestLogs/year=')
        assert '/month=' in key
        assert key.endswith('.jsonl.gz')
        # Decompress and verify content
        raw = gzip.decompress(call_kwargs['Body']).decode('utf-8')
        lines = raw.strip().split('\n')
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        assert rec0['operation'] == 'llm.generate'
        assert rec0['input_tokens'] == 10

    def test_flush_empty_buffer_noop(self, mock_s3):
        lgr = self._make_logger(mock_s3)
        lgr.flush()
        mock_s3.put_object.assert_not_called()

    def test_flush_failure_restores_buffer(self, mock_s3):
        mock_s3.put_object.side_effect = Exception('S3 error')
        lgr = self._make_logger(mock_s3, flush_threshold=100)
        lgr.record(operation='llm.generate', model_id='m1')
        lgr.flush()
        # Records should be put back
        assert len(lgr._buffer) == 1
        assert lgr._buffer[0].operation == 'llm.generate'

    def test_none_values_stripped_from_output(self, mock_s3):
        lgr = self._make_logger(mock_s3, flush_threshold=100)
        lgr.record(operation='llm.generate', model_id='m1')
        lgr.flush()
        raw = gzip.decompress(mock_s3.put_object.call_args.kwargs['Body']).decode()
        rec = json.loads(raw)
        # Fields that are None should not appear
        assert 'prompt_name' not in rec
        assert 'error_message' not in rec

    def test_latency_rounded(self, mock_s3):
        lgr = self._make_logger(mock_s3, flush_threshold=100)
        lgr.record(operation='llm.generate', model_id='m1', latency_ms=123.456789)
        assert lgr._buffer[0].latency_ms == 123.5


# ---------------------------------------------------------------------------
# create_s3_logger factory
# ---------------------------------------------------------------------------

class TestCreateS3Logger:
    def test_disabled_explicitly(self):
        with patch.dict(os.environ, {'S3_LOG_ENABLED': 'false', 'S3_LOG_BUCKET': 'b'}):
            assert create_s3_logger() is None

    def test_no_bucket_no_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env vars
            for k in ('S3_LOG_ENABLED', 'S3_LOG_BUCKET'):
                os.environ.pop(k, None)
            assert create_s3_logger() is None

    @patch('graphiti_core.logging.s3_logger.boto3')
    def test_enabled_with_bucket(self, mock_boto3):
        mock_boto3.client.return_value = MagicMock()
        with patch.dict(os.environ, {'S3_LOG_BUCKET': 'my-bucket'}):
            lgr = create_s3_logger()
            assert lgr is not None
            assert lgr.bucket == 'my-bucket'

    @patch('graphiti_core.logging.s3_logger.boto3')
    def test_auto_enable_when_bucket_set(self, mock_boto3):
        mock_boto3.client.return_value = MagicMock()
        env = {'S3_LOG_BUCKET': 'auto-bucket'}
        # S3_LOG_ENABLED not set — should auto-enable
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop('S3_LOG_ENABLED', None)
            lgr = create_s3_logger()
            assert lgr is not None

    def test_creation_failure_returns_none(self):
        with patch.dict(os.environ, {'S3_LOG_BUCKET': 'b', 'S3_LOG_ENABLED': 'true'}):
            with patch('graphiti_core.logging.s3_logger.boto3') as mock_boto3:
                mock_boto3.client.side_effect = Exception('boom')
                assert create_s3_logger() is None
