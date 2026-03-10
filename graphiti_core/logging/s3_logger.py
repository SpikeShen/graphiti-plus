"""
S3 Invocation Logger for Graphiti.

Captures LLM and embedding invocation logs in JSON Lines format (gzip compressed),
following a path convention similar to Bedrock Model Invocation Logs:

    s3://{bucket}/GraphitiLogs/{year}/{month}/{day}/{hour}/{timestamp}_{uuid}.jsonl.gz

Designed for downstream Athena queries with Hive-style partitioning.
"""

import gzip
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import boto3

logger = logging.getLogger(__name__)


@dataclass
class LogRecord:
    """A single invocation log record."""

    timestamp: str
    operation: str  # 'llm.generate' | 'embedding.create' | 'embedding.create_batch'
    model_id: str
    prompt_name: str | None = None
    group_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    status: str = 'ok'  # 'ok' | 'error'
    error_message: str | None = None
    input_preview: str | None = None  # first N chars of input for debugging
    metadata: dict[str, Any] = field(default_factory=dict)


class S3InvocationLogger:
    """
    Buffered logger that writes invocation records to S3 as gzip JSON Lines.

    Records are buffered in memory and flushed to S3 either when the buffer
    reaches `flush_threshold` or when `flush()` is called explicitly.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Key prefix (default 'GraphitiLogs').
    region_name : str
        AWS region for the S3 client.
    flush_threshold : int
        Number of records to buffer before auto-flushing (default 50).
    input_preview_chars : int
        Max chars to capture from input for preview (default 200). Set 0 to disable.
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str | None = None,
        region_name: str | None = None,
        flush_threshold: int | None = None,
        input_preview_chars: int | None = None,
    ):
        self.bucket = bucket or os.environ.get('S3_LOG_BUCKET', '')
        self.prefix = (prefix or os.environ.get('S3_LOG_PREFIX', 'GraphitiLogs')).rstrip('/')
        self.region_name = region_name or os.environ.get('S3_LOG_REGION', os.environ.get('AWS_REGION', 'us-east-1'))
        self.flush_threshold = flush_threshold or int(os.environ.get('S3_LOG_FLUSH_THRESHOLD', '50'))
        self.input_preview_chars = input_preview_chars if input_preview_chars is not None else int(os.environ.get('S3_LOG_PREVIEW_CHARS', '200'))

        if not self.bucket:
            raise ValueError(
                'S3 log bucket is required. Set S3_LOG_BUCKET env var or pass bucket parameter.'
            )

        self._buffer: list[LogRecord] = []
        self._lock = threading.Lock()
        self._s3 = boto3.client('s3', region_name=self.region_name)

    def record(
        self,
        operation: str,
        model_id: str,
        prompt_name: str | None = None,
        group_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        status: str = 'ok',
        error_message: str | None = None,
        input_preview: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a log record to the buffer. Auto-flushes when threshold is reached."""
        now = datetime.now(timezone.utc)

        # Truncate input preview
        preview = None
        if input_preview and self.input_preview_chars > 0:
            preview = input_preview[: self.input_preview_chars]

        rec = LogRecord(
            timestamp=now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            operation=operation,
            model_id=model_id,
            prompt_name=prompt_name,
            group_id=group_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 1),
            status=status,
            error_message=error_message,
            input_preview=preview,
            metadata=metadata or {},
        )

        with self._lock:
            self._buffer.append(rec)
            if len(self._buffer) >= self.flush_threshold:
                self._flush_locked()

    def flush(self) -> None:
        """Flush buffered records to S3."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Internal flush, must be called with self._lock held."""
        if not self._buffer:
            return

        records = self._buffer[:]
        self._buffer.clear()

        try:
            self._write_to_s3(records)
        except Exception:
            logger.exception('Failed to flush %d log records to S3', len(records))
            # Put records back so they aren't lost
            self._buffer = records + self._buffer

    def _write_to_s3(self, records: list[LogRecord]) -> None:
        """Compress records as JSON Lines and upload to S3."""
        now = datetime.now(timezone.utc)
        key = (
            f'{self.prefix}'
            f'/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour={now.hour:02d}'
            f'/{now.strftime("%Y%m%dT%H%M%S")}_{uuid.uuid4().hex[:12]}.jsonl.gz'
        )

        lines = []
        for rec in records:
            d = asdict(rec)
            # Remove None values for cleaner output
            d = {k: v for k, v in d.items() if v is not None}
            lines.append(json.dumps(d, ensure_ascii=False))

        payload = gzip.compress('\n'.join(lines).encode('utf-8'))

        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType='application/x-ndjson',
            ContentEncoding='gzip',
        )
        logger.info('Flushed %d log records to s3://%s/%s', len(records), self.bucket, key)

    def __del__(self):
        """Best-effort flush on garbage collection."""
        try:
            self.flush()
        except Exception:
            pass


def create_s3_logger(**kwargs) -> S3InvocationLogger | None:
    """Create an S3InvocationLogger if S3_LOG_ENABLED=true (or bucket is configured).

    Returns None if logging is disabled, allowing zero-config opt-out.
    All parameters fall back to environment variables.

    Env vars:
        S3_LOG_ENABLED         - 'true' to enable (default: 'true' if S3_LOG_BUCKET is set)
        S3_LOG_BUCKET          - S3 bucket name (required)
        S3_LOG_PREFIX          - Key prefix (default: 'GraphitiLogs')
        S3_LOG_REGION          - AWS region (default: AWS_REGION or 'us-east-1')
        S3_LOG_FLUSH_THRESHOLD - Buffer size before auto-flush (default: 50)
        S3_LOG_PREVIEW_CHARS   - Max input preview chars (default: 200, 0 to disable)
    """
    enabled = os.environ.get('S3_LOG_ENABLED', '').lower()
    bucket = kwargs.get('bucket') or os.environ.get('S3_LOG_BUCKET', '')

    # Auto-enable if bucket is set, unless explicitly disabled
    if enabled == 'false':
        return None
    if not enabled and not bucket:
        return None

    try:
        return S3InvocationLogger(**kwargs)
    except Exception:
        logger.warning('Failed to create S3InvocationLogger, logging disabled', exc_info=True)
        return None
