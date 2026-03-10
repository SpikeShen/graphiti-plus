"""
Bedrock Reranker Client for Graphiti.

Uses OpenAI SDK with Bedrock Mantle endpoint for boolean relevance classification.
"""

import logging

import numpy as np
from openai import AsyncOpenAI

from ..helpers import semaphore_gather
from .client import CrossEncoderClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'moonshotai.kimi-k2.5'


class BedrockRerankerClient(CrossEncoderClient):
    """
    Reranker using Bedrock Mantle OpenAI-compatible endpoint.

    Runs a boolean classifier prompt for each passage and uses
    log-probabilities (when available) or text matching to score relevance.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model_id: str = DEFAULT_MODEL,
    ):
        self.model_id = model_id
        if client is not None:
            self.client = client
        else:
            raise ValueError('An AsyncOpenAI client must be provided')

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        openai_messages_list = [
            [
                {
                    'role': 'system',
                    'content': (
                        'You are an expert tasked with determining whether '
                        'the passage is relevant to the query.'
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        'Respond with "True" if PASSAGE is relevant to QUERY '
                        'and "False" otherwise.\n'
                        f'<PASSAGE>\n{passage}\n</PASSAGE>\n'
                        f'<QUERY>\n{query}\n</QUERY>'
                    ),
                },
            ]
            for passage in passages
        ]

        try:
            responses = await semaphore_gather(
                *[
                    self.client.chat.completions.create(
                        model=self.model_id,
                        messages=msgs,
                        temperature=0,
                        max_tokens=5,
                    )
                    for msgs in openai_messages_list
                ]
            )

            scores: list[float] = []
            for response in responses:
                text = (response.choices[0].message.content or '').strip().lower()
                if text.startswith('true'):
                    scores.append(1.0)
                elif text.startswith('false'):
                    scores.append(0.0)
                else:
                    scores.append(0.5)

            results = [
                (passage, score)
                for passage, score in zip(passages, scores, strict=True)
            ]
            results.sort(reverse=True, key=lambda x: x[1])
            return results
        except Exception as e:
            logger.error(f'Bedrock reranker error: {e}')
            raise
