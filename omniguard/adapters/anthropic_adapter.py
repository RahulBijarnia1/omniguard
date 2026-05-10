"""
omniguard.adapters.anthropic_adapter
=====================================
Sends attack vectors to Anthropic Claude API.
"""

from __future__ import annotations

import time
import os
import logging
import asyncio
from typing import Any

import anthropic

from omniguard.core.models import AttackVector
from omniguard.adapters.base import BaseAdapter, ModelResponse

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    """
    Adapter for Anthropic Claude models.

    Usage::

        adapter = AnthropicAdapter(api_key="sk-ant-...")
        response = await adapter.send_attack(vector)
        print(response.response_text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        """
        Args:
            api_key:    Anthropic API key. If None, reads from
                        ANTHROPIC_API_KEY environment variable.
            model:      Model to use. Defaults to claude-haiku (fastest/cheapest).
            max_tokens: Maximum tokens in response.
            timeout:    Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "No Anthropic API key provided! "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key="
            )
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = anthropic.Anthropic(api_key=self._api_key)

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def send_attack(self, vector: AttackVector) -> ModelResponse:
        """Send attack vector to Claude and return response."""
        start = time.monotonic()
        try:
            # Run in thread pool since anthropic SDK is sync
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[{"role": "user", "content": vector.prompt}],
                )
            )

            latency_ms = (time.monotonic() - start) * 1000
            response_text = response.content[0].text if response.content else ""

            logger.info(
                "[Claude] Vector %s | %d chars response | %.0fms",
                vector.vector_id[:8],
                len(response_text),
                latency_ms,
            )

            return ModelResponse(
                vector_id=vector.vector_id,
                prompt_sent=vector.prompt,
                response_text=response_text,
                model_name=self._model,
                provider=self.provider,
                latency_ms=latency_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                success=True,
            )

        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error("[Claude] Error: %s", exc)
            return ModelResponse(
                vector_id=vector.vector_id,
                prompt_sent=vector.prompt,
                response_text="",
                model_name=self._model,
                provider=self.provider,
                latency_ms=latency_ms,
                error=str(exc),
                success=False,
            )