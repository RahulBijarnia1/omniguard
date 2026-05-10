"""
omniguard.adapters.openai_adapter
===================================
Sends attack vectors to OpenAI GPT API.
"""

from __future__ import annotations

import time
import os
import logging
import asyncio

from omniguard.core.models import AttackVector
from omniguard.adapters.base import BaseAdapter, ModelResponse

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI GPT models.

    Usage::

        adapter = OpenAIAdapter(api_key="sk-...")
        response = await adapter.send_attack(vector)
        print(response.response_text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        """
        Args:
            api_key:    OpenAI API key. If None, reads from
                        OPENAI_API_KEY environment variable.
            model:      Model to use. Defaults to gpt-4o-mini (fastest/cheapest).
            max_tokens: Maximum tokens in response.
            timeout:    Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "No OpenAI API key provided! "
                "Set OPENAI_API_KEY environment variable or pass api_key="
            )
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout

        # Import here so missing package gives a clear error
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed! Run: pip install openai"
            )

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def send_attack(self, vector: AttackVector) -> ModelResponse:
        """Send attack vector to OpenAI and return response."""
        start = time.monotonic()
        try:
            # Run in thread pool since openai SDK is sync
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[{"role": "user", "content": vector.prompt}],
                )
            )

            latency_ms = (time.monotonic() - start) * 1000
            response_text = response.choices[0].message.content or ""

            logger.info(
                "[OpenAI] Vector %s | %d chars response | %.0fms",
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
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                success=True,
            )

        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error("[OpenAI] Error: %s", exc)
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