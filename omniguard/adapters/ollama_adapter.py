"""
omniguard.adapters.ollama_adapter
===================================
Sends attack vectors to Ollama (free, runs locally).
"""
from __future__ import annotations
import time
import logging
import asyncio
from omniguard.core.models import AttackVector
from omniguard.adapters.base import BaseAdapter, ModelResponse

logger = logging.getLogger(__name__)

class OllamaAdapter(BaseAdapter):
    def __init__(self, model: str = "llama3.2") -> None:
        self._model = model
        import ollama
        self._client = ollama

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def send_attack(self, vector: AttackVector) -> ModelResponse:
        start = time.monotonic()
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.chat(
                    model=self._model,
                    messages=[{"role": "user", "content": vector.prompt}],
                )
            )
            latency_ms = (time.monotonic() - start) * 1000
            response_text = response.message.content or ""
            return ModelResponse(
                vector_id=vector.vector_id,
                prompt_sent=vector.prompt,
                response_text=response_text,
                model_name=self._model,
                provider=self.provider,
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error("[Ollama] Error: %s", exc)
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