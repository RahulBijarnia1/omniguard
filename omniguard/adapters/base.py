"""
omniguard.adapters.base
========================
Base interface that ALL model adapters must implement.
This is what makes the system model-agnostic — swap Claude
for OpenAI by just changing one line in your config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from omniguard.core.models import AttackVector


@dataclass
class ModelResponse:
    """
    A single model response to an attack vector.
    This is the unit passed to the Judge (Module 3).
    """

    vector_id: str
    prompt_sent: str
    response_text: str
    model_name: str
    provider: str  # "anthropic" or "openai"

    # Timing
    latency_ms: float = 0.0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Error tracking
    error: str | None = None
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for saving to file."""
        return {
            "vector_id": self.vector_id,
            "prompt_sent": self.prompt_sent,
            "response_text": self.response_text,
            "model_name": self.model_name,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "success": self.success,
            "created_at": self.created_at.isoformat(),
        }


class BaseAdapter(ABC):
    """
    Abstract base class for all LLM adapters.
    Every adapter (Claude, OpenAI, Ollama) must implement these methods.
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name e.g. 'anthropic', 'openai'"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Full model name e.g. 'claude-sonnet-4-20250514'"""
        ...

    @abstractmethod
    async def send_attack(self, vector: AttackVector) -> ModelResponse:
        """
        Send a single attack vector to the model and return its response.

        Args:
            vector: The attack vector to send.

        Returns:
            ModelResponse with the model's reply and metadata.
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the adapter can reach the API.
        Returns True if healthy, False otherwise.
        """
        try:
            test_vector = AttackVector(
                prompt="Say the word OK and nothing else.",
                category=__import__(
                    "omniguard.core.models", fromlist=["AttackCategory"]
                ).AttackCategory.PROMPT_INJECTION,
                seed=0,
            )
            response = await self.send_attack(test_vector)
            return response.success
        except Exception:
            return False 