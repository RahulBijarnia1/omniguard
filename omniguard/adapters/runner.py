"""
omniguard.adapters.runner
==========================
Sends attack vectors to ALL configured adapters,
shows results live in terminal, and saves to a JSON file.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omniguard.core.models import AttackVector
from omniguard.adapters.base import BaseAdapter, ModelResponse

logger = logging.getLogger(__name__)


class CampaignRunner:
    """
    Runs attack vectors against multiple AI models simultaneously.

    Shows live results in terminal and saves everything to a JSON file.

    Usage::

        runner = CampaignRunner(
            adapters=[claude_adapter, openai_adapter],
            output_dir="results"
        )
        all_responses = await runner.run(vectors)
    """

    def __init__(
        self,
        adapters: list[BaseAdapter],
        output_dir: str = "results",
        show_live: bool = True,
    ) -> None:
        self.adapters = adapters
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.show_live = show_live

    async def run(
        self, vectors: list[AttackVector]
    ) -> list[ModelResponse]:
        """
        Send all vectors to all adapters.

        Args:
            vectors: Attack vectors from Module 1.

        Returns:
            All responses from all models.
        """
        print("\n" + "="*60)
        print("  OmniGuard — Attack Campaign Starting")
        print("="*60)
        print(f"  Vectors : {len(vectors)}")
        print(f"  Models  : {[a.model_name for a in self.adapters]}")
        print("="*60 + "\n")

        all_responses: list[ModelResponse] = []

        for i, vector in enumerate(vectors, 1):
            print(f">>> Attack {i}/{len(vectors)}")
            print(f"    Prompt: {vector.prompt[:80]}...")
            print()

            # Send to all adapters at the same time
            tasks = [adapter.send_attack(vector) for adapter in self.adapters]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for adapter, response in zip(self.adapters, responses):
                if isinstance(response, Exception):
                    print(f"    [{adapter.provider.upper()}] ERROR: {response}")
                    continue

                all_responses.append(response)

                if self.show_live:
                    self._print_response(response)

            print("-" * 60 + "\n")

        # Save all results to file
        output_file = self._save_results(vectors, all_responses)

        print("\n" + "="*60)
        print(f"  Campaign Complete!")
        print(f"  Total responses : {len(all_responses)}")
        print(f"  Results saved to: {output_file}")
        print("="*60 + "\n")

        return all_responses

    def _print_response(self, response: ModelResponse) -> None:
        """Print a single response live in the terminal."""
        status = "✅ RESPONDED" if response.success else "❌ ERROR"
        provider = response.provider.upper()

        print(f"    [{provider}] {status} | {response.latency_ms:.0f}ms | "
              f"{response.output_tokens} tokens")

        if response.success and response.response_text:
            # Show first 150 chars of response
            preview = response.response_text[:150].replace("\n", " ")
            print(f"    Response: {preview}...")
        elif response.error:
            print(f"    Error: {response.error}")
        print()

    def _save_results(
        self,
        vectors: list[AttackVector],
        responses: list[ModelResponse],
    ) -> Path:
        """Save all results to a timestamped JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"campaign_{timestamp}.json"

        data: dict[str, Any] = {
            "campaign_timestamp": timestamp,
            "total_vectors": len(vectors),
            "total_responses": len(responses),
            "models_tested": list({r.model_name for r in responses}),
            "vectors": [
                {
                    "vector_id": v.vector_id,
                    "category": v.category.value,
                    "severity": v.severity,
                    "prompt": v.prompt,
                }
                for v in vectors
            ],
            "responses": [r.to_dict() for r in responses],
        }

        filename.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return filename