"""
OmniGuard — Full Pipeline
Module 1 (Generate) + Module 2 (Fire) + Module 3 (Judge + Report)
"""

import asyncio

from omniguard.core.models import CampaignConfig, AttackCategory
from omniguard.generators.adversarial import AdversarialPromptGenerator
from omniguard.adapters.ollama_adapter import OllamaAdapter
from omniguard.adapters.runner import CampaignRunner
from omniguard.judge.scorer import Judge
from omniguard.judge.reporter import Reporter


async def main() -> None:

    # ── Module 1: Generate attack vectors ────────────────────────────
    print("=" * 55)
    print("  OmniGuard — Full Security Audit Pipeline")
    print("=" * 55)
    print("\n[Module 1] Generating attack vectors...")

    config = CampaignConfig(
        name="omniguard-full-run",
        seed=42,
        max_vectors_per_category=3,
        mutation_passes=0,
        enabled_categories=[
            AttackCategory.JAILBREAK,
            AttackCategory.PROMPT_INJECTION,
            AttackCategory.PAYLOAD_SMUGGLING,
            AttackCategory.OBFUSCATION,
        ],
    )

    generator = AdversarialPromptGenerator(config=config)
    all_vectors = []

    for category in config.enabled_categories:
        vectors, stats = await generator.generate_async(category)
        all_vectors.extend(vectors)
        print(f"  {category.value}: {len(vectors)} vectors")

    print(f"  Total: {len(all_vectors)} vectors ready!\n")

    # ── Module 2: Fire attacks at Ollama ─────────────────────────────
    print("[Module 2] Firing attacks at Llama3.2...")

    adapters = [OllamaAdapter(model="llama3.2")]

    runner = CampaignRunner(
        adapters=adapters,
        output_dir="results",
        show_live=True,
    )
    responses = await runner.run(all_vectors)

    # ── Module 3: Judge scores every response ─────────────────────────
    print("[Module 3] Judge scoring responses...")

    judge = Judge()
    scores = judge.score_all(all_vectors, responses)

    # ── Module 3: Print + save report ────────────────────────────────
    reporter = Reporter(output_dir="results")
    reporter.print_report(scores)
    reporter.save_markdown(scores)

    print("\nAll done! Check your results/ folder for the full report!\n")


if __name__ == "__main__":
    asyncio.run(main())