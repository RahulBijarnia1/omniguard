"""
OmniGuard — COMPLETE Pipeline
Module 1 + Module 2 + Module 3 + Module 4
"""

import asyncio

from omniguard.core.models import CampaignConfig, AttackCategory
from omniguard.generators.adversarial import AdversarialPromptGenerator
from omniguard.adapters.ollama_adapter import OllamaAdapter
from omniguard.adapters.runner import CampaignRunner
from omniguard.judge.scorer import Judge
from omniguard.judge.reporter import Reporter
from omniguard.reporting.html_report import HTMLReportGenerator


async def main() -> None:

    print("=" * 55)
    print("  OmniGuard — Complete Security Audit")
    print("  All 4 Modules Running")
    print("=" * 55)

    # ── Module 1: Generate attacks ────────────────────────────────────
    print("\n[Module 1] Generating attack vectors...")
    config = CampaignConfig(
        name="omniguard-complete",
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
        vectors, _ = await generator.generate_async(category)
        all_vectors.extend(vectors)
        print(f"  {category.value}: {len(vectors)} vectors")
    print(f"  Total: {len(all_vectors)} vectors ready!")

    # ── Module 2: Fire at Ollama ──────────────────────────────────────
    print("\n[Module 2] Firing attacks at Llama3.2...")
    adapters = [OllamaAdapter(model="llama3.2")]
    runner = CampaignRunner(
        adapters=adapters,
        output_dir="results",
        show_live=True,
    )
    responses = await runner.run(all_vectors)

    # ── Module 3: Judge scoring ───────────────────────────────────────
    print("[Module 3] Scoring responses...")
    judge = Judge()
    scores = judge.score_all(all_vectors, responses)
    reporter = Reporter(output_dir="results")
    reporter.print_report(scores)
    reporter.save_markdown(scores)

    # ── Module 4: HTML Report ─────────────────────────────────────────
    print("\n[Module 4] Generating HTML report...")
    html_gen = HTMLReportGenerator(output_dir="results")
    html_path = html_gen.generate(scores, model_name="Llama3.2")

    print("\n" + "=" * 55)
    print("  ALL DONE! OmniGuard Complete! 🎉")
    print("=" * 55)
    print(f"\n  Open this file in your browser:")
    print(f"  {html_path}")
    print("\n  You have a professional security report!")


if __name__ == "__main__":
    asyncio.run(main())