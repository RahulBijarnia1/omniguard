import asyncio
from omniguard.core.models import CampaignConfig, AttackCategory
from omniguard.generators.adversarial import AdversarialPromptGenerator
from omniguard.adapters.ollama_adapter import OllamaAdapter
from omniguard.adapters.runner import CampaignRunner


async def main():
    print("Generating attack vectors...")
    config = CampaignConfig(
        name="omniguard-run",
        seed=42,
        max_vectors_per_category=3,
        mutation_passes=0,
        enabled_categories=[
            AttackCategory.JAILBREAK,
            AttackCategory.PROMPT_INJECTION,
        ],
    )
    generator = AdversarialPromptGenerator(config=config)
    all_vectors = []

    for category in config.enabled_categories:
        vectors, stats = await generator.generate_async(category)
        all_vectors.extend(vectors)
        print(f"  {category.value}: {len(vectors)} vectors")

    print(f"Total: {len(all_vectors)} vectors ready!")

    adapters = [OllamaAdapter(model="llama3.2")]
    print("Ollama ready!\n")

    runner = CampaignRunner(
        adapters=adapters,
        output_dir="results",
        show_live=True,
    )
    await runner.run(all_vectors)


if __name__ == "__main__":
    asyncio.run(main())