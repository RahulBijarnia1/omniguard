from omniguard.core.models import CampaignConfig, AttackCategory
from omniguard.generators.adversarial import AdversarialPromptGenerator

config = CampaignConfig(name="my-first-run", seed=42, max_vectors_per_category=5)
gen = AdversarialPromptGenerator(config=config)
vectors, stats = gen.generate_sync(AttackCategory.JAILBREAK)

print(f"Generated {len(vectors)} attack vectors!")
print()
for i, v in enumerate(vectors, 1):
    print(f"Attack {i}: {v.prompt[:100]}")
    print(f"Severity: {v.severity}/10")
    print() 