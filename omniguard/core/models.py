"""
omniguard.core.models
=====================
Canonical data contracts for the OmniGuard red-teaming framework.

Every module (generator, adapter, judge, reporter) imports from here.
No module-specific logic lives in this file — pure data shapes only.

OWASP LLM Top 10 references: https://owasp.org/www-project-top-10-for-large-language-model-applications/
MITRE ATLAS references:       https://atlas.mitre.org/
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AttackCategory(str, Enum):
    """
    Top-level taxonomy of adversarial attack strategies.

    Maps directly to OWASP LLM Top 10 and MITRE ATLAS TTPs.
    """

    PROMPT_INJECTION = "prompt_injection"
    """LLM01 · AML.T0051.000 — Direct user-turn injection."""

    JAILBREAK = "jailbreak"
    """LLM01 · AML.T0054 — Role-play, DAN, persona hijack templates."""

    PAYLOAD_SMUGGLING = "payload_smuggling"
    """LLM01 · AML.T0051.001 — Encoding-based evasion (Base64, ROT13, leetspeak)."""

    OBFUSCATION = "obfuscation"
    """LLM01 — Character-level transforms: homoglyphs, zero-width chars, RTL override."""


class HarmCategory(str, Enum):
    """
    Target harm domain for payload-specific campaigns.

    Use ``HarmCategory.AGNOSTIC`` for structural robustness testing
    where the semantic payload is intentionally neutral.
    """

    AGNOSTIC = "agnostic"
    CBRN = "cbrn"           # Chemical, Biological, Radiological, Nuclear
    PII_EXFIL = "pii_exfil" # Personally Identifiable Information extraction
    EXTREMISM = "extremism"
    FRAUD = "fraud"
    CSAM = "csam"           # Always refused by well-aligned models; presence = critical fail


class ObfuscationLevel(int, Enum):
    """
    Graduated obfuscation intensity controlled per campaign run.

    0 — No transformation, raw template output.
    1 — Single transform: homoglyph substitution on targeted chars.
    2 — Stacked: homoglyphs + zero-width character insertion.
    3 — Maximum: stacked + RTL Unicode override (U+202E) injection.
    """

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class OWASPTag(str, Enum):
    """OWASP Top 10 for LLMs — 2025 edition."""

    LLM01_PROMPT_INJECTION = "LLM01:2025"
    LLM02_SENSITIVE_INFO = "LLM02:2025"
    LLM06_EXCESSIVE_AGENCY = "LLM06:2025"
    LLM08_MISINFORMATION = "LLM08:2025"
    LLM09_OVERRELIANCE = "LLM09:2025"


class MITRETag(str, Enum):
    """MITRE ATLAS TTPs relevant to LLM adversarial attacks."""

    T0051_000 = "AML.T0051.000"  # LLM Prompt Injection
    T0051_001 = "AML.T0051.001"  # Stored Prompt Injection
    T0054 = "AML.T0054"          # LLM Jailbreak
    T0048 = "AML.T0048"          # Societal Harm


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class AttackVector(BaseModel):
    """
    A single, fully-described adversarial attack payload.

    This is the unit of work passed from the Adversarial Orchestrator
    to the Model-Agnostic Interface. It is immutable after construction
    (``model_config`` enforces this via ``frozen=True``).

    Serialisation round-trip::

        vec = AttackVector(...)
        json_str = vec.model_dump_json()
        restored = AttackVector.model_validate_json(json_str)
        assert restored == vec
    """

    model_config = {"frozen": True}

    # ── Identity ──────────────────────────────────────────────────────────
    vector_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable UUID for this attack vector across runs.",
    )
    content_hash: str = Field(
        default="",
        description=(
            "SHA-256 of the final prompt text. Populated by __post_init__. "
            "Used for deduplication across corpus merges."
        ),
    )

    # ── Payload ───────────────────────────────────────────────────────────
    prompt: Annotated[str, Field(min_length=1, max_length=32_768)]
    """The adversarial prompt text sent verbatim to the target model."""

    system_prompt_override: str | None = Field(
        default=None,
        description=(
            "If set, replaces the system prompt in the model call. "
            "Used for persona-hijack jailbreaks."
        ),
    )

    # ── Taxonomy ──────────────────────────────────────────────────────────
    category: AttackCategory
    harm_category: HarmCategory = HarmCategory.AGNOSTIC
    owasp_tags: list[OWASPTag] = Field(default_factory=list)
    mitre_tags: list[MITRETag] = Field(default_factory=list)

    # ── Generation metadata ───────────────────────────────────────────────
    template_id: str | None = Field(
        default=None,
        description="Identifier of the Jinja2 template this vector was rendered from.",
    )
    obfuscation_level: ObfuscationLevel = ObfuscationLevel.NONE
    mutation_generation: int = Field(
        default=0,
        ge=0,
        description=(
            "LLM paraphrase pass count. 0 = raw template output, "
            "1+ = nth LLM mutation."
        ),
    )
    seed: int = Field(
        description="RNG seed used during generation — enables exact replay.",
    )

    # ── Evaluation hints (consumed by the Judge module) ───────────────────
    expected_refusal_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Regex patterns that indicate a correct refusal. "
            "The Judge uses these to compute Attack Success Rate."
        ),
    )
    severity: Annotated[int, Field(ge=1, le=10)] = Field(
        default=5,
        description="CVSS-inspired severity score 1–10 set by the generator.",
    )

    # ── Timestamps ────────────────────────────────────────────────────────
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # ── Validators ───────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _populate_content_hash(self) -> "AttackVector":
        """Compute SHA-256 of prompt text for dedup; bypass frozen via object.__setattr__."""
        digest = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        object.__setattr__(self, "content_hash", digest)
        return self

    @field_validator("expected_refusal_patterns")
    @classmethod
    def _validate_regex_patterns(cls, patterns: list[str]) -> list[str]:
        """Eagerly validate that all refusal patterns compile as valid regex."""
        import re
        for pat in patterns:
            try:
                re.compile(pat)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex in expected_refusal_patterns: {pat!r} — {exc}"
                ) from exc
        return patterns

    def is_duplicate_of(self, other: "AttackVector") -> bool:
        """Hash-based exact-duplicate check (content only, ignores metadata)."""
        return self.content_hash == other.content_hash


class CampaignConfig(BaseModel):
    """
    Top-level configuration for a single red-teaming campaign run.

    Passed to the Campaign Controller and propagated to all sub-generators.
    Serialise this to YAML and commit it alongside test results for
    full reproducibility.
    """

    campaign_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Human-readable campaign name.")
    description: str = ""

    # ── Attack surface ─────────────────────────────────────────────────
    enabled_categories: list[AttackCategory] = Field(
        default_factory=lambda: list(AttackCategory),
        description="Attack categories active for this run.",
    )
    harm_filter: HarmCategory = HarmCategory.AGNOSTIC
    obfuscation_level: ObfuscationLevel = ObfuscationLevel.NONE

    # ── Generation controls ────────────────────────────────────────────
    seed: int = Field(
        default=42,
        description=(
            "Master RNG seed. All sub-generators derive their seeds from this "
            "via deterministic offset to guarantee independence without collisions."
        ),
    )
    max_vectors_per_category: int = Field(
        default=50,
        ge=1,
        le=10_000,
        description="Hard cap on generated vectors per AttackCategory.",
    )
    mutation_passes: int = Field(
        default=1,
        ge=0,
        le=5,
        description=(
            "Number of LLM paraphrase passes. 0 disables LLM mutation entirely, "
            "keeping the generator fully self-contained and fast."
        ),
    )

    # ── Corpus sources ─────────────────────────────────────────────────
    use_bundled_corpus: bool = True
    huggingface_datasets: list[str] = Field(
        default_factory=list,
        description="HuggingFace dataset identifiers to merge, e.g. 'walledai/AdvBench'.",
    )
    custom_corpus_dirs: list[str] = Field(
        default_factory=list,
        description="Filesystem paths to directories containing .txt/.json seed files.",
    )

    # ── Timestamps ─────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("enabled_categories")
    @classmethod
    def _at_least_one_category(cls, cats: list[AttackCategory]) -> list[AttackCategory]:
        if not cats:
            raise ValueError("enabled_categories must contain at least one AttackCategory.")
        return cats

    def derive_seed(self, category: AttackCategory) -> int:
        """
        Deterministically derive a per-category seed from the master seed.

        Uses category enum ordinal as an offset so each category's generator
        is statistically independent while remaining fully reproducible.

        Args:
            category: The AttackCategory requiring a unique seed.

        Returns:
            A deterministic integer seed in the range [0, 2**31).
        """
        ordinal = list(AttackCategory).index(category)
        # XOR with a large prime to spread the distribution
        return (self.seed + ordinal * 2_654_435_761) & 0x7FFF_FFFF


class GeneratorStats(BaseModel):
    """
    Runtime statistics emitted by the generator after a campaign run.

    Consumed by the Reporting Engine to compute Attack Surface Coverage.
    """

    category: AttackCategory
    templates_loaded: int = 0
    seeds_loaded: int = 0
    vectors_generated: int = 0
    duplicates_dropped: int = 0
    mutation_passes_applied: int = 0

    # Attack Surface Coverage: vectors_generated / theoretical_max
    # where theoretical_max = templates × obfuscation_variants × mutation_passes
    theoretical_max: int = 0

    @property
    def attack_surface_coverage(self) -> float:
        """
        Fraction of the combinatorial transform space sampled.

        ASC = vectors_generated / theoretical_max.

        A low ASC indicates budget constraints are limiting coverage;
        a high ASC (≥ 0.8) provides strong combinatorial guarantees.
        """
        if self.theoretical_max == 0:
            return 0.0
        return min(self.vectors_generated / self.theoretical_max, 1.0)

    @property
    def dedup_rate(self) -> float:
        """Proportion of generated vectors dropped as duplicates."""
        total = self.vectors_generated + self.duplicates_dropped
        if total == 0:
            return 0.0
        return self.duplicates_dropped / total
