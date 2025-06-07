"""
Stage 2: Natural and Controlled AI Prompt Generation

Transforms Stage 1 customer archetypes and queries into executable
prompt packages for manual AI testing across platforms.

Key Features:
- Natural AI prompts (minimal instructions, mirrors typical customer interactions)
- Controlled AI prompts (search mandates, source attribution, hallucination prevention)
- Complete execution packages with traceability
- Human-readable execution guides
- JSON schema compliance for response validation
"""
from .prompt_executor import Stage2PromptExecutor, execute_stage2

__all__ = [
    "Stage2PromptExecutor",
    "execute_stage2"
]