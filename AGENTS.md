# CoNarrative Studio instructions

## Mission
Build and maintain a runnable local-first CoNarrative Studio for long-form story generation with explicit memory, planning, critics, revision, and export/evaluation tooling.

## Operating rules
- Keep the repository runnable at every meaningful checkpoint.
- Prefer the smallest change that preserves the scene-based architecture.
- Treat Pydantic models and JSON payloads as API contracts.
- Keep static memory, dynamic state, and narrative graph storage separate.
- Keep prompts and schemas versioned in the repo.
- Before adding dependencies, prefer lightweight defaults.
- Run focused validation after code changes and report command output.
- Preserve reproducibility: document commands, config, sample inputs, and expected outputs.

## Architectural invariants
- The runtime unit is a scene, not a full novel.
- Story Bible stores static facts.
- State Tracker stores dynamic scene-to-scene state.
- Narrative KG stores events, causality, intent, and foreshadowing edges.
- Revision patches failing spans; repeated failure should be visible in logs.
- Accepted, pairwise, prompt-only, and hard-negative dataset pools remain separated.
