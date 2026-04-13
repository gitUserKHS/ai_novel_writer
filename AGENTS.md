# CoNarrative Codex instructions

## Mission
Build and maintain a working CoNarrative prototype for long-form story generation with explicit memory, planning, critics, revision, and a gated self-improving loop.

## Operating rules
- Keep the repository runnable at every meaningful checkpoint.
- Prefer the smallest change that preserves the architecture.
- Treat JSON / Pydantic schemas as API contracts.
- Separate static memory, dynamic state, and event graph storage.
- Validate with focused tests after code changes.
- Keep reproducibility notes, commands, and example configs aligned with reality.

## Architectural invariants
- The runtime unit is a scene, not a full novel.
- Story Bible stores static facts.
- State Tracker stores per-scene dynamic state.
- Narrative KG stores events and foreshadowing edges.
- Revision patches failing spans; repeated failure can trigger replanning.
- Self-improvement pools remain separate.
