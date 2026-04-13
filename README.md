# CoNarrative

CoNarrative is a local-first prototype for long-form fiction generation built around a `scene` runtime rather than a full-book monolith.

The core loop is:

1. build or adapt an outline card
2. plan one scene
3. generate multiple scene candidates
4. score them with consistency and creativity critics
5. revise the winner if needed
6. extract memory updates into structured stores
7. export manuscripts, evaluations, and distillation / fine-tuning datasets

## Purpose

This repository is designed to test a story-generation architecture with explicit memory separation:

- Story Bible: static facts and rules
- State Tracker: dynamic per-scene state
- Narrative KG: event and foreshadowing edges
- Revision loop: patch failing spans instead of regenerating everything
- Training pools: accepted, pairwise, hard-negative, and prompt-only data kept separate

## Working Status

Verified on `Windows` with `Python 3.12`.

Current status:

- `mock` backend: works end-to-end
- FastAPI app: works
- SQLite storage: works
- dataset export for distillation / SFT / DPO: works
- local `ollama` backend: config added for `gemma4:e4b`, `gemma4:e2b`, and `qwen3:4b`

Local Ollama presets use `candidate_count: 1` and shorter scene-length targets by default to keep scene latency realistic on a single consumer GPU or CPU-bound local runtime. Increase them if you want richer pairwise preference data and longer prose.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --force-reinstall pip==25.3
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[dev]"
```

If you also want training dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[dev,training]"
```

## Quickstart: Mock Backend

This path is the fastest way to confirm the project is healthy without any model server.

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/quickstart_mock.yaml init
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/quickstart_mock.yaml health
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/quickstart_mock.yaml create-story --input-file examples/story.yaml
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/quickstart_mock.yaml auto-novel --story-id moon-theater
```

Generated outputs land under:

- `configs/workspace_quickstart_mock/exports`

## Quickstart: Ollama With `gemma4:e4b`

This repository uses an OpenAI-compatible backend interface, so Ollama works through `http://127.0.0.1:11434/v1`.

Confirmed locally in this environment:

- `ollama version`: `0.20.5`
- installed model: `gemma4:e4b`
- OpenAI-compatible endpoint: `http://127.0.0.1:11434/v1/models`

Health check:

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e4b.yaml health
```

Run a story:

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e4b.yaml init
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e4b.yaml create-story --input-file examples/story.yaml
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e4b.yaml auto-novel --story-id moon-theater
```

Practical smoke test first:

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e4b.yaml create-story --input-file examples/story.yaml
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e4b.yaml run-scene --story-id moon-theater --input-file examples/scene.yaml --print-text
```

Outputs land under:

- `configs/workspace_ollama_gemma4_e4b/exports`

## Faster Ollama Option: `gemma4:e2b`

If `gemma4:e4b` is too slow for the full multi-call scene loop, use:

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e2b.yaml health
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e2b.yaml create-story --input-file examples/story.yaml
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_gemma4_e2b.yaml run-scene --story-id moon-theater --input-file examples/scene.yaml --print-text
```

Config file:

- `configs/ollama_gemma4_e2b.yaml`

## Alternative: Ollama With `qwen3:4b`

If you prefer a small Qwen path:

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_qwen3_4b.yaml health
```

Config file:

- `configs/ollama_qwen3_4b.yaml`

This preset explicitly disables Qwen 3 reasoning mode for the OpenAI-compatible path and avoids `response_format`, because Qwen 3 can otherwise spend the token budget on reasoning traces and starve the final JSON payload in a structured pipeline like this one.

In this environment on Ollama `0.20.5`, Qwen 3 is still less reliable than Gemma 4 for this repository's multi-stage structured JSON loop.

## Preferred Qwen Path: Ollama Native

For Qwen 3 specifically, the more reliable path is the Ollama native API with JSON schema output:

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_native_qwen3_4b.yaml health
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_native_qwen3_4b.yaml create-story --input-file examples/story.yaml
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/ollama_native_qwen3_4b.yaml run-scene --story-id moon-theater --input-file examples/scene.yaml --print-text
```

Config file:

- `configs/ollama_native_qwen3_4b.yaml`

For a local mini self-improvement loop with pairwise data, use the pairwise preset:

- `configs/ollama_native_qwen3_4b_pairwise.yaml`
- `examples/scene_smoke.yaml`

For role-specialized local adapters on top of Ollama generation, use:

- `configs/ollama_native_qwen3_4b_local_critic.yaml`
- `configs/ollama_native_qwen3_4b_local_critic_world.yaml`
- `configs/ollama_native_qwen3_4b_local_critic_world_strict.yaml`
- `configs/ollama_native_qwen3_4b_pairwise_local_critic_world.yaml`
- `configs/ollama_native_qwen3_4b_pairwise_local_critic_world_strict.yaml`

These configs keep planner / writer / extractor on Ollama, route the consistency critic to a locally trained adapter, and optionally attach a locally trained world-model adapter as a post-selection verifier. The learned world-model path is best-effort: if the adapter emits malformed output, CoNarrative records the error and falls back to the built-in symbolic world-model instead of failing the scene run.

In the current validated setup, the world-model adapter uses a compact prompt format and unloads active Ollama models before each local role call so that a single small GPU can switch between remote generation and local adapter verification reliably.

These local role presets also use a stronger release gate:

- minimum consistency threshold
- minimum world-plausibility threshold
- zero tolerated `high` issues
- configurable limit on remaining `medium` issues
- one rescue round with fresh candidates before forced acceptance

The `*_strict.yaml` variants hard-fail the scene after rescue exhaustion instead of forcing acceptance. Use them when you want quality gating to stop the run rather than log-and-continue.

## API Server

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/quickstart_mock.yaml serve --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/`

The web UI now ships with:

- guided quick-start cards
- sample story templates for `달빛 극장`, `유리 항만`, `붉은 기록 보관소`
- runtime preset buttons for `mock`, `qwen native`, and local critic/world-model flows
- browser-side launch panels for `one-click loop`, `generalist loop`, and direct `train_qlora.py` jobs
- a scene inspector that shows accepted text, candidate scores, and raw JSON side by side
- a clearer export / evaluate / training-bundle workflow

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Data Export For Distillation / Fine-Tuning

After scenes are generated, CoNarrative exports:

- writer SFT data
- pairwise DPO-style preference data
- stepwise teacher traces for distillation
- hard-negative critic data
- balanced consistency-critic SFT data
- world-model transition data

Example:

```powershell
.\.venv\Scripts\python.exe -m conarrative.cli --config configs/quickstart_mock.yaml export-datasets --story-id moon-theater
```

Generated training files:

- [moon-theater_writer_sft.jsonl](/E:/Desktop/프로젝트/ai_revolution/configs/workspace_quickstart_mock/exports/moon-theater/training/moon-theater_writer_sft.jsonl)
- [moon-theater_writer_dpo.jsonl](/E:/Desktop/프로젝트/ai_revolution/configs/workspace_quickstart_mock/exports/moon-theater/training/moon-theater_writer_dpo.jsonl)
- [moon-theater_distill_stepwise.jsonl](/E:/Desktop/프로젝트/ai_revolution/configs/workspace_quickstart_mock/exports/moon-theater/training/moon-theater_distill_stepwise.jsonl)
- [moon-theater_critic_consistency_sft.jsonl](/E:/Desktop/프로젝트/ai_revolution/configs/workspace_quickstart_mock/exports/moon-theater/training/moon-theater_critic_consistency_sft.jsonl)
- [moon-theater_world_model_transitions.jsonl](/E:/Desktop/프로젝트/ai_revolution/configs/workspace_quickstart_mock/exports/moon-theater/training/moon-theater_world_model_transitions.jsonl)

## QLoRA Presets

Included presets:

- `configs/training_qwen3_4b_sft_smoke.yaml`
- `configs/training_qwen3_4b_sft.yaml`
- `configs/training_qwen3_4b_dpo.yaml`
- `configs/training_qwen3_4b_distill.yaml`
- `configs/training_qwen3_4b_critic_consistency.yaml`
- `configs/training_qwen3_4b_world_model.yaml`
- `configs/training_gemma3_4b_it_sft.yaml`

The trainer now supports:

- explicit `--eval-file`
- story-aware validation split with `--validation-split-ratio`
- fixed `seed`
- `early stopping` and best-checkpoint selection
- chained adapter training where DPO / distill can continue from the previous LoRA output
- optional `pool_key` selection so SFT can target writer or critic corpora
- optional `chain_from_previous: false` for role-specific stages such as critic training

Validate the config without installing the heavy training stack:

```powershell
.\.venv\Scripts\python.exe scripts/train_qlora.py --config configs/training_qwen3_4b_sft.yaml --dry-run
.\.venv\Scripts\python.exe scripts/train_qlora.py --config configs/training_qwen3_4b_sft.yaml --print-config
```

Run Qwen 3 SFT:

```powershell
.\.venv\Scripts\python.exe scripts/train_qlora.py --config configs/training_qwen3_4b_sft.yaml
```

Run Qwen 3 DPO:

```powershell
.\.venv\Scripts\python.exe scripts/train_qlora.py --config configs/training_qwen3_4b_dpo.yaml
```

Run Qwen 3 stepwise distillation:

```powershell
.\.venv\Scripts\python.exe scripts/train_qlora.py --config configs/training_qwen3_4b_distill.yaml
```

Run Qwen 3 consistency critic SFT:

```powershell
.\.venv\Scripts\python.exe scripts/train_qlora.py --config configs/training_qwen3_4b_critic_consistency.yaml
```

Run Gemma 3 4B IT SFT:

```powershell
.\.venv\Scripts\python.exe scripts/train_qlora.py --config configs/training_gemma3_4b_it_sft.yaml
```

## One-Click Loop

Windows PowerShell wrapper:

- `scripts/one_click_loop.ps1`
- `scripts/one_click_loop.cmd`

With the current defaults, running the wrapper with no arguments executes a practical local smoke loop:

- `qwen-native` generation
- evaluation and training-bundle export
- `qwen-sft-smoke` training run

For local Ollama presets, the pipeline now unloads the active Ollama model before training starts so a single 8 GB GPU can switch cleanly from inference to QLoRA.

Examples:

Smoke test with Qwen native generation and Qwen SFT dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_loop.ps1 -Preset qwen-native -Mode smoke -TrainPreset qwen-sft-smoke -TrainAction dry-run -RunTests
```

Full novel with Gemma 4 e2b and no training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_loop.ps1 -Preset gemma-e2b -Mode full -TrainPreset none
```

Full mock generation with chained `SFT -> DPO -> distill`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_loop.ps1 -Preset mock -Mode full -TrainPreset qwen-sft-dpo-distill -TrainAction run
```

Local Ollama smoke loop with pairwise generation and chained `SFT -> DPO -> distill`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_loop.ps1 -Preset qwen-loop -Mode smoke -TrainPreset qwen-sft-dpo-distill -TrainAction run
```

Local Ollama smoke loop with a trained consistency critic and learned world-model verifier:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_loop.ps1 -Preset qwen-local-critic-world -Mode smoke -SceneFile examples/scene_smoke.yaml -TrainPreset none -TrainAction skip
```

Install training dependencies and run the loop:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_loop.ps1 -Preset qwen-native -Mode smoke -TrainPreset qwen-sft-smoke -TrainAction run -InstallTrainingDeps
```

CMD wrapper with the same defaults. With no arguments it also runs `pytest -q` first:

```powershell
.\scripts\one_click_loop.cmd
```

Python automation entry:

- `scripts/run_pipeline.py`

You can chain multiple training stages by passing `--train-config` more than once. Each later stage automatically uses the previous stage's output directory as `model_name_or_path`. If a stage config sets `chain_from_previous: false`, it starts from its own `model_name_or_path` instead. The critic preset uses this to train a separate critic adapter from the base model.

## Generalist Loop

For broader fiction behavior, avoid training on a single story only. This repository now includes:

- `examples/story_pack/`
- `examples/story_pack_balanced_54/`
- `scripts/build_mixed_corpus.py`
- `scripts/generate_story_pack.py`
- `scripts/run_generalist_loop.py`
- `scripts/one_click_generalist.ps1`
- `scripts/one_click_generalist.cmd`

The generalist loop:

1. generates multiple stories
2. exports per-story SFT / DPO / distill bundles
3. exports balanced consistency-critic SFT bundles
4. merges them into a mixed corpus
5. creates a story-level holdout split for validation
6. runs chained `SFT -> DPO -> distill` and optional critic tuning with eval loss tracking

`scripts/run_generalist_loop.py` also supports:

- `--story-offset`
- `--story-limit`
- `--resume`

That lets you shard a large pack across multiple runs and then continue without regenerating completed stories.

`examples/story_pack/` stays small for fast smoke runs.

`examples/story_pack_balanced_54/` is the larger generalization pack:

- `54` stories
- `6` genre families with `9` stories each
- `3` tone families with `18` stories each
- `3` conflict engines with `18` stories each
- `5` scenes per story to keep full-pack runs computationally realistic

Regenerate the balanced pack and manifest:

```powershell
.\.venv\Scripts\python.exe scripts/generate_story_pack.py
```

Default one-click command:

```powershell
.\scripts\one_click_generalist.cmd
```

Mock full-data validation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_generalist.ps1 -Preset mock -Mode full -TrainPreset qwen-sft-dpo-distill -TrainAction run
```

Balanced 54-story smoke loop with writer and critic tuning:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_generalist.ps1 -Preset mock -Mode smoke -StoryDir examples/story_pack_balanced_54 -TrainPreset qwen-sft-dpo-distill-critic -TrainAction run
```

Local Qwen pairwise smoke loop across the story pack with local critic and learned world-model verifier:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_generalist.ps1 -Preset qwen-loop-local-critic-world -Mode smoke -TrainPreset none -TrainAction skip
```

Local Qwen full-data loop across the 3-story pack with chained training stages prepared:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_generalist.ps1 -Preset qwen-loop-local-critic-world -Mode full -StoryDir examples/story_pack -TrainPreset qwen-sft-dpo-distill-critic-world -TrainAction dry-run
```

Strict local gate on a single-story smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_loop.ps1 -Preset qwen-local-critic-world-strict -Mode smoke -SceneFile examples/scene_smoke.yaml -TrainPreset none -TrainAction skip
```

Balanced 54-pack sharded local run, first batch:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_generalist.ps1 -Preset qwen-loop-local-critic-world -Mode full -StoryDir examples/story_pack_balanced_54 -StoryOffset 0 -StoryLimit 6 -Resume -TrainPreset none -TrainAction skip
```

Balanced 54-pack sharded local run, next batch:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_generalist.ps1 -Preset qwen-loop-local-critic-world -Mode full -StoryDir examples/story_pack_balanced_54 -StoryOffset 6 -StoryLimit 6 -Resume -TrainPreset none -TrainAction skip
```

Local Qwen pairwise smoke loop across the story pack:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\one_click_generalist.ps1 -Preset qwen-loop -Mode smoke -TrainPreset qwen-sft-dpo-distill -TrainAction run
```

Corpus builder only:

```powershell
.\.venv\Scripts\python.exe scripts/build_mixed_corpus.py --manifest path\to\story_a_training_manifest.json --manifest path\to\story_b_training_manifest.json --output-dir outputs\generalist_corpus
```

## Model Guidance

For a local RTX 4060-class workflow:

- inference: `gemma4:e4b` is viable if you want the strongest small local Ollama option already present here
- lighter inference: `gemma4:e2b`
- writer distillation / LoRA target: `Qwen/Qwen3-4B` is a pragmatic small-model target
- Gemma path for HF fine-tuning: use a small instruction-tuned Gemma checkpoint that matches your hardware budget

The training script is:

- `scripts/train_qlora.py`

## Notes

- `configs/workspace` contains checked-in sample artifacts from an earlier run. Use the new quickstart configs above for clean runs.
- The built-in world model is a lightweight abstract-state scorer, not a generative JEPA.
- The repository is optimized around scene-level iteration and dataset generation, not direct full-novel one-shot generation.

