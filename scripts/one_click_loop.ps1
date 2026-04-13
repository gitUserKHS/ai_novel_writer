param(
  [ValidateSet("mock","gemma-e2b","gemma-e4b","qwen-native","qwen-loop","qwen-local-critic","qwen-local-critic-world","qwen-local-critic-world-strict","qwen-loop-local-critic-world","qwen-loop-local-critic-world-strict")]
  [string]$Preset = "qwen-native",

  [ValidateSet("smoke","full")]
  [string]$Mode = "smoke",

  [ValidateSet("skip","dry-run","run")]
  [string]$TrainAction = "run",

  [ValidateSet("none","qwen-sft-smoke","qwen-sft","qwen-dpo","qwen-distill","qwen-sft-dpo","qwen-sft-dpo-distill","gemma-sft")]
  [string]$TrainPreset = "qwen-sft-smoke",

  [string]$StoryFile = "examples/story.yaml",
  [string]$SceneFile = "examples/scene.yaml",
  [string]$StoryId,
  [int]$SceneLimit = 0,
  [switch]$RunTests,
  [switch]$InstallTrainingDeps
)

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Virtual environment not found: $python"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Test-TrainingDeps {
  & $python -c "import torch, transformers, datasets, peft, trl, bitsandbytes, accelerate"
  return $LASTEXITCODE -eq 0
}

if ($InstallTrainingDeps) {
  & powershell -ExecutionPolicy Bypass -File ".\scripts\install_training_deps.ps1"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$appConfig = switch ($Preset) {
  "mock" { "configs/quickstart_mock.yaml" }
  "gemma-e2b" { "configs/ollama_gemma4_e2b.yaml" }
  "gemma-e4b" { "configs/ollama_gemma4_e4b.yaml" }
  "qwen-native" { "configs/ollama_native_qwen3_4b.yaml" }
  "qwen-loop" { "configs/ollama_native_qwen3_4b_pairwise.yaml" }
  "qwen-local-critic" { "configs/ollama_native_qwen3_4b_local_critic.yaml" }
  "qwen-local-critic-world" { "configs/ollama_native_qwen3_4b_local_critic_world.yaml" }
  "qwen-local-critic-world-strict" { "configs/ollama_native_qwen3_4b_local_critic_world_strict.yaml" }
  "qwen-loop-local-critic-world" { "configs/ollama_native_qwen3_4b_pairwise_local_critic_world.yaml" }
  "qwen-loop-local-critic-world-strict" { "configs/ollama_native_qwen3_4b_pairwise_local_critic_world_strict.yaml" }
}

if ($Preset -in @("qwen-loop", "qwen-loop-local-critic-world", "qwen-loop-local-critic-world-strict") -and $SceneFile -eq "examples/scene.yaml") {
  $SceneFile = "examples/scene_smoke.yaml"
}

$trainConfigs = @()
if ($TrainPreset -ne "none") {
  $trainConfigs = switch ($TrainPreset) {
    "qwen-sft-smoke" { @("configs/training_qwen3_4b_sft_smoke.yaml") }
    "qwen-sft" { @("configs/training_qwen3_4b_sft.yaml") }
    "qwen-dpo" { @("configs/training_qwen3_4b_dpo.yaml") }
    "qwen-distill" { @("configs/training_qwen3_4b_distill.yaml") }
    "qwen-sft-dpo" { @("configs/training_qwen3_4b_sft.yaml", "configs/training_qwen3_4b_dpo.yaml") }
    "qwen-sft-dpo-distill" { @("configs/training_qwen3_4b_sft.yaml", "configs/training_qwen3_4b_dpo.yaml", "configs/training_qwen3_4b_distill.yaml") }
    "gemma-sft" { @("configs/training_gemma3_4b_it_sft.yaml") }
  }
}

if ($TrainAction -ne "skip" -and $trainConfigs.Count -eq 0) {
  throw "TrainAction is '$TrainAction' but no training preset is selected. Set -TrainPreset or use -TrainAction skip."
}

if ($TrainAction -ne "skip" -and -not (Test-TrainingDeps)) {
  throw "Training dependencies are missing in .venv. Re-run with -InstallTrainingDeps or install .[training] first."
}

$args = @(
  "scripts/run_pipeline.py",
  "--app-config", $appConfig,
  "--story-file", $StoryFile,
  "--mode", $Mode,
  "--train-action", $TrainAction
)

if ($SceneFile) {
  $args += @("--scene-file", $SceneFile)
}
if ($StoryId) {
  $args += @("--story-id", $StoryId)
}
if ($SceneLimit -gt 0) {
  $args += @("--scene-limit", "$SceneLimit")
}
if ($RunTests) {
  $args += "--run-tests"
}
foreach ($trainConfig in $trainConfigs) {
  $args += @("--train-config", $trainConfig)
}

& $python @args
exit $LASTEXITCODE
