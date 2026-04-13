param(
  [ValidateSet("mock","qwen-loop","qwen-loop-local-critic-world","qwen-loop-local-critic-world-strict")]
  [string]$Preset = "qwen-loop",

  [ValidateSet("smoke","full")]
  [string]$Mode = "smoke",

  [ValidateSet("skip","dry-run","run")]
  [string]$TrainAction = "run",

  [ValidateSet("none","qwen-sft","qwen-dpo","qwen-distill","qwen-critic","qwen-world","qwen-sft-dpo","qwen-sft-dpo-distill","qwen-sft-dpo-distill-critic","qwen-sft-dpo-distill-critic-world")]
  [string]$TrainPreset = "qwen-sft-dpo-distill",

  [string]$StoryDir = "",
  [string]$SceneFile = "",
  [string]$CorpusOutputDir = "outputs/generalist_corpus",
  [double]$ValidationStoryRatio = 0.34,
  [int]$StoryOffset = 0,
  [int]$StoryLimit = 0,
  [int]$SceneLimit = 0,
  [switch]$Resume,
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
  "qwen-loop" { "configs/ollama_native_qwen3_4b_pairwise.yaml" }
  "qwen-loop-local-critic-world" { "configs/ollama_native_qwen3_4b_pairwise_local_critic_world.yaml" }
  "qwen-loop-local-critic-world-strict" { "configs/ollama_native_qwen3_4b_pairwise_local_critic_world_strict.yaml" }
}

if ([string]::IsNullOrWhiteSpace($StoryDir)) {
  if ($Mode -eq "full") {
    $StoryDir = "examples/story_pack_balanced_54"
  }
  else {
    $StoryDir = "examples/story_pack"
  }
}

$trainConfigs = @()
if ($TrainPreset -ne "none") {
  $trainConfigs = switch ($TrainPreset) {
    "qwen-sft" { @("configs/training_qwen3_4b_sft.yaml") }
    "qwen-dpo" { @("configs/training_qwen3_4b_dpo.yaml") }
    "qwen-distill" { @("configs/training_qwen3_4b_distill.yaml") }
    "qwen-critic" { @("configs/training_qwen3_4b_critic_consistency.yaml") }
    "qwen-world" { @("configs/training_qwen3_4b_world_model.yaml") }
    "qwen-sft-dpo" { @("configs/training_qwen3_4b_sft.yaml", "configs/training_qwen3_4b_dpo.yaml") }
    "qwen-sft-dpo-distill" { @("configs/training_qwen3_4b_sft.yaml", "configs/training_qwen3_4b_dpo.yaml", "configs/training_qwen3_4b_distill.yaml") }
    "qwen-sft-dpo-distill-critic" { @("configs/training_qwen3_4b_sft.yaml", "configs/training_qwen3_4b_dpo.yaml", "configs/training_qwen3_4b_distill.yaml", "configs/training_qwen3_4b_critic_consistency.yaml") }
    "qwen-sft-dpo-distill-critic-world" { @("configs/training_qwen3_4b_sft.yaml", "configs/training_qwen3_4b_dpo.yaml", "configs/training_qwen3_4b_distill.yaml", "configs/training_qwen3_4b_critic_consistency.yaml", "configs/training_qwen3_4b_world_model.yaml") }
  }
}

if ($TrainAction -ne "skip" -and $trainConfigs.Count -eq 0) {
  throw "TrainAction is '$TrainAction' but no training preset is selected. Set -TrainPreset or use -TrainAction skip."
}

if ($TrainAction -ne "skip" -and -not (Test-TrainingDeps)) {
  throw "Training dependencies are missing in .venv. Re-run with -InstallTrainingDeps or install .[training] first."
}

$args = @(
  "scripts/run_generalist_loop.py",
  "--app-config", $appConfig,
  "--story-dir", $StoryDir,
  "--mode", $Mode,
  "--train-action", $TrainAction,
  "--corpus-output-dir", $CorpusOutputDir,
  "--validation-story-ratio", "$ValidationStoryRatio"
)

if ($SceneFile) {
  $args += @("--scene-file", $SceneFile)
}
if ($SceneLimit -gt 0) {
  $args += @("--scene-limit", "$SceneLimit")
}
if ($StoryOffset -gt 0) {
  $args += @("--story-offset", "$StoryOffset")
}
if ($StoryLimit -gt 0) {
  $args += @("--story-limit", "$StoryLimit")
}
if ($Resume) {
  $args += "--resume"
}
if ($RunTests) {
  $args += "--run-tests"
}
foreach ($trainConfig in $trainConfigs) {
  $args += @("--train-config", $trainConfig)
}

& $python @args
exit $LASTEXITCODE
