$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Virtual environment not found: $python"
}

& $python -m pip install --force-reinstall pip==25.3
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m pip install --no-build-isolation -e ".[training,dev]"
exit $LASTEXITCODE
