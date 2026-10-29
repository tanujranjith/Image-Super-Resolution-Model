$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& '.\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed' }
& '.\.venv\Scripts\python.exe' -m ruff check superres app.py tests
if ($LASTEXITCODE -ne 0) { throw 'Lint failed' }
& '.\.venv\Scripts\python.exe' -m pytest -q --junitxml=validation-tests.xml
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& '.\.venv\Scripts\python.exe' scripts/smoke_ui.py
if ($LASTEXITCODE -ne 0) { throw 'Local HTTP startup validation failed' }
