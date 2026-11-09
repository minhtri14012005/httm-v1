param([ValidateSet('cpu','nvidia')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw 'Setup command failed; no driver or global Python changes made.' } }
if (-not (Test-Path -LiteralPath $pythonPath)) { & python -m venv .venv; Check-Exit }
$requestedProfile = Join-Path $PSScriptRoot '.venv\app-profile.txt'
if (Test-Path -LiteralPath $requestedProfile) {
    $existing = (Get-Content -LiteralPath $requestedProfile -Raw).Trim()
    if ($existing -ne $Profile) { throw "This venv uses $existing. Use a separate project checkout/venv for $Profile; no automatic environment replacement." }
}
$ortPackages = & $pythonPath -m pip list --format=freeze
if ($Profile -eq 'cpu' -and ($ortPackages -match '^onnxruntime-gpu==')) { throw 'GPU runtime already installed in this venv. Choose nvidia or create a separate CPU environment manually.' }
if ($Profile -eq 'nvidia' -and ($ortPackages -match '^onnxruntime==')) { throw 'CPU runtime already installed in this venv. Use a separate NVIDIA environment.' }
$torchIndex = if ($Profile -eq 'cpu') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu121' }
& $pythonPath -m pip install torch==2.5.1 torchvision==0.20.1 --index-url $torchIndex; Check-Exit
if ($ortPackages -match '^opencv-python-headless==') { & $pythonPath -m pip uninstall -y opencv-python-headless; Check-Exit }
& $pythonPath -m pip install -c requirements/constraints.txt -r "requirements/$Profile.txt" -r requirements/app.txt; Check-Exit
& $pythonPath -m pip check; Check-Exit
& $pythonPath download_models.py; Check-Exit
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCmd) { & $pythonPath bootstrap_tools.py; Check-Exit }
$portable = Get-ChildItem -LiteralPath '.tools' -Directory -Filter 'node-*-win-x64' -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
$nodeDirectory = if ($nodeCmd) { Split-Path $nodeCmd.Source } else { $portable.FullName }
$env:PATH = $nodeDirectory + ';' + $env:PATH
Push-Location -LiteralPath 'frontend'
try {
    & (Join-Path $nodeDirectory 'npm.cmd') ci --no-audit --no-fund; Check-Exit
    & (Join-Path $nodeDirectory 'npm.cmd') run build; Check-Exit
} finally { Pop-Location }
Set-Content -LiteralPath $requestedProfile -Value $Profile -Encoding ascii
Write-Host 'Ready. Run .\start.ps1'
