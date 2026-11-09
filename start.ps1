param([switch]$Build)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup.ps1 first to create .venv.' }
if ($Build -or -not (Test-Path -LiteralPath 'frontend\dist\index.html')) {
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCmd) { $nodeDirectory = Split-Path $nodeCmd.Source }
    else {
        $portable = Get-ChildItem -LiteralPath '.tools' -Directory -Filter 'node-*-win-x64' -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $portable) { throw 'Node.js missing. Run setup.ps1 or bootstrap_tools.py.' }
        $nodeDirectory = $portable.FullName
    }
    $env:PATH = $nodeDirectory + ';' + $env:PATH
    Push-Location -LiteralPath 'frontend'
    try {
        & (Join-Path $nodeDirectory 'npm.cmd') run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
Write-Host 'Open http://127.0.0.1:8000 - Ctrl+C to stop.'
& $pythonPath -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
