<#
    Umbrella - start the whole stack with one command.

        .\run-all.ps1

    Opens one window per service so each keeps its own log and can be stopped
    with Ctrl+C independently. Stale listeners on the ports below are cleared
    first, which is what otherwise produces [WinError 10048].

    Options:
        .\run-all.ps1 -SkipFrontend    # no Vite dev server
        .\run-all.ps1 -SkipDashboard   # no Streamlit dashboard
        .\run-all.ps1 -Stop            # stop everything, start nothing
#>

param(
    [switch]$SkipFrontend,
    [switch]$SkipDashboard,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$root      = $PSScriptRoot
$rootPy    = Join-Path $root "venv\Scripts\python.exe"
$agentPy   = Join-Path $root "backend\agents\biodiversity_agent\.venv\Scripts\python.exe"
$frontend  = Join-Path $root "frontend"

# ---------- free the ports ----------

# Only ports this project owns. The frontend's own port (8080, per the Lovable
# vite preset) is deliberately NOT in this list: other apps on this machine use
# 8080 and 3000, and Vite picks the next free port by itself anyway.
$ports = @(8000..8009) + 8501
$killed = 0
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        $killed++
    }
}
if ($killed -gt 0) {
    Write-Host "freed $killed stale listener(s)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

if ($Stop) {
    Write-Host "stopped. nothing started." -ForegroundColor Yellow
    exit 0
}

# ---------- preflight ----------

$problems = @()
if (-not (Test-Path $rootPy))  { $problems += "root venv missing - run:  py -3.11 -m venv venv" }
if (-not (Test-Path $agentPy)) { $problems += "biodiversity venv missing - run:  py -3.11 -m backend.run_agents --setup" }
if (-not $SkipFrontend -and -not (Test-Path (Join-Path $frontend "node_modules"))) {
    $problems += "frontend/node_modules missing - run:  cd frontend ; npm install"
}
if ($problems.Count -gt 0) {
    Write-Host "cannot start:" -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}

# ---------- launch, one window each ----------

function Start-Service-Window {
    param([string]$Title, [string]$Command, [string]$WorkDir = $root)

    $inner = "`$host.UI.RawUI.WindowTitle = '$Title'; $Command"
    Start-Process powershell -WorkingDirectory $WorkDir -ArgumentList @(
        "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $inner
    ) | Out-Null
    Write-Host "  started  $Title" -ForegroundColor Green
}

Write-Host ""
Write-Host "starting Umbrella..." -ForegroundColor Cyan

# The nine agent services. Must come first - the orchestrator reaches them over
# HTTP, so without them every worker returns FAILED.
Start-Service-Window "Umbrella agents 8001-8009" "py -3.11 -m backend.run_agents"

# Give the agents a head start so the API's first request finds them listening.
Start-Sleep -Seconds 8

Start-Service-Window "Umbrella API 8000" "& '$rootPy' -m uvicorn backend.api:app --reload --port 8000"

if (-not $SkipDashboard) {
    Start-Service-Window "Biodiversity dashboard 8501" `
        "& '$agentPy' -m streamlit run backend\agents\biodiversity_agent\dashboard.py --server.address 127.0.0.1"
}

if (-not $SkipFrontend) {
    Start-Service-Window "Umbrella frontend 5173" "npm run dev" $frontend
}

# ---------- summary ----------

Write-Host ""
Write-Host "URLs (give them ~20s to come up):" -ForegroundColor Cyan
Write-Host "  frontend    http://localhost:8080   <- if taken, Vite prints the real"
Write-Host "                                         port in its own window (8081, ...)"
Write-Host "  API docs    http://localhost:8000/docs"
Write-Host "  dashboard   http://127.0.0.1:8501"
Write-Host "  your agent  http://localhost:8003/docs"
Write-Host ""
Write-Host "Ctrl+C in a window stops that service. Or run:  .\run-all.ps1 -Stop" -ForegroundColor DarkGray

# The chat loop needs a real Azure key; warn if the placeholder is still there.
$envFile = Join-Path $root "backend\orchestrator\.env"
if ((Test-Path $envFile) -and (Select-String -Path $envFile -Pattern "<your-resource>|REPLACE_WITH" -Quiet)) {
    Write-Host ""
    Write-Host "NOTE: backend\orchestrator\.env still has placeholder Azure values." -ForegroundColor Yellow
    Write-Host "      Agents, dashboard and tests work; chat via the frontend will fail" -ForegroundColor Yellow
    Write-Host "      at the planner until you put the real endpoint and key there." -ForegroundColor Yellow
}
