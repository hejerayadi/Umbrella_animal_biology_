<#
    Starts (or stops) the M3 agent API without tying up the terminal.

    Running `uvicorn` in the foreground occupies the window it was launched from,
    so testing it from the same prompt either races the server or kills it. This
    starts it detached, waits until it actually answers, and hands the prompt back.

        .\backend\agents\biodiversity_agent\workers\hotspots\serve.ps1
        .\backend\agents\biodiversity_agent\workers\hotspots\serve.ps1 -Port 8013
        .\backend\agents\biodiversity_agent\workers\hotspots\serve.ps1 -Stop

    The port is freed before binding, so a forgotten instance no longer causes
    "[Errno 10048] only one usage of each socket address is normally permitted".

    Run it from the repository root. It uses the agent's own venv when there is
    one, because the interpreter on PATH may be a Python the dependencies were
    never installed for.
#>

param(
    [int]$Port = 8003,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$hotspots = $PSScriptRoot                                     # workers/hotspots
$agent = Split-Path -Parent (Split-Path -Parent $hotspots)    # biodiversity_agent
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $agent))
$log = Join-Path $hotspots "serve.log"

function Stop-Port([int]$p) {
    $listeners = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        try {
            Stop-Process -Id $listener.OwningProcess -Force -Confirm:$false
            "stopped PID $($listener.OwningProcess) on port $p"
        } catch {
            "could not stop PID $($listener.OwningProcess): $($_.Exception.Message)"
        }
    }
}

Stop-Port $Port
if ($Stop) { "port $Port is free"; return }

# The agent's venv holds scikit-learn, pandas and folium. A bare `python` may be
# a different interpreter entirely - that is what the Python 3.14 wheel failures
# were - so it is only the fallback, and the script says which one it used.
$py = Join-Path $agent ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = (Get-Command python).Source
    "warning: no agent venv found, falling back to $py"
}

$arguments = @(
    "-m", "uvicorn",
    "backend.agents.biodiversity_agent.workers.hotspots.api_m3:app",
    "--port", "$Port"
)

$process = Start-Process -FilePath $py -ArgumentList $arguments `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

# Answering is the only proof it started: binding can still fail after the
# process exists, which is exactly how the 10048 above looked like a success.
$base = "http://localhost:$Port"
$ready = $false
foreach ($attempt in 1..30) {
    Start-Sleep -Milliseconds 700
    try {
        $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3
        $ready = $true
        break
    } catch {
        if ($process.HasExited) { break }
    }
}

if (-not $ready) {
    "failed to start (PID $($process.Id), exited: $($process.HasExited))"
    "last lines of $log.err:"
    if (Test-Path "$log.err") { Get-Content "$log.err" -Tail 12 }
    return
}

""
"M3 agent API is up  ->  $base   (PID $($process.Id), python: $py)"
"  hotspots : $($health.hotspots)"
"  regions  : $($health.regions -join ', ')"
"  log      : $log"
""
"Try it from this same prompt - the server is detached, so it stays up:"
"  Invoke-RestMethod $base/health"
# Single-quoted, with doubled single quotes for the ones in the command itself:
# a double-quoted string here would need backslash-free escaping of every " in
# the JSON body, and would print something that does not paste back in.
'  Invoke-RestMethod BASE/execute -Method Post -ContentType application/json -Body ''{"instruction": "hotspots in Madagascar", "context": {}}''' -replace 'BASE', $base
"  Start-Process $base/map/hotspots_madagascar.html    # opens the map in a browser"
""
"Stop it with:  .\backend\agents\biodiversity_agent\workers\hotspots\serve.ps1 -Stop"
