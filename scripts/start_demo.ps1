$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$apCheckpoint = Join-Path $projectRoot "runs\arena_v2_ap_smoke_head_50step"
$trainedCheckpoint = Join-Path $projectRoot "runs\arena_rollout_memory_head_50step"
$upstreamCheckpoint = Join-Path $projectRoot "checkpoints\NanoJev\variants\games_gold_seed17"
$checkpoint = if (Test-Path (Join-Path $apCheckpoint "config.json")) {
    $apCheckpoint
} elseif (Test-Path (Join-Path $trainedCheckpoint "config.json")) {
    $trainedCheckpoint
} else {
    $upstreamCheckpoint
}
$nanoJevRoot = Join-Path $projectRoot "third_party\NanoJev"
$serviceStarted = $false

try {
    if (-not (Test-Path (Join-Path $checkpoint "config.json"))) {
        throw "未找到 NanoJev checkpoint，请先按 README 下载模型"
    }
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/health" -TimeoutSec 2 | Out-Null
    } catch {
        $logRoot = Join-Path $projectRoot "runs"
        New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
        Start-Process -FilePath $python `
            -ArgumentList @("scripts\serve_decisions.py", "--checkpoint-dir", $checkpoint,
                            "--web-root", "web", "--host", "127.0.0.1", "--port", "8765",
                            "--precision", "fp32") `
            -WorkingDirectory $nanoJevRoot -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot "demo_service.stdout.log") `
            -RedirectStandardError (Join-Path $logRoot "demo_service.stderr.log") | Out-Null
        $serviceStarted = $true
        $ready = $false
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            try {
                Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/health" -TimeoutSec 2 | Out-Null
                $ready = $true
                break
            } catch {
                Start-Sleep -Seconds 1
            }
        }
        if (-not $ready) { throw "NanoJev 模型服务启动超时，请查看 runs\demo_service.stderr.log" }
    }
    & $python (Join-Path $projectRoot "scripts\play_gui.py") --agent nanojev --seed 61005 --decision-ms 280
} finally {
    if ($serviceStarted) {
        Get-CimInstance Win32_Process | Where-Object {
            $_.Name -like "python*.exe" -and $_.CommandLine -like "*serve_decisions.py*"
        } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}
