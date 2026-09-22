$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataset = Join-Path $projectRoot "datasets\generated\arena_v2_beam_10k.jsonl"
$partial = Join-Path $projectRoot "runs\arena_v2_beam_10k.partial.jsonl"
$manifest = Join-Path $projectRoot "datasets\generated\arena_v2_beam_10k.manifest.json"
$checkpoint = Join-Path $projectRoot "runs\arena_v2_beam_10k_head_500step"

if (-not (Test-Path -LiteralPath $dataset)) {
    & $python (Join-Path $projectRoot "scripts\generate_dataset.py") --v2 --records 10000 `
        --targets beam --beam-depth 6 --beam-width 16 --output $partial --resume
    if ($LASTEXITCODE) { throw "Beam Teacher 数据生成失败" }
    Move-Item -LiteralPath $partial -Destination $dataset
}

& $python (Join-Path $projectRoot "scripts\validate_dataset.py") $dataset --manifest $manifest
if ($LASTEXITCODE) { throw "Beam Teacher 数据校验失败" }
& $python (Join-Path $projectRoot "scripts\audit_tokens.py") $dataset --max-length 256 --manifest $manifest
if ($LASTEXITCODE) { throw "Beam Teacher token 审计失败" }
& $python (Join-Path $projectRoot "third_party\NanoJev\scripts\train_pipeline_decisions.py") `
    --input $dataset --output-dir $checkpoint `
    --init-checkpoint (Join-Path $projectRoot "checkpoints\NanoJev\variants\games_gold_seed17") `
    --objective gold_distribution --loss ce --freeze-backbone --backbone-lr 0 `
    --head-steps 0 --steps 500 --batch-questions 8 --microbatch-questions 1 `
    --max-microbatch-tokens 4096 --eval-every 250 --max-length 256 `
    --head-lr 2e-4 --precision fp32
if ($LASTEXITCODE) { throw "Beam Teacher 蒸馏训练失败" }
& $python (Join-Path $projectRoot "scripts\calibrate_predictions.py") $checkpoint `
    --output (Join-Path $checkpoint "calibration.json")
if ($LASTEXITCODE) { throw "Beam Teacher calibration 失败" }
