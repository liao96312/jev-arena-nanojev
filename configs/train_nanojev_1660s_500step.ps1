$projectRoot = Split-Path -Parent $PSScriptRoot
& "$projectRoot\.venv\Scripts\python.exe" `
  "$projectRoot\third_party\NanoJev\scripts\train_pipeline_decisions.py" `
  --input "$projectRoot\datasets\generated\arena_rollout_memory_1k.jsonl" `
  --output-dir "$projectRoot\runs\arena_rollout_memory_head_500step" `
  --init-checkpoint "$projectRoot\checkpoints\NanoJev\variants\games_gold_seed17" `
  --objective gold_distribution --loss ce --freeze-backbone --backbone-lr 0 `
  --head-steps 0 --steps 500 --batch-questions 8 --microbatch-questions 1 `
  --max-microbatch-tokens 1024 --eval-every 100 --max-length 192 `
  --head-lr 2e-4 --precision fp32
