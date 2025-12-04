#! /bin/bash

module load CUDA/12.8

source .venv/bin/activate

model_id="Qwen/Qwen3-0.6B"

context_length=40960

python3 -m sglang.launch_server \
    --model-path "$model_id" \
    --host 0.0.0.0 \
    --context-length "$context_length" \
    --port 30000 \
    --mem-fraction-static "0.95"





