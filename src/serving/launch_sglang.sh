module load CUDA/12.8

source .venv/bin/activate

# model_id="qwen/qwen2.5-0.5b-instruct"
model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct"
# model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

python3 -m sglang.launch_server --model-path $model_id --host 0.0.0.0
