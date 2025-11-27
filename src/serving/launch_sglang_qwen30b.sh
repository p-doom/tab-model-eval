module load CUDA/12.8

source .venv/bin/activate

# model_id="qwen/qwen2.5-0.5b-instruct"
model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct"
# model_id="/fast/project/HFMI_SynergyUnit/tab_model/data/outputs/hf/qwen3_600m_full_finetune_crowd_code_dataset_seqlen_16k_token_batch_size_128k/2025-11-24_18-17-40"
# model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
# model_id="Qwen/Qwen3-0.6B"

# context_length=128000
context_length=40960

python3 -m sglang.launch_server \
    --model-path $model_id \
    --host 0.0.0.0 \
    --context-length $context_length \
    --port 30000 \
    --mem-fraction-static 0.95





