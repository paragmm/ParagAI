#!/usr/bin/env bash
# ParagAI vLLM Serving Launch Script
# Serves the exported HuggingFace SafeTensors model with OpenAI-compatible API

set -euo pipefail

MODEL_DIR="${1:-checkpoints/export_hf}"
PORT="${2:-8000}"
HOST="${3:-0.0.0.0}"
TP_SIZE="${4:-1}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Export directory '$MODEL_DIR' does not exist."
    echo "Please run: python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir $MODEL_DIR"
    exit 1
fi

echo "=========================================================="
echo " Starting ParagAI vLLM Inference Engine"
echo " Model Directory : $MODEL_DIR"
echo " Host            : $HOST"
echo " Port            : $PORT"
echo " Tensor Parallel : $TP_SIZE"
echo " API Endpoint    : http://$HOST:$PORT/v1/chat/completions"
echo "=========================================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90
