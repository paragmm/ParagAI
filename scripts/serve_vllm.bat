@echo off
REM ParagAI vLLM Serving Launch Script for Windows / WSL
REM Serves the exported HuggingFace SafeTensors model with OpenAI-compatible API

set MODEL_DIR=%~1
if "%MODEL_DIR%"=="" set MODEL_DIR=checkpoints/export_hf

set PORT=%~2
if "%PORT%"=="" set PORT=8000

set HOST=%~3
if "%HOST%"=="" set HOST=0.0.0.0

set TP_SIZE=%~4
if "%TP_SIZE%"=="" set TP_SIZE=1

if not exist "%MODEL_DIR%" (
    echo [ERROR] Export directory '%MODEL_DIR%' does not exist.
    echo Please run: python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --output_dir %MODEL_DIR%
    exit /b 1
)

echo ==========================================================
echo  Starting ParagAI vLLM Inference Engine
echo  Model Directory : %MODEL_DIR%
echo  Host            : %HOST%
echo  Port            : %PORT%
echo  Tensor Parallel : %TP_SIZE%
echo  API Endpoint    : http://%HOST%:%PORT%/v1/chat/completions
echo ==========================================================

python -m vllm.entrypoints.openai.api_server ^
    --model "%MODEL_DIR%" ^
    --host "%HOST%" ^
    --port "%PORT%" ^
    --tensor-parallel-size %TP_SIZE% ^
    --trust-remote-code ^
    --dtype bfloat16 ^
    --gpu-memory-utilization 0.90
