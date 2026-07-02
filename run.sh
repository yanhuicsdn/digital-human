#!/usr/bin/env bash
# ============================================================
# 数字人视频生成服务 — 启动脚本
# SoulX-FlashHead + LongCat-AudioDiT
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo " 数字人视频生成服务 (Digital Human Video Gen)"
echo " 整合: SoulX-FlashHead + LongCat-AudioDiT"
echo " 视频后端: ${VIDEO_BACKEND:-flashhead}"
echo "================================================"

# --- Environment setup ---
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# --- Configuration (override via env vars) ---
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-5000}"

# Model paths (update these to match your setup)
export FLASHHEAD_CKPT_DIR="${FLASHHEAD_CKPT_DIR:-${SCRIPT_DIR}/models/SoulX-FlashHead-1_3B}"
export FLASHHEAD_WAV2VEC_DIR="${FLASHHEAD_WAV2VEC_DIR:-${SCRIPT_DIR}/models/wav2vec2-base-960h}"
export FLASHHEAD_MODEL_TYPE="${FLASHHEAD_MODEL_TYPE:-lite}"

export AUDIODIT_MODEL_DIR="${AUDIODIT_MODEL_DIR:-${SCRIPT_DIR}/models/LongCat-AudioDiT-1B}"

# LLM config (optional — for text generation endpoints)
export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export LLM_DEFAULT_MODEL="${LLM_DEFAULT_MODEL:-qwen-turbo}"

echo ""
echo " Configuration:"
echo "   Host:               ${HOST}:${PORT}"
echo "   Video Backend:      ${VIDEO_BACKEND:-flashhead}"
if [ "${VIDEO_BACKEND:-flashhead}" = "longcat" ]; then
    echo "   LongCat model:      ${LONGCAVA_CKPT_DIR:-${SCRIPT_DIR}/models/LongCat-Video-Avatar-1.5}"
else
    echo "   FlashHead model:    ${FLASHHEAD_MODEL_TYPE:-lite} @ ${FLASHHEAD_CKPT_DIR}"
fi
echo "   AudioDiT model:     ${AUDIODIT_MODEL_DIR}"
echo "   LLM backend:        ${LLM_BASE_URL} / ${LLM_DEFAULT_MODEL}"
echo ""

# Check Python env
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please activate your conda/virtual environment."
    exit 1
fi

# Check flash_head dependency
if [ ! -d "flash_head" ] || [ ! -f "flash_head/inference.py" ]; then
    echo "⚠️  WARNING: flash_head/ not found. Run setup.sh first,"
    echo "   or manually clone SoulX-FlashHead and copy its flash_head/ directory:"
    echo "     git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git /tmp/fh"
    echo "     cp -r /tmp/fh/flash_head ./"
    echo ""
fi

echo " Starting server..."
echo ""

exec python3 -m app.main
