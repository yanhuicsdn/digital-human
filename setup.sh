#!/usr/bin/env bash
# ============================================================
# 数字人视频生成服务 — 一键部署脚本
# 整合: SoulX-FlashHead + LongCat-AudioDiT
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo " 数字人视频生成服务 - 环境准备"
echo " 整合: SoulX-FlashHead + LongCat-AudioDiT"
echo "================================================"

# --- Check Python ---
PYTHON="python3"
if ! command -v $PYTHON &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.10+ and activate your environment."
    exit 1
fi

echo ""
echo "📦 Step 1/5: Installing Python dependencies..."
$PYTHON -m pip install -r requirements.txt -q
echo "   ✅ Dependencies installed"

# --- Install ModelScope ---
echo ""
echo "📦 Step 2/5: Installing ModelScope (for model download)..."
$PYTHON -m pip install modelscope -q
echo "   ✅ ModelScope installed"

# --- Clone SoulX-FlashHead for flash_head package ---
echo ""
echo "📦 Step 3/5: Setting up SoulX-FlashHead dependency..."
if [ -d "flash_head" ] && [ -f "flash_head/inference.py" ]; then
    echo "   ✅ flash_head/ already exists, skipping"
else
    TMPDIR=$(mktemp -d)
    echo "   Cloning from GitHub..."
    git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git "$TMPDIR"
    cp -r "$TMPDIR/flash_head" ./
    cp -r "$TMPDIR/flash_head/configs" ./flash_head/configs 2>/dev/null || true
    rm -rf "$TMPDIR"
    echo "   ✅ flash_head/ installed"
fi

# --- Create data directories ---
echo ""
echo "📁 Step 4/5: Ensuring data directories..."
mkdir -p data/avatars data/generated_audio data/generated_videos
echo "   ✅ Data directories ready"

# --- Download models via ModelScope ---
echo ""
echo "📦 Step 5/5: Downloading models via ModelScope..."
mkdir -p models

download_model() {
    local model_id="$1"
    local local_dir="$2"
    if [ -d "$local_dir" ] && [ -n "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "   ✅ Already exists: $local_dir"
    else
        echo "   Downloading $model_id → $local_dir ..."
        modelscope download --model "$model_id" --local_dir "$local_dir"
        echo "   ✅ Downloaded: $model_id"
    fi
}

# SoulX-FlashHead 1.3B — via ModelScope
FLASHCKPT="${FLASHHEAD_CKPT_DIR:-${SCRIPT_DIR}/models/SoulX-FlashHead-1_3B}"
download_model "Soul-AILab/SoulX-FlashHead-1_3B" "$FLASHCKPT"

# Wav2Vec 音频编码器
WAV2VEC="${FLASHHEAD_WAV2VEC_DIR:-${SCRIPT_DIR}/models/wav2vec2-base-960h}"
download_model "AI-ModelScope/wav2vec2-base-960h" "$WAV2VEC"

# LongCat-AudioDiT 语音克隆/TTS 模型
AUDIODIT="${AUDIODIT_MODEL_DIR:-${SCRIPT_DIR}/models/LongCat-AudioDiT-1B}"
download_model "meituan-longcat/LongCat-AudioDiT-1B" "$AUDIODIT"

echo ""
echo "🎉 Setup complete! Start the server with:"
echo "   bash run.sh"
echo "   # or"
echo "   python start.py"
echo ""
