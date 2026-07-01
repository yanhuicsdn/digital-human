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
echo "📦 Step 1/4: Installing Python dependencies..."
$PYTHON -m pip install -r requirements.txt -q
echo "   ✅ Dependencies installed"

# --- Clone SoulX-FlashHead for flash_head package ---
echo ""
echo "📦 Step 2/4: Setting up SoulX-FlashHead dependency..."
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
echo "📁 Step 3/4: Ensuring data directories..."
mkdir -p data/avatars data/generated_audio data/generated_videos
echo "   ✅ Data directories ready"

# --- Verify model paths (warn only) ---
echo ""
echo "🔍 Step 4/4: Checking model paths..."
FLASHCKPT="${FLASHHEAD_CKPT_DIR:-${SCRIPT_DIR}/models/SoulX-FlashHead-1_3B}"
WAV2VEC="${FLASHHEAD_WAV2VEC_DIR:-${SCRIPT_DIR}/models/wav2vec2-base-960h}"
AUDIODIT="${AUDIODIT_MODEL_DIR:-${SCRIPT_DIR}/models/LongCat-AudioDiT-1B}"

if [ -d "$FLASHCKPT" ]; then
    echo "   ✅ FlashHead model: $FLASHCKPT"
else
    echo "   ⚠️  FlashHead model not found at: $FLASHCKPT"
    echo "       Download: huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir $FLASHCKPT"
fi

if [ -d "$WAV2VEC" ]; then
    echo "   ✅ Wav2Vec model: $WAV2VEC"
else
    echo "   ⚠️  Wav2Vec model not found at: $WAV2VEC"
    echo "       Download: huggingface-cli download facebook/wav2vec2-base-960h --local-dir $WAV2VEC"
fi

if [ -d "$AUDIODIT" ]; then
    echo "   ✅ AudioDiT model: $AUDIODIT"
else
    echo "   ⚠️  AudioDiT model not found at: $AUDIODIT"
    echo "       Download: huggingface-cli download meituan-longcat/LongCat-AudioDiT-1B --local-dir $AUDIODIT"
fi

echo ""
echo "🎉 Setup complete! Start the server with:"
echo "   bash run.sh"
echo "   # or"
echo "   python start.py"
echo ""
