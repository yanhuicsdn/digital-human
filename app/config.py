"""Application configuration."""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directories
DATA_DIR = BASE_DIR / "data"
AVATAR_DIR = DATA_DIR / "avatars"
GENERATED_AUDIO_DIR = DATA_DIR / "generated_audio"
GENERATED_VIDEO_DIR = DATA_DIR / "generated_videos"
DB_PATH = DATA_DIR / "db.json"

# Create directories
for d in [DATA_DIR, AVATAR_DIR, GENERATED_AUDIO_DIR, GENERATED_VIDEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== Model paths =====

# SoulX-FlashHead
FLASHHEAD_CKPT_DIR = os.environ.get(
    "FLASHHEAD_CKPT_DIR",
    str(BASE_DIR / "models" / "SoulX-FlashHead-1_3B"),
)
FLASHHEAD_WAV2VEC_DIR = os.environ.get(
    "FLASHHEAD_WAV2VEC_DIR",
    str(BASE_DIR / "models" / "wav2vec2-base-960h"),
)
FLASHHEAD_MODEL_TYPE = os.environ.get("FLASHHEAD_MODEL_TYPE", "lite")  # "pro" or "lite"

# LongCat-AudioDiT
AUDIODIT_MODEL_DIR = os.environ.get(
    "AUDIODIT_MODEL_DIR",
    str(BASE_DIR / "models" / "LongCat-AudioDiT-1B"),
)

# ===== Server config =====
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))

# ===== TTS config =====
TTS_SAMPLE_RATE = 24000  # LongCat-AudioDiT output sample rate
TTS_DEFAULT_NFE = int(os.environ.get("TTS_NFE", "16"))  # 8=faster/lower quality, 16=balanced, 32=best quality
TTS_GUIDANCE_STRENGTH = float(os.environ.get("TTS_GUIDANCE_STRENGTH", "4.0"))  # higher = more expressive
TTS_GUIDANCE_METHOD = os.environ.get("TTS_GUIDANCE_METHOD", "apg")  # "cfg" or "apg" (APG is better)

# ===== Video generation config =====
VIDEO_AUDIO_ENCODE_MODE = os.environ.get("VIDEO_AUDIO_ENCODE_MODE", "once")  # "once" or "stream"
VIDEO_USE_FACE_CROP = os.environ.get("VIDEO_USE_FACE_CROP", "False").lower() == "true"
VIDEO_DEFAULT_SEED = int(os.environ.get("VIDEO_DEFAULT_SEED", "42"))

# Video resolution (from infer_params.yaml — read at runtime)
# The actual resolution is set in flash_head/configs/infer_params.yaml
VIDEO_RESOLUTION_WIDTH = int(os.environ.get("VIDEO_RESOLUTION_WIDTH", "576"))
VIDEO_RESOLUTION_HEIGHT = int(os.environ.get("VIDEO_RESOLUTION_HEIGHT", "1024"))

# FFmpeg video encoding quality (lower CRF = better quality, 0=lossless, 18=visually lossless, 23=default)
VIDEO_CRF = int(os.environ.get("VIDEO_CRF", "18"))
VIDEO_PRESET = os.environ.get("VIDEO_PRESET", "medium")  # ultrafast, fast, medium, slow, veryslow

# ===== LLM config for text generation =====
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_DEFAULT_MODEL = os.environ.get("LLM_DEFAULT_MODEL", "qwen-turbo")
