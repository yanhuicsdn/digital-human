"""LongCat-Video-Avatar-1.5 integration for talking-head video generation.

Alternative backend to SoulX-FlashHead. Switch via config.VIDEO_BACKEND.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
from loguru import logger

# Lazy-loaded globals
_pipeline = None
_pipeline_config = {}


def _ensure_longcat_path():
    """Ensure longcat_video/ is on sys.path."""
    project_root = Path(__file__).parent.parent.parent.resolve()
    lc_dir = project_root / "longcat_video"
    if not lc_dir.is_dir():
        raise RuntimeError(
            f"longcat_video/ directory not found at {lc_dir}. "
            "Run setup_lc.sh first."
        )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if os.getcwd() != str(project_root):
        os.chdir(str(project_root))


def _lazy_import():
    """Import LongCat modules on first use."""
    _ensure_longcat_path()
    try:
        from longcat_video.pipeline_longcat_video_avatar import (
            LongCatVideoAvatarPipeline,
        )
        from longcat_video.modules.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.avatar.longcat_video_dit_avatar import (
            LongCatVideoAvatarTransformer3DModel,
        )
        from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        return (
            LongCatVideoAvatarPipeline,
            FlowMatchEulerDiscreteScheduler,
            AutoencoderKLWan,
            LongCatVideoAvatarTransformer3DModel,
            get_audio_encoder,
            get_audio_feature_extractor,
        )
    except ImportError as e:
        raise ImportError(
            f"Cannot import longcat_video: {e}. "
            "Make sure LongCat-Video is installed. See README for setup."
        )


def load_model(
    ckpt_dir: Optional[str] = None,
    model_type: str = "avatar-v1.5",
    use_distill: bool = True,
    use_int8: bool = False,
) -> None:
    """Load the LongCat-Video-Avatar model (global singleton)."""
    global _pipeline, _pipeline_config

    if _pipeline is not None:
        logger.info("LongCat model already loaded, reusing")
        return

    if ckpt_dir is None:
        from app.config import LONGCAVA_CKPT_DIR as default_ckpt
        ckpt_dir = default_ckpt

    logger.info(f"Loading LongCat-Video-Avatar-1.5: type={model_type}, ckpt={ckpt_dir}")

    (
        LongCatVideoAvatarPipeline,
        FlowMatchEulerDiscreteScheduler,
        AutoencoderKLWan,
        LongCatVideoAvatarTransformer3DModel,
        get_audio_encoder,
        get_audio_feature_extractor,
    ) = _lazy_import()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    try:
        # Load audio encoder
        audio_encoder = get_audio_encoder(
            os.path.join(ckpt_dir, "whisper-large-v3"),
            model_type=model_type,
        ).to(device, dtype=dtype)

        # Load VAE
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(ckpt_dir, "vae"),
            torch_dtype=dtype,
        ).to(device)

        # Load DiT
        text_encoder = None  # Will be loaded by pipeline
        tokenizer = None

        if use_int8:
            from longcat_video.modules.quantization import load_quantized_dit
            dit = load_quantized_dit(
                os.path.join(ckpt_dir, "base_model_int8"),
                dtype=dtype,
                device=device,
            )
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
                os.path.join(ckpt_dir, "base_model"),
                torch_dtype=dtype,
            ).to(device)

        # Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler()

        # Build pipeline
        _pipeline = LongCatVideoAvatarPipeline(
            vae=vae,
            dit=dit,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            audio_encoder=audio_encoder,
            device=device,
            dtype=dtype,
        )

        _pipeline_config = {
            "use_distill": use_distill,
            "model_type": model_type,
            "use_int8": use_int8,
        }

        logger.info("LongCat-Video-Avatar-1.5 model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LongCat model: {e}")
        _pipeline = None
        raise


def unload_model() -> None:
    """Unload the model from memory."""
    global _pipeline, _pipeline_config
    _pipeline = None
    _pipeline_config = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("LongCat model unloaded")


def is_loaded() -> bool:
    return _pipeline is not None


def generate_video(
    cond_image_path: str,
    audio_path: str,
    output_video_path: str,
    seed: int = 42,
    use_face_crop: bool = False,
    audio_encode_mode: str = "once",
    progress_callback=None,
) -> str:
    """Generate a talking-head video using LongCat-Video-Avatar-1.5."""
    if _pipeline is None:
        raise RuntimeError("LongCat model not loaded. Call load_model() first.")

    pipeline = _pipeline
    cfg = _pipeline_config
    device = next(pipeline.dit.parameters()).device

    # Prepare input image
    if progress_callback:
        progress_callback(0.05, "Preparing input data...")

    from diffusers.utils import load_image
    cond_image = load_image(cond_image_path).convert("RGB")

    # Prepare audio
    if progress_callback:
        progress_callback(0.1, "Loading audio...")

    import librosa
    speech_array, _ = librosa.load(audio_path, sr=16000, mono=True)

    if progress_callback:
        progress_callback(0.15, f"Audio loaded ({len(speech_array)/16000:.1f}s), generating video...")

    torch.manual_seed(seed)

    with torch.no_grad():
        # Run inference
        video_frames = pipeline(
            image=cond_image,
            audio=speech_array,
            model_type=cfg["model_type"],
            use_distill=cfg["use_distill"],
            generator=torch.Generator(device=device).manual_seed(seed),
            num_frames=81,          # ~3.2s at 25fps, will loop via video continuation
            fps=25,
        )

    if progress_callback:
        progress_callback(0.85, "Saving video...")

    # Save video
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    temp_video_path = output_video_path.replace(".mp4", "_tmp.mp4")

    try:
        # Write video frames with imageio
        frames_np = video_frames.cpu().numpy()
        if frames_np.ndim == 4:  # (T, C, H, W)
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))  # (T, H, W, C)
        frames_np = (frames_np.clip(0, 1) * 255).astype(np.uint8)

        with imageio.get_writer(
            temp_video_path,
            format="mp4",
            mode="I",
            fps=25,
            codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for frame in frames_np:
                writer.append_data(frame)

        # Merge audio
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_video_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Video saved to {output_video_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save video: {e}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    if progress_callback:
        progress_callback(1.0, "Done!")

    return output_video_path
