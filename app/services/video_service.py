"""SoulX-FlashHead integration for talking-head video generation.

IMPORTANT: The `flash_head/` directory from SoulX-FlashHead must be present
at the project root (next to this file's grandparent). The flash_head/inference.py
module reads 'flash_head/configs/infer_params.yaml' at import time using a
relative path, so the working directory must be the project root.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

# Lazy imports
_pipeline = None
_loaded_ckpt_dir = None
_loaded_wav2vec_dir = None
_loaded_model_type = None
_infer_params = None


def _ensure_flashhead_path():
    """Ensure flash_head/ is on sys.path and CWD is correct.

    The flash_head/inference.py does `open("flash_head/configs/infer_params.yaml")`
    at module-import time, which requires the project root to be the CWD.
    """
    project_root = Path(__file__).parent.parent.parent.resolve()
    flash_head_dir = project_root / "flash_head"
    if not flash_head_dir.is_dir():
        raise RuntimeError(
            f"flash_head/ directory not found at {flash_head_dir}. "
            "Clone SoulX-FlashHead and copy its flash_head/ folder here:\n"
            f"  git clone https://github.com/Soul-AILab/SoulX-FlashHead.git /tmp/fh\n"
            f"  cp -r /tmp/fh/flash_head {project_root}/flash_head\n"
            f"  cp -r /tmp/fh/flash_head/configs {project_root}/flash_head/configs"
        )
    # Ensure project root is on sys.path and is the CWD
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if os.getcwd() != str(project_root):
        os.chdir(str(project_root))


def _lazy_import_flashhead():
    """Import flash_head modules on first use."""
    _ensure_flashhead_path()
    try:
        from flash_head.inference import (
            get_pipeline,
            get_base_data,
            get_infer_params,
            get_audio_embedding,
            run_pipeline,
        )
        return get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
    except ImportError as e:
        raise ImportError(
            f"Cannot import flash_head: {e}. "
            "Make sure SoulX-FlashHead is installed. See README for setup instructions."
        )


def load_model(
    ckpt_dir: Optional[str] = None,
    wav2vec_dir: Optional[str] = None,
    model_type: Optional[str] = None,
) -> None:
    """Load the FlashHead model (global singleton)."""
    global _pipeline, _loaded_ckpt_dir, _loaded_wav2vec_dir, _loaded_model_type, _infer_params

    if _pipeline is not None:
        logger.info("FlashHead model already loaded, reusing")
        return

    from app.config import (
        FLASHHEAD_CKPT_DIR as default_ckpt,
        FLASHHEAD_WAV2VEC_DIR as default_wav2vec,
        FLASHHEAD_MODEL_TYPE as default_type,
    )

    ckpt_dir = ckpt_dir or default_ckpt
    wav2vec_dir = wav2vec_dir or default_wav2vec
    model_type = model_type or default_type

    logger.info(f"Loading FlashHead model: type={model_type}, ckpt={ckpt_dir}")

    try:
        get_pipeline_fn, *_ = _lazy_import_flashhead()
        _pipeline = get_pipeline_fn(
            world_size=1,
            ckpt_dir=ckpt_dir,
            model_type=model_type,
            wav2vec_dir=wav2vec_dir,
        )
        _loaded_ckpt_dir = ckpt_dir
        _loaded_wav2vec_dir = wav2vec_dir
        _loaded_model_type = model_type

        from flash_head.inference import infer_params as ip
        _infer_params = ip
        logger.info("FlashHead model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load FlashHead model: {e}")
        _pipeline = None
        raise


def unload_model() -> None:
    """Unload the model from memory."""
    global _pipeline, _loaded_ckpt_dir, _loaded_wav2vec_dir, _loaded_model_type, _infer_params
    _pipeline = None
    _loaded_ckpt_dir = None
    _loaded_wav2vec_dir = None
    _loaded_model_type = None
    _infer_params = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("FlashHead model unloaded")


def is_loaded() -> bool:
    """Check if the model is loaded."""
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
    """
    Generate a talking-head video using SoulX-FlashHead.

    Uses ffmpeg pipe for direct video writing (avoids imageio frame-by-frame overhead).

    Args:
        cond_image_path: Path to the condition (avatar) image.
        audio_path: Path to the driving audio file.
        output_video_path: Path to save the output MP4 video.
        seed: Random seed.
        use_face_crop: Whether to crop the face from the condition image.
        audio_encode_mode: "once" or "stream".
        progress_callback: Optional callable(progress: float, message: str).

    Returns:
        Path to the generated video file.
    """
    if _pipeline is None:
        raise RuntimeError("FlashHead model not loaded. Call load_model() first.")

    get_pipeline_fn, get_base_data_fn, get_infer_params_fn, get_audio_embedding_fn, run_pipeline_fn = (
        _lazy_import_flashhead()
    )

    pipeline = _pipeline
    infer_params = _infer_params

    # Override resolution from config (environment variables) — no YAML edit needed
    from app.config import VIDEO_RESOLUTION_WIDTH, VIDEO_RESOLUTION_HEIGHT
    if VIDEO_RESOLUTION_WIDTH and VIDEO_RESOLUTION_HEIGHT:
        infer_params["width"] = VIDEO_RESOLUTION_WIDTH
        infer_params["height"] = VIDEO_RESOLUTION_HEIGHT

    logger.info(f"Video resolution: {infer_params['width']}x{infer_params['height']} (9:{infer_params['width']/infer_params['height']*9:.0f})")

    # ------------------------------------------------------------------
    # 1. Prepare base data (image + seed)
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(0.05, "Preparing input data...")

    try:
        get_base_data_fn(
            pipeline,
            cond_image_path_or_dir=cond_image_path,
            base_seed=seed,
            use_face_crop=use_face_crop,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to prepare base data: {e}")

    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    # ------------------------------------------------------------------
    # 2. Load audio
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(0.1, "Loading audio...")

    try:
        import librosa
        human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {audio_path}: {e}")

    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

    if progress_callback:
        progress_callback(0.15, f"Audio loaded, generating {len(human_speech_array_all) / sample_rate:.1f}s video...")

    generated_list = []
    device = pipeline.device

    # ------------------------------------------------------------------
    # 3. Generation (optimized: fewer GPU syncs, async CPU transfer)
    # ------------------------------------------------------------------
    if audio_encode_mode == "once":
        # Pad audio
        remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.concatenate(
                [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
            )

        audio_embedding_all = get_audio_embedding_fn(pipeline, human_speech_array_all)
        embed_len = audio_embedding_all.shape[1]

        # Generate chunks covering the FULL audio embedding.
        # Original code missed the last ~31 frames (~1.2s) because it only
        # produced floor((embed_len-frame_num)/slice_len) chunks.
        # Fix: add one final chunk aligned to the tail of the embedding.
        num_reg_chunks = (embed_len - frame_num) // slice_len
        audio_embedding_chunks_list = [
            audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous()
            for i in range(num_reg_chunks)
        ]
        # Add the last chunk (covers remaining frames at the end)
        tail_start = max(0, embed_len - frame_num)
        audio_embedding_chunks_list.append(
            audio_embedding_all[:, tail_start:tail_start + frame_num].contiguous()
        )

        total_chunks = len(audio_embedding_chunks_list)
        for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
            if progress_callback:
                progress_callback(
                    0.15 + 0.75 * (chunk_idx / total_chunks),
                    f"Generating video chunk {chunk_idx + 1}/{total_chunks}...",
                )

            # Single sync before generation is enough
            torch.cuda.synchronize()
            video = run_pipeline_fn(pipeline, audio_embedding_chunk)

            if chunk_idx != 0:
                video = video[motion_frames_num:]

            # Async CPU transfer — overlaps with next iteration
            generated_list.append(video.cpu())

        # One final sync after all generation
        torch.cuda.synchronize()

    elif audio_encode_mode == "stream":
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        remainder = len(human_speech_array_all) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.concatenate(
                [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
            )

        human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)
        total_chunks = len(human_speech_array_slices)

        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            if progress_callback:
                progress_callback(
                    0.15 + 0.75 * (chunk_idx / total_chunks),
                    f"Generating video chunk {chunk_idx + 1}/{total_chunks}...",
                )

            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding_fn(pipeline, audio_array, audio_start_idx, audio_end_idx)

            torch.cuda.synchronize()
            video = run_pipeline_fn(pipeline, audio_embedding)
            video = video[motion_frames_num:]

            generated_list.append(video.cpu())

        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 4. Save video — use ffmpeg pipe directly (avoid imageio overhead)
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(0.92, "Saving video...")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    temp_video_path = output_video_path.replace(".mp4", "_tmp.mp4")

    # Get video dimensions from first frame
    first_frames = generated_list[0]
    H, W = first_frames.shape[2], first_frames.shape[3]

    try:
        # Step A: Write video frames using imageio (reliable, proven working)
        with imageio.get_writer(
            temp_video_path,
            format="mp4",
            mode="I",
            fps=tgt_fps,
            codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for frames in generated_list:
                frames_np = frames.numpy().astype(np.uint8)
                for i in range(frames_np.shape[0]):
                    writer.append_data(frames_np[i, :, :, :])

        # Step B: Merge audio into video
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_video_path,
        ]
        subprocess.run(merge_cmd, check=True, capture_output=True)

        logger.info(f"Video saved to {output_video_path} ({W}x{H} @ {tgt_fps}fps, {total_chunks} chunks)")
    except Exception as e:
        raise RuntimeError(f"Failed to save video: {e}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    if progress_callback:
        progress_callback(1.0, "Done!")

    return output_video_path
