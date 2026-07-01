"""LongCat-AudioDiT integration for TTS / voice cloning."""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger

# Lazy imports — will only be imported when TTS is actually used
AudioDiTModel = None
AutoTokenizer = None
load_audio_fn = None
normalize_text_fn = None
approx_duration_from_text_fn = None

_model = None
_tokenizer = None
_device = None


def _lazy_import():
    """Import AudioDiT modules on first use."""
    global AudioDiTModel, AutoTokenizer, load_audio_fn, normalize_text_fn, approx_duration_from_text_fn

    if AudioDiTModel is not None:
        return

    try:
        import audiodit  # noqa: F401 — auto-registers with transformers
        from audiodit import AudioDiTModel as ADM
        from transformers import AutoTokenizer as AT

        # Import helpers
        from app.utils import load_audio, normalize_text, approx_duration_from_text

        AudioDiTModel = ADM
        AutoTokenizer = AT
        load_audio_fn = load_audio
        normalize_text_fn = normalize_text
        approx_duration_from_text_fn = approx_duration_from_text
        logger.info("AudioDiT modules imported successfully")
    except ImportError as e:
        logger.warning(f"AudioDiT import failed: {e}. TTS/voice cloning will be unavailable.")
        raise


def load_model(model_dir: Optional[str] = None) -> None:
    """Load the AudioDiT model into memory (global singleton)."""
    global _model, _tokenizer, _device

    if _model is not None:
        logger.info("AudioDiT model already loaded, reusing")
        return

    _lazy_import()

    if model_dir is None:
        from app.config import AUDIODIT_MODEL_DIR as default_dir
        model_dir = default_dir

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading AudioDiT model from {model_dir} on {_device}...")

    try:
        _model = AudioDiTModel.from_pretrained(model_dir).to(_device)
        _model.vae.to_half()
        _model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(_model.config.text_encoder_model)
        logger.info("AudioDiT model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load AudioDiT model: {e}")
        _model = None
        _tokenizer = None
        raise


def unload_model() -> None:
    """Unload the model from memory."""
    global _model, _tokenizer, _device
    _model = None
    _tokenizer = None
    _device = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("AudioDiT model unloaded")


def is_loaded() -> bool:
    """Check if the model is loaded."""
    return _model is not None


def generate_speech(
    text: str,
    output_path: str,
    prompt_text: Optional[str] = None,
    prompt_audio: Optional[str] = None,
    nfe: int = 16,
    guidance_strength: float = 4.0,
    guidance_method: str = "apg",
    seed: int = 1024,
) -> str:
    """
    Generate speech using LongCat-AudioDiT.

    Args:
        text: Text to synthesize.
        output_path: Path to save the output WAV file.
        prompt_text: Text of the prompt audio (for voice cloning).
        prompt_audio: Path to the prompt audio file (for voice cloning).
        nfe: Number of ODE steps.
        guidance_strength: CFG/APG strength.
        guidance_method: "cfg" or "apg".
        seed: Random seed.

    Returns:
        Path to the generated audio file.
    """
    if _model is None:
        raise RuntimeError("AudioDiT model not loaded. Call load_model() first.")

    device = _device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = _model
    tokenizer = _tokenizer
    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    # Text preparation
    text = normalize_text_fn(text)
    no_prompt = prompt_audio is None

    if not no_prompt:
        prompt_text_norm = normalize_text_fn(prompt_text)
        full_text = f"{prompt_text_norm} {text}"
    else:
        full_text = text

    logger.info(f"TTS full text: {full_text}")
    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

    # Prompt audio
    if not no_prompt:
        prompt_wav = load_audio_fn(prompt_audio, sr).unsqueeze(0)

        # Compute prompt duration
        off = 3
        pw = load_audio_fn(prompt_audio, sr)
        if pw.shape[-1] % full_hop != 0:
            pw = torch.nn.functional.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
        pw = torch.nn.functional.pad(pw, (0, full_hop * off))
        with torch.no_grad():
            plt = model.vae.encode(pw.unsqueeze(0).to(device))
        if off:
            plt = plt[..., :-off]
        prompt_dur = plt.shape[-1]
    else:
        prompt_wav = None
        prompt_dur = 0

    # Duration estimation
    prompt_time = prompt_dur * full_hop / sr
    dur_sec = approx_duration_from_text_fn(text, max_duration=max_duration - prompt_time)
    if not no_prompt:
        approx_pd = approx_duration_from_text_fn(prompt_text, max_duration=max_duration)
        ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
        dur_sec = dur_sec * ratio

    logger.info(f"Approx TTS duration: {dur_sec:.3f}s")
    duration = int(dur_sec * sr // full_hop)
    duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

    # Move inputs to device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    if prompt_wav is not None:
        prompt_wav = prompt_wav.to(device)

    # Generate
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_audio=prompt_wav,
            duration=duration,
            steps=nfe,
            cfg_strength=guidance_strength,
            guidance_method=guidance_method,
        )

    wav = output.waveform.squeeze().detach().cpu().numpy()
    sf.write(output_path, wav, sr)
    logger.info(f"TTS audio saved to {output_path} ({len(wav)/sr:.2f}s)")
    return output_path
