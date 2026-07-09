"""LongCat-AudioDiT integration for TTS / voice cloning.

Supports chunked generation for arbitrarily long audio: text is split into
sentence segments, each generated separately and concatenated with crossfade.
Chunks beyond the first use the tail of the previous chunk as the voice prompt,
ensuring natural voice continuity.
"""

from __future__ import annotations

import os
import re
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

# ── config ──
MAX_CHUNK_DURATION = 25  # seconds per TTS chunk (leave headroom under max_wav_duration)
PROMPT_TAIL_DURATION = 3.0  # seconds of previous chunk used as next prompt
CROSSFADE_DURATION = 0.08  # seconds of crossfade between chunks


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
        # Disable cuDNN benchmark to avoid autotuner-related init issues
        if _device.type == "cuda":
            torch.backends.cudnn.benchmark = False

        _model = AudioDiTModel.from_pretrained(model_dir).to(_device)
        _model.vae.to_half()
        _model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(_model.config.text_encoder_model)

        # ── cuDNN warmup ──────────────────────────────────────────────
        # Force cuDNN initialization with a small dummy forward pass to
        # avoid CUDNN_STATUS_NOT_INITIALIZED on the first real call.
        if _device.type == "cuda":
            logger.info("Warming up cuDNN for AudioDiT VAE...")
            try:
                # Small dummy audio (1, 1, 2048 samples ~= 0.085s at 24kHz)
                dummy_audio = torch.zeros(1, 1, 2048, device=_device, dtype=torch.float16)
                with torch.no_grad():
                    _model.vae.encode(dummy_audio)
                logger.info("cuDNN warmup complete")
            except Exception as warmup_err:
                logger.warning(f"cuDNN warmup failed (non-fatal): {warmup_err}")

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


# ── text splitting ──

_SENTENCE_SPLIT_RE = re.compile(r"([。！？.!?\n]+)")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving delimiters."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    sentences = []
    buf = ""
    for part in parts:
        buf += part
        if _SENTENCE_SPLIT_RE.fullmatch(part):
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return [s for s in sentences if s]


def _chunk_sentences(
    sentences: list[str],
    max_dur: float,
    speed: float,
    prompt_dur_override: float = 0.0,
) -> list[list[str]]:
    """Group sentences into chunks, each under *max_dur* seconds (accounting for speed)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_dur = 0.0

    for s in sentences:
        s_dur = approx_duration_from_text_fn(s, max_duration=max_dur) / speed

        if current_dur + s_dur > max_dur and current:
            chunks.append(current)
            current = [s]
            current_dur = s_dur
        else:
            current.append(s)
            current_dur += s_dur

    if current:
        chunks.append(current)

    # Safety: if any single sentence exceeds max_dur, keep it as a chunk anyway
    return chunks


# ── crossfade ──

def _crossfade(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    """Crossfade two audio arrays."""
    if fade_samples <= 0:
        return np.concatenate([a, b])
    fade_samples = min(fade_samples, len(a), len(b))
    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
    fade_out = 1.0 - fade_in
    tail = a[-fade_samples:]
    head = b[:fade_samples]
    blended = tail * fade_out + head * fade_in
    return np.concatenate([a[:-fade_samples], blended, b[fade_samples:]])


# ── core generation ──

def _generate_one(
    text: str,
    prompt_text: Optional[str],
    prompt_audio_path: Optional[str],
    nfe: int,
    guidance_strength: float,
    guidance_method: str,
    seed: int,
    speed: float,
    max_duration_limit: float,
) -> np.ndarray:
    """Generate one audio chunk and return the waveform as numpy array."""
    model = _model
    tokenizer = _tokenizer
    device = _device
    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    text_norm = normalize_text_fn(text)
    no_prompt = prompt_audio_path is None or prompt_text is None

    if not no_prompt:
        full_text = f"{normalize_text_fn(prompt_text)} {text_norm}"
    else:
        full_text = text_norm

    logger.debug(f"TTS chunk text: {full_text[:120]}...")
    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

    # Prompt audio
    if not no_prompt:
        pw = load_audio_fn(prompt_audio_path, sr)
        if pw.shape[-1] % full_hop != 0:
            pw = torch.nn.functional.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
        prompt_wav = pw.unsqueeze(0)
        with torch.no_grad():
            plt = model.vae.encode(pw[None, None, :].to(device))
        prompt_dur = plt.shape[-1]
        prompt_time = prompt_dur * full_hop / sr
    else:
        prompt_wav = None
        prompt_dur = 0
        prompt_time = 0.0

    # Duration estimation
    dur_sec = approx_duration_from_text_fn(
        text, max_duration=max_duration_limit - prompt_time
    )
    if not no_prompt:
        approx_pd = approx_duration_from_text_fn(
            prompt_text, max_duration=max_duration_limit
        )
        ratio = np.clip(prompt_time / (approx_pd or 1.0), 1.0, 1.5)
        dur_sec = dur_sec * ratio

    dur_sec = dur_sec / speed
    logger.debug(f"  chunk est duration: {dur_sec:.2f}s (speed={speed:.2f})")
    duration = int(dur_sec * sr // full_hop)
    duration = min(duration + prompt_dur, int(max_duration_limit * sr // full_hop))

    # Move inputs
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
    return wav


def generate_speech(
    text: str,
    output_path: str,
    prompt_text: Optional[str] = None,
    prompt_audio: Optional[str] = None,
    nfe: int = 16,
    guidance_strength: float = 4.0,
    guidance_method: str = "apg",
    seed: int = 1024,
    speed: float = 1.0,
) -> str:
    """
    Generate speech using LongCat-AudioDiT with chunked generation.

    Long texts are automatically split into sentence chunks, generated
    sequentially, and concatenated with crossfade. Each chunk beyond the
    first uses the tail of the previous chunk as the voice prompt,
    ensuring natural voice continuity.

    Args:
        text: Text to synthesize.
        output_path: Path to save the output WAV file.
        prompt_text: Text of the prompt audio (for voice cloning).
        prompt_audio: Path to the prompt audio file (for voice cloning).
        nfe: Number of ODE steps.
        guidance_strength: CFG/APG strength.
        guidance_method: "cfg" or "apg".
        seed: Random seed.
        speed: Speaking speed multiplier (1.0=normal, 1.2=faster, 0.8=slower).

    Returns:
        Path to the generated audio file.
    """
    if _model is None:
        raise RuntimeError("AudioDiT model not loaded. Call load_model() first.")

    sr = _model.config.sampling_rate
    full_hop = _model.config.latent_hop
    max_dur_config = _model.config.max_wav_duration  # e.g. 60
    # Leave headroom: use slightly less than max to avoid clipping
    chunk_max_dur = min(MAX_CHUNK_DURATION, max_dur_config * 0.9)

    text = normalize_text_fn(text)

    # ── Estimate total duration ──
    total_est = approx_duration_from_text_fn(text, max_duration=9999) / speed
    logger.info(
        f"TTS text length={len(text)} chars, "
        f"estimated total duration={total_est:.1f}s, "
        f"chunk_max={chunk_max_dur:.0f}s"
    )

    # ── Split into chunks ──
    sentences = _split_sentences(text)

    # If total is within a single chunk, skip chunking
    if total_est <= chunk_max_dur:
        logger.info("Single-chunk generation (short text)")
        wav = _generate_one(
            text=text,
            prompt_text=prompt_text,
            prompt_audio_path=prompt_audio,
            nfe=nfe,
            guidance_strength=guidance_strength,
            guidance_method=guidance_method,
            seed=seed,
            speed=speed,
            max_duration_limit=max_dur_config,
        )
        sf.write(output_path, wav, sr)
        logger.info(f"TTS audio saved to {output_path} ({len(wav)/sr:.2f}s)")
        return output_path

    # ── Multi-chunk generation ──
    text_chunks = _chunk_sentences(sentences, chunk_max_dur, speed)
    logger.info(f"Multi-chunk generation: {len(text_chunks)} chunks")

    chunks_wav: list[np.ndarray] = []
    prev_audio_path: Optional[str] = prompt_audio
    prev_text: Optional[str] = prompt_text

    fade_samples = int(CROSSFADE_DURATION * sr)

    for idx, chunk_sentences in enumerate(text_chunks):
        chunk_text = "".join(chunk_sentences)
        logger.info(f"  Chunk {idx + 1}/{len(text_chunks)}: ~{len(chunk_text)} chars")

        wav = _generate_one(
            text=chunk_text,
            prompt_text=prev_text,
            prompt_audio_path=prev_audio_path,
            nfe=nfe,
            guidance_strength=guidance_strength,
            guidance_method=guidance_method,
            seed=seed + idx,
            speed=speed,
            max_duration_limit=max_dur_config,
        )
        chunks_wav.append(wav)

        # Prepare prompt for next chunk: use tail of this chunk
        tail_samples = int(PROMPT_TAIL_DURATION * sr)
        if len(wav) > tail_samples * 2:
            tail_audio = wav[-tail_samples:]
            # Save tail as temp wav for next chunk's prompt
            tail_path = output_path.replace(".wav", f"_tail_{idx}.wav")
            sf.write(tail_path, tail_audio, sr)
            prev_audio_path = tail_path

            # Corresponding text: last sentence(s) of this chunk
            prev_text = chunk_sentences[-1] if chunk_sentences else chunk_text
        else:
            # If chunk is too short, fall back to original prompt
            prev_audio_path = prompt_audio
            prev_text = prompt_text

    # ── Concatenate with crossfade ──
    logger.info(f"Concatenating {len(chunks_wav)} chunks with crossfade...")
    combined = chunks_wav[0]
    for wav in chunks_wav[1:]:
        combined = _crossfade(combined, wav, fade_samples)

    sf.write(output_path, combined, sr)

    # Cleanup temp tail files
    for idx in range(len(text_chunks) - 1):
        tail_path = output_path.replace(".wav", f"_tail_{idx}.wav")
        if os.path.exists(tail_path):
            os.unlink(tail_path)

    logger.info(
        f"TTS audio saved to {output_path} "
        f"({len(combined)/sr:.2f}s, {len(text_chunks)} chunks)"
    )
    return output_path
