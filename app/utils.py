"""Utility functions shared across the project."""

import re

import librosa
import torch


def load_audio(wavpath, sr):
    """Load an audio file and return as a tensor."""
    audio, _ = librosa.load(wavpath, sr=sr, mono=True)
    return torch.from_numpy(audio)


def normalize_text(text):
    """Normalize text for TTS input."""
    text = text.lower()
    text = re.sub(r'["""\'\u2018\u2019]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def approx_duration_from_text(text, max_duration=30.0):
    """Approximate audio duration from text (char-based heuristic)."""
    EN_DUR_PER_CHAR = 0.082
    ZH_DUR_PER_CHAR = 0.21
    text = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)
