"""File storage helpers for avatars and generated files."""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from app.config import AVATAR_DIR, GENERATED_AUDIO_DIR, GENERATED_VIDEO_DIR


def save_avatar_image(file_path: str, ext: Optional[str] = None) -> str:
    """Save an avatar image and return the relative path."""
    if ext is None:
        ext = os.path.splitext(file_path)[1] or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = AVATAR_DIR / filename
    shutil.copy2(file_path, str(dest))
    return f"avatars/{filename}"


def save_avatar_audio(file_path: str, ext: Optional[str] = None) -> str:
    """Save an avatar audio clip and return the relative path."""
    if ext is None:
        ext = os.path.splitext(file_path)[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = AVATAR_DIR / filename
    shutil.copy2(file_path, str(dest))
    return f"avatars/{filename}"


def save_generated_audio(file_path: str) -> str:
    """Save a generated TTS audio file and return the relative path."""
    ext = os.path.splitext(file_path)[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = GENERATED_AUDIO_DIR / filename
    shutil.copy2(file_path, str(dest))
    return f"generated_audio/{filename}"


def save_generated_video(file_path: str) -> str:
    """Save a generated video file and return the relative path."""
    ext = os.path.splitext(file_path)[1] or ".mp4"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = GENERATED_VIDEO_DIR / filename
    shutil.copy2(file_path, str(dest))
    return f"generated_videos/{filename}"


def get_avatar_image_path(relative_path: str) -> str:
    """Get the full path to an avatar image."""
    return str(AVATAR_DIR.parent / relative_path)


def get_avatar_audio_path(relative_path: str) -> str:
    """Get the full path to an avatar audio."""
    return str(AVATAR_DIR.parent / relative_path)


def get_generated_video_path(relative_path: str) -> str:
    """Get the full path to a generated video."""
    return str(GENERATED_VIDEO_DIR.parent / relative_path)


def get_generated_audio_path(relative_path: str) -> str:
    """Get the full path to a generated audio."""
    return str(GENERATED_AUDIO_DIR.parent / relative_path)


def delete_file(relative_path: str) -> bool:
    """Delete a file by its relative path."""
    full_path = AVATAR_DIR.parent / relative_path
    if full_path.exists():
        os.remove(str(full_path))
        return True
    return False
