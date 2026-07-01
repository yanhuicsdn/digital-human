"""Background task queue for processing video generation jobs."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import traceback
import uuid
from typing import Optional

from loguru import logger

from app.config import (
    TTS_DEFAULT_NFE as TTS_NFE,
    TTS_GUIDANCE_STRENGTH,
    TTS_GUIDANCE_METHOD,
    TTS_SAMPLE_RATE,
    VIDEO_AUDIO_ENCODE_MODE,
    VIDEO_DEFAULT_SEED,
    VIDEO_USE_FACE_CROP,
)
from app.database import db
from app.storage import (
    get_avatar_audio_path,
    get_avatar_image_path,
    save_generated_audio,
    save_generated_video,
)
from app.services.tts_service import generate_speech, load_model as load_tts_model
from app.services.video_service import generate_video, load_model as load_video_model


class TaskQueue:
    """Background task queue that processes video generation jobs sequentially."""

    def __init__(self):
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_task_id: Optional[str] = None

    def start(self):
        """Start the background worker."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Task queue worker already running")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Task queue worker started")

    def stop(self):
        """Stop the background worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        logger.info("Task queue worker stopped")

    @property
    def current_task_id(self) -> Optional[str]:
        return self._current_task_id

    def _worker_loop(self):
        """Main worker loop — processes queued tasks one by one."""
        logger.info("Task worker loop started")

        # Load models on startup
        try:
            logger.info("Pre-loading models...")
            load_tts_model()
            load_video_model()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to pre-load models: {e}")
            # Will retry on each task

        while not self._stop_event.is_set():
            try:
                # Find the next queued task
                tasks = db.list_tasks()
                queued = [t for t in tasks if t["status"] == "queued"]

                if not queued:
                    time.sleep(1)
                    continue

                # Pick the oldest queued task
                next_task = sorted(queued, key=lambda t: t["created_at"])[0]
                task_id = next_task["id"]
                self._current_task_id = task_id

                logger.info(f"Processing task {task_id}...")
                db.update_task(task_id, status="processing", progress=0.0)

                self._process_task(task_id)

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(2)

        logger.info("Task worker loop ended")

    def _process_task(self, task_id: str):
        """Process a single video generation task."""
        task = db.get_task(task_id)
        if task is None:
            logger.error(f"Task {task_id} not found")
            return

        avatar_id = task["avatar_id"]
        text = task["text"]

        avatar = db.get_avatar(avatar_id)
        if avatar is None:
            db.update_task(task_id, status="failed", error=f"Avatar {avatar_id} not found")
            return

        try:
            # --- Step 1: Generate speech via TTS ---
            db.update_task(task_id, progress=0.1, status="processing")
            logger.info(f"[Task {task_id}] Step 1: Generating TTS audio...")

            avatar_audio_path = get_avatar_audio_path(avatar["audio_path"])
            avatar_image_path = get_avatar_image_path(avatar["image_path"])

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tts_output_path = tmp.name

            try:
                generate_speech(
                    text=text,
                    output_path=tts_output_path,
                    prompt_text=avatar.get("prompt_text", ""),
                    prompt_audio=avatar_audio_path,
                    nfe=TTS_NFE,
                    guidance_strength=TTS_GUIDANCE_STRENGTH,
                    guidance_method=TTS_GUIDANCE_METHOD,
                )
            except Exception as e:
                raise RuntimeError(f"TTS generation failed: {e}")

            # Save TTS result
            audio_rel_path = save_generated_audio(tts_output_path)
            db.update_task(task_id, progress=0.4, status="processing", audio_path=audio_rel_path)
            logger.info(f"[Task {task_id}] TTS audio saved: {audio_rel_path}")

            # --- Step 2: Generate video via FlashHead ---
            db.update_task(task_id, progress=0.4, status="processing")
            logger.info(f"[Task {task_id}] Step 2: Generating video...")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                video_output_path = tmp.name

            try:
                generate_video(
                    cond_image_path=avatar_image_path,
                    audio_path=tts_output_path,
                    output_video_path=video_output_path,
                    seed=VIDEO_DEFAULT_SEED,
                    use_face_crop=VIDEO_USE_FACE_CROP,
                    audio_encode_mode=VIDEO_AUDIO_ENCODE_MODE,
                )
            except Exception as e:
                raise RuntimeError(f"Video generation failed: {e}")

            # Save video result
            video_rel_path = save_generated_video(video_output_path)
            db.update_task(task_id, progress=1.0, status="completed", video_path=video_rel_path)
            logger.info(f"[Task {task_id}] Video saved: {video_rel_path}")

            # Cleanup temp files
            try:
                os.unlink(tts_output_path)
                os.unlink(video_output_path)
            except OSError:
                pass

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[Task {task_id}] Failed: {error_msg}")
            logger.error(traceback.format_exc())
            db.update_task(task_id, status="failed", error=error_msg)


# Global singleton
task_queue = TaskQueue()
