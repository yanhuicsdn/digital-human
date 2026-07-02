"""Simple JSON file-based database for avatars and tasks."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any, Optional

from app.config import DB_PATH


class JsonDatabase:
    """Thread-safe JSON file database."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"avatars": {}, "tasks": {}}
        self._load()

    def _load(self):
        """Load data from disk."""
        if DB_PATH.exists():
            try:
                with open(str(DB_PATH), "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {"avatars": {}, "tasks": {}}

    def _save(self):
        """Save data to disk (caller must hold lock)."""
        with open(str(DB_PATH), "w") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ---- Avatars ----

    def create_avatar(
        self,
        avatar_id: str,
        name: str,
        description: str,
        prompt_text: str,
        image_path: str,
        audio_path: str,
    ) -> dict:
        """Create a new avatar record."""
        with self._lock:
            now = datetime.now().isoformat()
            record = {
                "id": avatar_id,
                "name": name,
                "description": description,
                "prompt_text": prompt_text,
                "image_path": image_path,
                "audio_path": audio_path,
                "created_at": now,
            }
            self._data["avatars"][avatar_id] = record
            self._save()
            return record

    def get_avatar(self, avatar_id: str) -> Optional[dict]:
        """Get an avatar by ID."""
        with self._lock:
            return self._data["avatars"].get(avatar_id)

    def list_avatars(self) -> list[dict]:
        """List all avatars."""
        with self._lock:
            return list(self._data["avatars"].values())

    def delete_avatar(self, avatar_id: str) -> bool:
        """Delete an avatar by ID. Returns True if deleted."""
        with self._lock:
            if avatar_id in self._data["avatars"]:
                del self._data["avatars"][avatar_id]
                self._save()
                return True
            return False

    # ---- Tasks ----

    def create_task(self, task_id: str, avatar_id: str, text: str, speed: float = 1.0) -> dict:
        """Create a new task record."""
        with self._lock:
            now = datetime.now().isoformat()
            record = {
                "id": task_id,
                "avatar_id": avatar_id,
                "text": text,
                "speed": speed,
                "status": "queued",
                "progress": 0.0,
                "video_path": None,
                "audio_path": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            self._data["tasks"][task_id] = record
            self._save()
            return record

    def get_task(self, task_id: str) -> Optional[dict]:
        """Get a task by ID."""
        with self._lock:
            return self._data["tasks"].get(task_id)

    def list_tasks(self) -> list[dict]:
        """List all tasks."""
        with self._lock:
            return list(self._data["tasks"].values())

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[dict]:
        """Update task fields."""
        with self._lock:
            task = self._data["tasks"].get(task_id)
            if task is None:
                return None
            if status is not None:
                task["status"] = status
            if progress is not None:
                task["progress"] = progress
            if video_path is not None:
                task["video_path"] = video_path
            if audio_path is not None:
                task["audio_path"] = audio_path
            if error is not None:
                task["error"] = error
            task["updated_at"] = datetime.now().isoformat()
            self._save()
            return task

    def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID."""
        with self._lock:
            if task_id in self._data["tasks"]:
                del self._data["tasks"][task_id]
                self._save()
                return True
            return False

    def get_queue_length(self) -> int:
        """Get the number of queued tasks."""
        with self._lock:
            return sum(
                1 for t in self._data["tasks"].values() if t["status"] == "queued"
            )

    def get_queue_stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            tasks = self._data["tasks"].values()
            return {
                "queue_length": sum(1 for t in tasks if t["status"] == "queued"),
                "active_tasks": sum(1 for t in tasks if t["status"] == "processing"),
                "completed_tasks": sum(1 for t in tasks if t["status"] == "completed"),
                "failed_tasks": sum(1 for t in tasks if t["status"] == "failed"),
            }


# Global singleton
db = JsonDatabase()
