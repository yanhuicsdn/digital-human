"""Pydantic models for the API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateTextRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Text prompt for LLM")
    api_key: str = Field(default="", description="API key for LLM")
    model_id: str = Field(default="qwen-turbo", description="LLM model ID")


class SubmitTaskRequest(BaseModel):
    """Request model for submitting a video generation task."""
    text: str = Field(..., description="Text content for the digital human to speak")
    avatar_id: str = Field(..., description="Avatar ID to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speaking speed (0.5=half speed, 1.0=normal, 2.0=double speed)")


class AvatarResponse(BaseModel):
    """Response model for an avatar."""
    id: str
    name: str
    description: str = ""
    prompt_text: str = ""
    image_url: str = ""
    audio_url: str = ""
    created_at: str = ""


class TaskResponse(BaseModel):
    """Response model for a task."""
    id: str
    avatar_id: str
    text: str
    status: TaskStatus
    progress: float = 0.0
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class QueueStatusResponse(BaseModel):
    """Queue status response."""
    queue_length: int
    active_tasks: int
    completed_tasks: int
    failed_tasks: int


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
