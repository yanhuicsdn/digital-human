"""Task management endpoints.

响应格式严格遵循 馨语数字人调用标准 (参见 馨语调用全套说明.md 第3节)。
"""

from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.database import db
from app.models import SubmitTaskRequest
from app.storage import get_generated_video_path, get_generated_audio_path

router = APIRouter(prefix="/api", tags=["Tasks"])


@router.post("/submit", summary="Submit Task")
async def submit_task(request: SubmitTaskRequest):
    """提交视频生成任务

    馨语标准响应格式: {"task_id": "...", "queue_position": 0}
    见 馨语调用全套说明.md §3.4
    """
    avatar = db.get_avatar(request.avatar_id)
    if avatar is None:
        raise HTTPException(
            status_code=404,
            detail=f"Avatar '{request.avatar_id}' not found",
        )

    task_id = uuid.uuid4().hex[:12]
    db.create_task(
        task_id=task_id,
        avatar_id=request.avatar_id,
        text=request.text,
        speed=request.speed,
    )

    # 获取当前队列长度作为排队位置
    queue_length = db.get_queue_length()

    logger.info(f"Task {task_id} submitted: avatar={request.avatar_id}, queue_pos={queue_length}")

    return {
        "task_id": task_id,
        "queue_position": queue_length,
    }


@router.get("/status/{task_id}", summary="Task Status")
async def task_status(task_id: str):
    """获取任务状态

    馨语兼容的 status 值: queued/processing/completed/failed
    见 馨语调用全套说明.md §3.5, §8
    """
    task = db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return _task_to_flat_response(task)


@router.get("/tasks", summary="Get All Tasks")
async def get_all_tasks():
    """获取所有任务列表"""
    tasks = db.list_tasks()
    return [_task_to_flat_response(t) for t in tasks]


@router.delete("/tasks/{task_id}", summary="Delete Task")
async def delete_task(task_id: str):
    """删除任务（只能删除排队中的任务）"""
    task = db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task["status"] not in ("queued", "failed", "completed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete task with status '{task['status']}'",
        )

    if task.get("video_path"):
        try:
            os.remove(get_generated_video_path(task["video_path"]))
        except OSError:
            pass
    if task.get("audio_path"):
        try:
            os.remove(get_generated_audio_path(task["audio_path"]))
        except OSError:
            pass

    db.delete_task(task_id)
    return {"message": f"Task {task_id} deleted"}


@router.get("/queue", summary="Get Queue Status")
async def get_queue_status():
    """获取队列状态

    见 馨语调用全套说明.md §3.3
    """
    stats = db.get_queue_stats()
    return {
        "queue_length": stats["queue_length"],
        "active_tasks": stats["active_tasks"],
        "completed_tasks": stats["completed_tasks"],
        "failed_tasks": stats["failed_tasks"],
    }


def _task_to_flat_response(task: dict) -> dict:
    """转换为扁平响应（无 data 包装），馨语客户端直接读取顶层字段。"""
    resp = {
        "task_id": task["id"],
        "avatar_id": task["avatar_id"],
        "status": task["status"],
        "progress": task.get("progress", 0),
        "video_url": None,
        "error": task.get("error"),
        "created_at": task.get("created_at", ""),
        "updated_at": task.get("updated_at", ""),
    }

    if task.get("video_path"):
        # 返回相对路径，客户端会自动拼接 BASE_URL
        # 见 馨语调用全套说明.md §3.5
        resp["video_url"] = f"/download/{os.path.basename(task['video_path'])}"

    return resp
