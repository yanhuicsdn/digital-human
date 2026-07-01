"""Avatar management endpoints.

响应格式严格遵循 馨语数字人调用标准。
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.database import db
from app.storage import (
    delete_file,
    get_avatar_audio_path,
    get_avatar_image_path,
    save_avatar_audio,
    save_avatar_image,
)

router = APIRouter(prefix="/api", tags=["Avatars"])


@router.get("/avatars", summary="List Avatars")
async def list_avatars():
    """获取所有数字人列表

    馨语客户端兼容 avatar_id/id, name/avatar_name/avatarName 等字段名。
    见 馨语调用全套说明.md §4.4
    """
    avatars = db.list_avatars()
    results = []
    for a in avatars:
        results.append(
            {
                "avatar_id": a["id"],
                "id": a["id"],
                "name": a["name"],
                "description": a.get("description", ""),
                "image_url": f"/api/avatar_image/{a['id']}",
                "preview_url": f"/api/avatar_image/{a['id']}",
            }
        )
    return results


@router.get("/avatar_image/{avatar_id}", summary="Get Avatar Image")
async def get_avatar_image(avatar_id: str):
    """获取数字人图像"""
    avatar = db.get_avatar(avatar_id)
    if avatar is None:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")

    image_path = get_avatar_image_path(avatar["image_path"])
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Avatar image file not found")

    return FileResponse(image_path, media_type="image/png")


@router.get("/avatar_audio/{avatar_id}", summary="Get Avatar Audio")
async def get_avatar_audio(avatar_id: str):
    """获取数字人音频"""
    avatar = db.get_avatar(avatar_id)
    if avatar is None:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")

    audio_path = get_avatar_audio_path(avatar["audio_path"])
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Avatar audio file not found")

    return FileResponse(audio_path, media_type="audio/wav")


@router.post("/avatar/create", summary="Create Avatar")
async def create_avatar(
    name: str = Form(..., description="数字人名称"),
    description: str = Form(default="", description="数字人描述"),
    prompt_text: str = Form(default="", description="数字人提示文本（用于语音克隆）"),
    image: UploadFile = File(..., description="数字人图像"),
    audio: UploadFile = File(..., description="数字人语音样本（用于语音克隆）"),
):
    """创建新数字人

    - image: 数字人头部照片（PNG/JPG）
    - audio: 语音样本（WAV格式，用于声音克隆）
    - prompt_text: 语音样本对应的文本内容（可选，用于更好的语音克隆效果）
    """
    avatar_id = uuid.uuid4().hex[:12]

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or ".png")[1]) as tmp_img:
        tmp_img.write(await image.read())
        tmp_img_path = tmp_img.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename or ".wav")[1]) as tmp_aud:
        tmp_aud.write(await audio.read())
        tmp_aud_path = tmp_aud.name

    try:
        image_rel = save_avatar_image(tmp_img_path)
        audio_rel = save_avatar_audio(tmp_aud_path)

        db.create_avatar(
            avatar_id=avatar_id,
            name=name,
            description=description,
            prompt_text=prompt_text,
            image_path=image_rel,
            audio_path=audio_rel,
        )

        return {
            "avatar_id": avatar_id,
            "id": avatar_id,
            "name": name,
            "description": description,
            "image_url": f"/api/avatar_image/{avatar_id}",
        }
    finally:
        os.unlink(tmp_img_path)
        os.unlink(tmp_aud_path)


@router.delete("/avatar/{avatar_id}", summary="Delete Avatar")
async def delete_avatar(avatar_id: str):
    """删除数字人"""
    avatar = db.get_avatar(avatar_id)
    if avatar is None:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")

    delete_file(avatar["image_path"])
    delete_file(avatar["audio_path"])
    db.delete_avatar(avatar_id)

    return {"message": f"Avatar {avatar_id} deleted"}
