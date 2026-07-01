"""File download endpoint."""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import DATA_DIR

router = APIRouter(tags=["Download"])


@router.get("/download/{filename}", summary="Download Video")
async def download_video(filename: str):
    """下载生成的文件（视频或音频）"""
    # Check video directory first, then audio
    possible_dirs = [
        DATA_DIR / "generated_videos" / filename,
        DATA_DIR / "generated_audio" / filename,
    ]

    for filepath in possible_dirs:
        if filepath.exists():
            # Determine media type
            ext = os.path.splitext(filename)[1].lower()
            media_types = {
                ".mp4": "video/mp4",
                ".wav": "audio/wav",
                ".webm": "video/webm",
                ".avi": "video/x-msvideo",
                ".mov": "video/quicktime",
            }
            media_type = media_types.get(ext, "application/octet-stream")
            return FileResponse(str(filepath), media_type=media_type)

    raise HTTPException(status_code=404, detail=f"File {filename} not found")
