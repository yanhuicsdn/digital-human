"""Root index endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health", summary="Health Check", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "service": "数字人视频生成服务 (Digital Human Video Generation Service)",
        "version": "0.1.0",
        "backend": "SoulX-FlashHead + LongCat-AudioDiT",
        "status": "running",
    }
