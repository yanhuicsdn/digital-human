"""Main FastAPI application — Digital Human Video Generation Service.

Integrates:
  - SoulX-FlashHead: Talking-head video generation
  - LongCat-AudioDiT: Voice cloning / TTS
  - LLM API: Text generation
"""

from __future__ import annotations

import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

# Add project root to sys.path so SoulX-FlashHead can be imported
# (flash_head package should be at the same level or via PYTHONPATH)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import HOST, PORT
from app.routers import avatars, download, index, tasks, text_generation
from app.services.task_queue import task_queue

# Suppress warnings from heavy ML libraries
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — start/stop background worker."""
    logger.info("=" * 60)
    logger.info("数字人视频生成服务 (Digital Human Video Generation)")
    logger.info(f"  FlashHead: SoulX-FlashHead (talking head)")
    logger.info(f"  TTS:       LongCat-AudioDiT (voice cloning)")
    logger.info(f"  Host:      {HOST}:{PORT}")
    logger.info("=" * 60)

    # Start background task queue worker
    task_queue.start()
    yield
    # Shutdown
    task_queue.stop()
    logger.info("Service shutdown complete")


app = FastAPI(
    title="数字人视频生成服务",
    version="0.1.0",
    description="""
    # 数字人视频生成服务 API
    
    基于 **SoulX-FlashHead**（数字人视频生成）和 **LongCat-AudioDiT**（语音克隆/TTS）构建。
    
    ## 工作流程
    
    1. **创建数字人** - 上传人物照片 + 语音样本 → POST `/api/avatar/create`
    2. **提交任务** - 输入文本 + 数字人ID → POST `/api/submit`
    3. **查看状态** - 轮询任务进度 → GET `/api/status/{task_id}`
    4. **下载视频** - 任务完成后下载结果 → GET `/download/{filename}`
    
    ## 可选：文本生成
    - POST `/api/generate_text` - 使用LLM生成文本内容
    - POST `/api/generate_text_stream` - 流式文本生成
    """,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS — allow frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(index.router)
app.include_router(avatars.router)
app.include_router(text_generation.router)
app.include_router(tasks.router)
app.include_router(download.router)

# Serve frontend static pages
STATIC_DIR = PROJECT_ROOT / "app" / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/create-avatar", include_in_schema=False)
    async def serve_create_avatar():
        return FileResponse(str(STATIC_DIR / "create-avatar.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
