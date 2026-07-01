"""Text generation endpoint (LLM integration)."""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_DEFAULT_MODEL
from app.models import GenerateTextRequest

router = APIRouter(prefix="/api", tags=["Text Generation"])


def _build_headers(api_key: str) -> dict:
    """Build headers for OpenAI-compatible API calls."""
    key = api_key or LLM_API_KEY
    if not key:
        raise HTTPException(status_code=400, detail="API key is required")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


@router.post("/generate_text", summary="Generate Text")
async def generate_text(request: GenerateTextRequest):
    """Generate text using LLM (OpenAI-compatible API).

    Uses the configured LLM backend to generate text from a prompt.
    """
    import httpx

    model_id = request.model_id or LLM_DEFAULT_MODEL

    headers = _build_headers(request.api_key)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "data": {
                    "text": content,
                    "model": model_id,
                }
            }
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="LLM request timed out")
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"LLM API error: {e.response.text}",
            )
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_text_stream", summary="Generate Text Stream")
async def generate_text_stream(request: GenerateTextRequest):
    """Generate text using LLM with streaming response (SSE)."""
    import httpx

    model_id = request.model_id or LLM_DEFAULT_MODEL

    headers = _build_headers(request.api_key)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": True,
    }

    async def stream_generator():
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{LLM_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield f"data: {json.dumps({'text': content, 'model': model_id})}\n\n"
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
