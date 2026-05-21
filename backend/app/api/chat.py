"""Chat endpoint — the entry point for the online inference pipeline."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.pipeline import run_pipeline
from app.schemas.api import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a user query within a multi-turn session."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    try:
        return run_pipeline(request.session_id, request.query)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        logger.exception("pipeline failed")
        raise HTTPException(status_code=500, detail=f"推理链路执行失败: {exc}") from exc
