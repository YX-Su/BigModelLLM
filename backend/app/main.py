"""FastAPI application entry point.

Wires together the GraphRAG online inference pipeline and exposes the chat,
session and health endpoints.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, session
from app.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """On startup, build the in-memory knowledge base when in mock mode."""
    if settings.mock_mode:
        from app.mock.bootstrap import mock_bootstrap

        mock_bootstrap()
        logger.info("mock mode enabled — running fully in-memory")
    yield


app = FastAPI(
    title="GraphRAG Portfolio Configuration Assistant",
    description="基于 GraphRAG 与大模型推理的投资组合智能配置助手",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str | bool]:
    """Liveness probe."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "mock_mode": settings.mock_mode,
    }


app.include_router(chat.router)
app.include_router(session.router)
