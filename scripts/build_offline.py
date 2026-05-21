#!/usr/bin/env python3
"""CLI entry point for the GraphRAG offline preparation pipeline.

Usage:
    python scripts/build_offline.py

Requires the storage stack (Milvus / Elasticsearch / Neo4j) to be running and
backend/.env to be configured with a working OpenAI-compatible endpoint.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings  # noqa: E402
from app.rag.offline import build_all  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    build_all()
