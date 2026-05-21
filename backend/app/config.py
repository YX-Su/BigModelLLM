"""Centralised application settings loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "graphrag-portfolio-assistant"
    app_env: str = "dev"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Mock mode — run the whole pipeline in-memory with no external services
    # (no Milvus / Elasticsearch / Neo4j / Redis, no LLM API key required).
    mock_mode: bool = False

    # LLM
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-replace-me"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_timeout: int = 60

    # Embedding
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = "sk-replace-me"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_chunk_collection: str = "portfolio_chunks"
    milvus_entity_collection: str = "portfolio_entities"

    # Elasticsearch
    es_host: str = "localhost"
    es_port: int = 9200
    es_chunk_index: str = "portfolio_chunks"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag123"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    session_ttl: int = 3600

    # Retrieval tuning
    entity_top_k: int = 5
    graph_hop: int = 2
    chunk_top_k: int = 8
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    similarity_threshold: float = 0.35

    @property
    def es_url(self) -> str:
        return f"http://{self.es_host}:{self.es_port}"

    @property
    def raw_data_dir(self) -> Path:
        return DATA_DIR / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return DATA_DIR / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
