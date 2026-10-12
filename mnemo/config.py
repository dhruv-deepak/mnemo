"""Runtime configuration, loaded from `MNEMO_*` environment variables or a `.env` file."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# OpenAI-compatible endpoints. Every provider here speaks the same chat-completions protocol,
# so switching providers is a config change, not a code change.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",
        "model": "llama3.1:8b",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MNEMO_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: Literal["groq", "openai", "gemini", "ollama", "custom"] = "groq"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_retries: int = 6
    llm_timeout_s: float = 60.0

    # Embeddings
    embedder: Literal["fastembed", "openai", "hash"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Storage
    store: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: str = "mnemo.db"
    postgres_url: str | None = None

    # Write path
    consolidation_candidates: int = Field(
        5, description="Similar memories shown to the LLM per new fact"
    )

    # Retrieval
    candidate_pool: int = 30
    rrf_k: int = 60
    recency_half_life_days: float = 30.0
    recency_weight: float = 0.1
    importance_weight: float = 0.1
    graph_hops: int = 1
    rerank: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # API server
    api_keys: str = ""
    rate_limit_per_minute: int = 120
    log_level: str = "INFO"

    def resolved_llm(self) -> tuple[str, str, str]:
        """Return ``(base_url, api_key, model)`` after applying provider presets."""
        preset = PROVIDER_PRESETS.get(self.llm_provider, {})
        base_url = self.llm_base_url or preset.get("base_url")
        if not base_url:
            raise ValueError("MNEMO_LLM_BASE_URL is required when MNEMO_LLM_PROVIDER=custom")
        api_key = self.llm_api_key or os.getenv(preset.get("key_env", ""), "")
        if not api_key and self.llm_provider == "ollama":
            api_key = "ollama"  # Ollama ignores the key but the client requires one
        model = self.llm_model or preset.get("model")
        if not model:
            raise ValueError("MNEMO_LLM_MODEL is required when MNEMO_LLM_PROVIDER=custom")
        return base_url, api_key, model

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}
