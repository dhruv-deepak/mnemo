from mnemo.config import Settings
from mnemo.store.base import Filters, MemoryStore
from mnemo.store.sqlite import SQLiteStore

__all__ = ["Filters", "MemoryStore", "SQLiteStore", "build_store"]


def build_store(settings: Settings, dim: int) -> MemoryStore:
    if settings.store == "postgres":
        if not settings.postgres_url:
            raise ValueError("MNEMO_POSTGRES_URL is required when MNEMO_STORE=postgres")
        from mnemo.store.postgres import PostgresStore  # optional dependency

        return PostgresStore(settings.postgres_url, dim=dim)
    return SQLiteStore(settings.sqlite_path)
