"""mnemo: long-term memory for LLM agents."""

from mnemo.config import Settings
from mnemo.memory import Memory, format_memories
from mnemo.types import (
    AddResult,
    EventType,
    MemoryEvent,
    MemoryItem,
    MemoryType,
    SearchResult,
)

__version__ = "0.1.0"

__all__ = [
    "AddResult",
    "EventType",
    "Memory",
    "MemoryEvent",
    "MemoryItem",
    "MemoryType",
    "SearchResult",
    "Settings",
    "__version__",
    "format_memories",
]
