"""Thin client over any OpenAI-compatible chat API, with JSON-mode helpers and token accounting."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Protocol

from mnemo.config import Settings
from mnemo.types import Usage

log = logging.getLogger(__name__)


class LLM(Protocol):
    usage: Usage

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 1024) -> str: ...

    def complete_json(
        self, system: str, user: str, *, max_tokens: int = 2048
    ) -> dict[str, Any]: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and leading chatter."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")
    return value


def _make_client(base_url: str, api_key: str, settings: Settings) -> Any:
    kwargs = {
        "base_url": base_url,
        "api_key": api_key or "missing-key",
        "max_retries": settings.llm_max_retries,  # SDK retries 429/5xx with backoff
        "timeout": settings.llm_timeout_s,
    }
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.openai import OpenAI as TracedOpenAI  # drop-in, traces every call

            log.info("Langfuse tracing enabled")
            return TracedOpenAI(**kwargs)
        except ImportError:
            log.warning("LANGFUSE_* is set but langfuse is not installed; tracing disabled")
    from openai import OpenAI

    return OpenAI(**kwargs)


class OpenAICompatLLM:
    def __init__(self, settings: Settings) -> None:
        base_url, api_key, model = settings.resolved_llm()
        self.model = model
        self.temperature = settings.llm_temperature
        self._client = _make_client(base_url, api_key, settings)
        self._lock = threading.Lock()
        self.usage = Usage()

    def _record(self, response: Any) -> None:
        u = getattr(response, "usage", None)
        with self._lock:
            self.usage.llm_calls += 1
            if u is not None:
                self.usage.prompt_tokens += u.prompt_tokens or 0
                self.usage.completion_tokens += u.completion_tokens or 0

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        self._record(response)
        return response.choices[0].message.content or ""

    def complete_json(self, system: str, user: str, *, max_tokens: int = 2048) -> dict[str, Any]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Exception | None = None
        for _ in range(2):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            self._record(response)
            text = response.choices[0].message.content or ""
            try:
                return parse_json_object(text)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                log.warning("LLM returned invalid JSON, retrying: %s", exc)
        raise ValueError(f"LLM did not return valid JSON: {last_error}")
