"""REST API. Run with ``mnemo serve`` or ``uvicorn mnemo.server.app:create_app --factory``."""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from mnemo import __version__
from mnemo.config import Settings
from mnemo.memory import Memory
from mnemo.server.ratelimit import TokenBucketLimiter
from mnemo.server.schemas import (
    AddRequest,
    DeleteUserResponse,
    MemoryList,
    SearchRequest,
    SearchResponse,
    UpdateRequest,
)
from mnemo.types import AddResult, MemoryEvent, MemoryItem, MemoryType

log = logging.getLogger("mnemo.api")


class Metrics:
    def __init__(self) -> None:
        # A private registry per app keeps tests (which build many apps) independent.
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "mnemo_http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "mnemo_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )
        self.memory_events = Counter(
            "mnemo_memory_events_total",
            "Memory changes applied",
            ["event"],
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "mnemo_llm_tokens_total", "LLM tokens consumed", ["kind"], registry=self.registry
        )


def create_app(memory: Memory | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (memory.settings if memory else Settings())
    metrics = Metrics()
    limiter = TokenBucketLimiter(settings.rate_limit_per_minute)
    api_keys = settings.api_key_set

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.memory = memory or Memory(settings)
        yield
        if memory is None:
            app.state.memory.close()

    app = FastAPI(
        title="mnemo",
        version=__version__,
        description="Long-term memory layer for LLM agents.",
        lifespan=lifespan,
    )

    # --- middleware: request id, timing, metrics, one structured log line ----------------
    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            metrics.requests.labels(request.method, path, str(status_code)).inc()
            metrics.latency.labels(request.method, path).observe(elapsed)
            log.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status": status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )

    # --- dependencies --------------------------------------------------------------------
    def authenticate(request: Request) -> str:
        """Return the caller identity used for rate limiting."""
        supplied = request.headers.get("x-api-key") or ""
        auth = request.headers.get("authorization", "")
        if not supplied and auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
        if api_keys:
            # Constant-time comparison so response timing doesn't leak key prefixes.
            if not any(secrets.compare_digest(supplied, key) for key in api_keys):
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
            return f"key:{supplied}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def rate_limit(caller: str = Depends(authenticate)) -> None:
        wait = limiter.acquire(caller)
        if wait > 0:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "rate limit exceeded",
                headers={"Retry-After": str(max(1, round(wait)))},
            )

    def get_memory(request: Request) -> Memory:
        memory: Memory = request.app.state.memory
        return memory

    guarded = [Depends(rate_limit)]

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # --- ops -----------------------------------------------------------------------------
    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # --- memories ------------------------------------------------------------------------
    @app.post("/v1/memories", response_model=AddResult, tags=["memories"], dependencies=guarded)
    def add_memories(body: AddRequest, mem: Memory = Depends(get_memory)) -> AddResult:
        messages = (
            body.messages
            if isinstance(body.messages, str)
            else [m.model_dump(exclude_none=True) for m in body.messages]
        )
        try:
            result = mem.add(
                messages,
                body.user_id,
                agent_id=body.agent_id,
                metadata=body.metadata,
                infer=body.infer,
                when=body.timestamp,
            )
        except ValueError:
            raise
        except Exception as exc:  # LLM provider errors, timeouts, ...
            log.exception("memory write failed")
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM call failed: {exc}") from exc
        for event in result.events:
            metrics.memory_events.labels(event.event.value).inc()
        metrics.llm_tokens.labels("prompt").inc(result.usage.prompt_tokens)
        metrics.llm_tokens.labels("completion").inc(result.usage.completion_tokens)
        return result

    @app.post(
        "/v1/memories/search",
        response_model=SearchResponse,
        tags=["memories"],
        dependencies=guarded,
    )
    def search_memories(body: SearchRequest, mem: Memory = Depends(get_memory)) -> SearchResponse:
        started = time.perf_counter()
        results = mem.search(
            body.query,
            body.user_id,
            agent_id=body.agent_id,
            types=list(body.types),
            limit=body.limit,
            rerank=body.rerank,
        )
        return SearchResponse(results=results, latency_ms=(time.perf_counter() - started) * 1000)

    @app.get("/v1/memories", response_model=MemoryList, tags=["memories"], dependencies=guarded)
    def list_memories(
        user_id: str,
        agent_id: str | None = None,
        type: list[MemoryType] = Query(default_factory=list),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        mem: Memory = Depends(get_memory),
    ) -> MemoryList:
        items = mem.get_all(
            user_id, agent_id=agent_id, types=list(type), limit=limit, offset=offset
        )
        return MemoryList(memories=items, total=mem.count(user_id, agent_id=agent_id))

    def _require(mem: Memory, memory_id: str) -> MemoryItem:
        item = mem.get(memory_id)
        if item is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "memory not found")
        return item

    @app.get(
        "/v1/memories/{memory_id}",
        response_model=MemoryItem,
        tags=["memories"],
        dependencies=guarded,
    )
    def get_memory_item(memory_id: str, mem: Memory = Depends(get_memory)) -> MemoryItem:
        return _require(mem, memory_id)

    @app.patch(
        "/v1/memories/{memory_id}",
        response_model=MemoryItem,
        tags=["memories"],
        dependencies=guarded,
    )
    def update_memory(
        memory_id: str, body: UpdateRequest, mem: Memory = Depends(get_memory)
    ) -> MemoryItem:
        _require(mem, memory_id)
        updated = mem.update(memory_id, body.content)
        assert updated is not None
        return updated

    @app.delete(
        "/v1/memories/{memory_id}", status_code=204, tags=["memories"], dependencies=guarded
    )
    def delete_memory(memory_id: str, mem: Memory = Depends(get_memory)) -> Response:
        if not mem.delete(memory_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "memory not found")
        return Response(status_code=204)

    @app.get(
        "/v1/memories/{memory_id}/history",
        response_model=list[MemoryEvent],
        tags=["memories"],
        dependencies=guarded,
    )
    def memory_history(memory_id: str, mem: Memory = Depends(get_memory)) -> list[MemoryEvent]:
        history = mem.history(memory_id)
        if not history:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "memory not found")
        return history

    # --- users ---------------------------------------------------------------------------
    @app.get("/v1/users/{user_id}/graph", tags=["users"], dependencies=guarded)
    def user_graph(user_id: str, mem: Memory = Depends(get_memory)) -> dict:
        return mem.graph(user_id)

    @app.delete(
        "/v1/users/{user_id}",
        response_model=DeleteUserResponse,
        tags=["users"],
        dependencies=guarded,
    )
    def delete_user(user_id: str, mem: Memory = Depends(get_memory)) -> DeleteUserResponse:
        return DeleteUserResponse(deleted=mem.delete_all(user_id))

    return app
