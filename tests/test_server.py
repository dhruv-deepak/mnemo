import json
import logging

import pytest
from fastapi.testclient import TestClient

from mnemo import Memory, Settings
from mnemo.server import create_app
from mnemo.server.logs import JsonFormatter
from mnemo.server.ratelimit import TokenBucketLimiter


@pytest.fixture
def client(memory):
    with TestClient(create_app(memory)) as c:
        yield c


def _secured_client(memory, **overrides) -> TestClient:
    settings = memory.settings.model_copy(update=overrides)
    return TestClient(create_app(memory, settings))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "x-request-id" in response.headers


def test_add_search_update_delete_flow(client, fake_llm):
    fake_llm.extraction.append({"facts": [{"content": "Priya is vegetarian", "importance": 0.9}]})
    added = client.post(
        "/v1/memories",
        json={
            "messages": [{"role": "user", "content": "I'm vegetarian", "name": "Priya"}],
            "user_id": "priya",
            "timestamp": "2025-05-03T10:00:00Z",
        },
    )
    assert added.status_code == 200, added.text
    event = added.json()["events"][0]
    assert event["event"] == "ADD"
    memory_id = event["memory_id"]

    found = client.post("/v1/memories/search", json={"query": "vegetarian", "user_id": "priya"})
    assert found.status_code == 200
    top = found.json()["results"][0]
    assert top["memory"]["content"] == "Priya is vegetarian"
    assert top["memory"]["created_at"].startswith("2025-05-03")
    assert "rrf" in top["breakdown"]

    listed = client.get("/v1/memories", params={"user_id": "priya"}).json()
    assert listed["total"] == 1

    patched = client.patch(f"/v1/memories/{memory_id}", json={"content": "Priya is vegan"})
    assert patched.json()["content"] == "Priya is vegan"
    history = client.get(f"/v1/memories/{memory_id}/history").json()
    assert [h["event"] for h in history] == ["ADD", "UPDATE"]

    assert client.delete(f"/v1/memories/{memory_id}").status_code == 204
    assert client.get(f"/v1/memories/{memory_id}").status_code == 404
    assert client.delete(f"/v1/memories/{memory_id}").status_code == 404


def test_validation_errors(client):
    assert client.post("/v1/memories", json={"messages": "hi"}).status_code == 422
    bad_limit = client.post("/v1/memories/search", json={"query": "x", "user_id": "u", "limit": 0})
    assert bad_limit.status_code == 422


def test_delete_user_and_graph(client):
    client.post("/v1/memories", json={"messages": "a", "user_id": "u", "infer": False})
    assert client.get("/v1/users/u/graph").json() == {"nodes": [], "edges": []}
    assert client.delete("/v1/users/u").json() == {"deleted": 1}


def test_llm_failure_returns_502(client, fake_llm):
    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    fake_llm.complete_json = boom
    response = client.post("/v1/memories", json={"messages": "hello", "user_id": "u"})
    assert response.status_code == 502
    assert "provider down" in response.json()["detail"]


def test_api_key_auth(memory):
    with _secured_client(memory, api_keys="k1,k2") as c:
        assert c.get("/health").status_code == 200  # health stays open for load balancers
        assert c.get("/v1/memories", params={"user_id": "u"}).status_code == 401
        wrong = c.get("/v1/memories", params={"user_id": "u"}, headers={"x-api-key": "nope"})
        assert wrong.status_code == 401
        ok = c.get("/v1/memories", params={"user_id": "u"}, headers={"x-api-key": "k2"})
        assert ok.status_code == 200
        bearer = c.get(
            "/v1/memories", params={"user_id": "u"}, headers={"authorization": "Bearer k1"}
        )
        assert bearer.status_code == 200


def test_rate_limit_returns_429_with_retry_after(memory):
    with _secured_client(memory, rate_limit_per_minute=2) as c:
        codes = [c.get("/v1/memories", params={"user_id": "u"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        limited = c.get("/v1/memories", params={"user_id": "u"})
        assert int(limited.headers["retry-after"]) >= 1


def test_metrics_endpoint(client):
    client.get("/health")
    body = client.get("/metrics").text
    assert 'mnemo_http_requests_total{method="GET",route="/health",status="200"}' in body
    assert "mnemo_http_request_duration_seconds_bucket" in body


def test_token_bucket_refills_over_time():
    now = [0.0]
    limiter = TokenBucketLimiter(per_minute=60, clock=lambda: now[0])  # 1 token / second
    assert all(limiter.acquire("a") == 0 for _ in range(60))
    wait = limiter.acquire("a")
    assert wait == pytest.approx(1.0)
    assert limiter.acquire("b") == 0  # buckets are per key
    now[0] += 1.0
    assert limiter.acquire("a") == 0


def test_token_bucket_evicts_idle_keys():
    now = [0.0]
    limiter = TokenBucketLimiter(per_minute=60, clock=lambda: now[0], max_keys=2)
    limiter.acquire("a")
    limiter.acquire("b")
    now[0] += 120
    limiter.acquire("c")
    assert set(limiter._buckets) == {"c"}


def test_json_log_formatter_includes_extra_fields():
    record = logging.LogRecord("mnemo.api", logging.INFO, __file__, 1, "request", (), None)
    record.status = 200
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "request"
    assert payload["status"] == 200
    assert payload["level"] == "INFO"


def test_app_builds_its_own_memory_from_settings(tmp_path):
    settings = Settings(_env_file=None, embedder="hash", sqlite_path=str(tmp_path / "x.db"))
    with TestClient(create_app(settings=settings)) as c:
        c.post("/v1/memories", json={"messages": "note", "user_id": "u", "infer": False})
        assert c.get("/v1/memories", params={"user_id": "u"}).json()["total"] == 1
    assert isinstance(c.app.state.memory, Memory)
