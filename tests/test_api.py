"""HTTP API tests using FastAPI's TestClient."""

from fastapi.testclient import TestClient

from flashindorank.api import app

client = TestClient(app)

PASSAGES = [
    {"id": "a", "text": "The giant panda is a bear species native to China."},
    {"id": "b", "text": "Python is a high-level programming language."},
    {"id": "c", "text": "Pandas eat mostly bamboo in the wild."},
]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models():
    resp = client.get("/models")
    assert resp.status_code == 200
    names = [m["name"] for m in resp.json()]
    assert "ms-marco-TinyBERT-L-2-v2" in names


def test_rerank_endpoint():
    resp = client.post(
        "/rerank",
        json={"query": "what is a panda?", "passages": PASSAGES, "top_k": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["results"][0]["id"] in {"a", "c"}
    assert body["took_ms"] >= 0


def test_rerank_accepts_plain_strings():
    resp = client.post(
        "/rerank",
        json={"query": "panda", "passages": ["a giant panda", "a laptop computer"]},
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


def test_rerank_rejects_bad_model():
    resp = client.post(
        "/rerank",
        json={"query": "x", "passages": ["y"], "model": "nope"},
    )
    assert resp.status_code == 400


def test_cascade_endpoint():
    resp = client.post(
        "/rerank/cascade",
        json={"query": "what is a panda?", "passages": PASSAGES, "prune_to": 2, "top_k": 1},
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
