from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import requires_models

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "online"


def test_root_is_health_alias():
    assert client.get("/").json() == client.get("/health").json()


def test_classify_rejects_empty_text():
    resp = client.post("/classify", json={"text": ""})
    assert resp.status_code == 422


@requires_models
def test_classify_escalates_on_urgency_language():
    with TestClient(app) as c:
        resp = c.post(
            "/classify",
            json={"text": "live rat in house urgent", "day_of_week": 1, "month": 6},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_priority"] == "HIGH"
    assert body["urgency_escalated"] is True
    assert "urgent" in body["urgency_terms"]
    assert body["department"]
    assert isinstance(body["similar_requests"], list)


@requires_models
def test_classify_routine_request():
    with TestClient(app) as c:
        resp = c.post("/classify", json={"text": "recycling bag not collected"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_priority"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["urgency_escalated"] is False
