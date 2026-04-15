from fastapi.testclient import TestClient

from app.main import app


def test_root():
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_ask_empty():
    c = TestClient(app)
    r = c.get("/ask", params={"q": ""})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data

