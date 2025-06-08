import pytest
from litestar.testing import TestClient

from api.backend import app

client = TestClient(app)

class DummySession:
    def __init__(self, id=1):
        self.id = id
        self.dataset = "Iris"
        self.gates = ["rx", "cz"]
        self.discount_rate = 0.9
        self.learning_rate = 0.001
        self.max_architecture_length = 4
        self.autoencoder_path = None


@pytest.mark.asyncio
def test_start_training(monkeypatch):
    monkeypatch.setattr("api.routes.subprocess.Popen", lambda *args, **kwargs: None)

    class DummyRepo:
        def create(self, *args, **kwargs):
            return DummySession(id=1)

    monkeypatch.setattr("api.routes.TrainingSessionRepository", lambda db: DummyRepo())

    response = client.post("/sessions/", json={
        "dataset": "Iris",
        "gates": ["rx", "cz"],
        "discount": 0.9,
        "lr": 0.001,
        "max_length": 4
    })

    assert response.status_code == 201
    assert response.json() == {"status": "started", "training_id": 1}


@pytest.mark.asyncio
def test_list_sessions(monkeypatch):
    dummy_sessions = [DummySession(id=1), DummySession(id=2)]

    class DummyRepo:
        def get_all(self):
            return dummy_sessions

    monkeypatch.setattr("api.routes.TrainingSessionDTO.model_validate", lambda s: {"id": s.id})
    monkeypatch.setattr("api.routes.TrainingSessionRepository", lambda db: DummyRepo())

    response = client.get("/sessions/")
    assert response.status_code == 200

    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 2
    assert all("id" in s for s in result)


@pytest.mark.asyncio
def test_get_session_by_id(monkeypatch):
    class DummyRepo:
        def get_by_id(self, session_id):
            if session_id == 1:
                return DummySession(id=1)
            return None

    monkeypatch.setattr("api.routes.TrainingSessionDTO.model_validate", lambda s: {"id": s.id})
    monkeypatch.setattr("api.routes.TrainingSessionRepository", lambda db: DummyRepo())

    # Valid session
    response = client.get("/sessions/1")
    assert response.status_code == 200
    assert response.json()["id"] == 1

    # Invalid session
    response = client.get("/sessions/999")
    assert response.status_code == 404
