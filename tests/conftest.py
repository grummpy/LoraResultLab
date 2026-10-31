import pytest
from fastapi.testclient import TestClient

from app.comfy import RecordingTransport
from app.main import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("LORALAB_CLIP_MODEL", raising=False)
    application = create_app(data_dir=tmp_path / "data")
    application.state.transport = RecordingTransport()
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
