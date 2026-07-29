def test_health_endpoint(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ENVIRONMENT", "test")

    from fastapi.testclient import TestClient

    from app.api.main import create_app
    from app.core.config import get_settings

    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "stock-configurator",
        "environment": "test",
    }
