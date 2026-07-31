from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.database import get_session


class HealthySession:
    async def execute(self, statement: object) -> None:
        del statement


class UnavailableSession:
    async def execute(self, statement: object) -> None:
        del statement
        raise RuntimeError("database unavailable")


def _client(
    monkeypatch,
    session: HealthySession | UnavailableSession,
) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ENVIRONMENT", "test")
    get_settings.cache_clear()

    from app.api.main import create_app

    app = create_app()

    async def override_get_session() -> AsyncGenerator[HealthySession | UnavailableSession, None]:
        yield session

    app.dependency_overrides[get_session] = override_get_session
    return TestClient(app)


def test_health_endpoint_is_a_database_independent_liveness_probe(monkeypatch) -> None:
    with _client(monkeypatch, UnavailableSession()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "stock-configurator",
        "environment": "test",
    }


def test_readiness_endpoint_checks_database(monkeypatch) -> None:
    with _client(monkeypatch, HealthySession()) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "stock-configurator",
        "environment": "test",
        "database": "ok",
    }


def test_readiness_endpoint_fails_closed_without_leaking_database_details(monkeypatch) -> None:
    with _client(monkeypatch, UnavailableSession()) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service database is not ready."}
