"""
Shared fixtures for the Phase 2A test suite.

Every test that touches the registry gets a fresh InMemoryRegistryStore
via the fresh_registry_service fixture (autouse=True).
PERSISTENCE_BACKEND is forced to 'memory' so _get_store() is never called.
"""
from __future__ import annotations

import datetime
import os

import jwt
import pytest
from fastapi.testclient import TestClient

# Force memory backend before any src imports resolve the env var
os.environ["PERSISTENCE_BACKEND"] = "memory"
os.environ["JWT_SECRET"] = "test-secret-for-pytest-at-least-32-bytes"

from src.deps import get_registry_service  # noqa: E402
from src.main import app  # noqa: E402

_JWT_SECRET = "test-secret-for-pytest-at-least-32-bytes"
_JWT_ALG = "HS256"


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def make_token(scopes: list[str], subject: str = "test-user", expiry_minutes: int = 60) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": subject,
        "scopes": scopes,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=expiry_minutes),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALG)


# ---------------------------------------------------------------------------
# Registry service fixture — isolates each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_registry_service():
    """Each test gets an isolated in-memory store."""
    from src.adapters.persistence.memory_store import InMemoryRegistryStore
    from src.core.services.registry_service import RegistryService

    service = RegistryService(InMemoryRegistryStore())
    app.dependency_overrides[get_registry_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_registry_service, None)


# ---------------------------------------------------------------------------
# HTTP clients
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def registry_write_token() -> str:
    return make_token(scopes=["registry:read", "registry:write"])


@pytest.fixture
def registry_read_token() -> str:
    return make_token(scopes=["registry:read"])


@pytest.fixture
def registry_write_headers(registry_write_token: str) -> dict:
    return {"Authorization": f"Bearer {registry_write_token}"}


@pytest.fixture
def registry_read_headers(registry_read_token: str) -> dict:
    return {"Authorization": f"Bearer {registry_read_token}"}


@pytest.fixture
def auth_client(client: TestClient, registry_write_headers: dict) -> TestClient:
    """TestClient pre-configured with write-scope auth headers."""
    client.headers.update(registry_write_headers)
    return client


# ---------------------------------------------------------------------------
# Minimal model payload helper
# ---------------------------------------------------------------------------

def minimal_model_payload(
    model_id: str = "test-model-7b",
    **overrides,
) -> dict:
    base = {
        "id": model_id,
        "name": "Test Model 7B",
        "hf_model_id": "test-org/test-model-7b",
        "language": "Tamil",
        "slug": "tamil",
        "region": "Indic",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    }
    base.update(overrides)
    return base
