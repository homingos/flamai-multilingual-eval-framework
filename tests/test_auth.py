"""Tests for JWT authentication and scope enforcement."""
from tests.conftest import make_token


def test_missing_token_returns_403(client):
    resp = client.get("/models")
    assert resp.status_code == 401


def test_invalid_token_returns_401(client):
    resp = client.get("/models", headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 401


def test_valid_read_token_allows_get(client, registry_read_headers):
    resp = client.get("/models", headers=registry_read_headers)
    assert resp.status_code == 200


def test_read_token_cannot_post(client, registry_read_headers):
    resp = client.post(
        "/models",
        json={
            "id": "m", "name": "M", "hf_model_id": "org/m",
            "language": "Tamil", "slug": "tamil", "region": "Indic",
            "gpu_preset": "l4", "params_billions": 7.0,
        },
        headers=registry_read_headers,
    )
    assert resp.status_code == 403


def test_write_token_allows_post(client, registry_write_headers):
    resp = client.post(
        "/models",
        json={
            "id": "m", "name": "M", "hf_model_id": "org/m",
            "language": "Tamil", "slug": "tamil", "region": "Indic",
            "gpu_preset": "l4", "params_billions": 7.0,
        },
        headers=registry_write_headers,
    )
    assert resp.status_code == 201


def test_expired_token_returns_401(client):
    import datetime
    import jwt as pyjwt

    past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
    token = pyjwt.encode(
        {"sub": "u", "scopes": ["registry:read"], "exp": past},
        "test-secret-for-pytest",
        algorithm="HS256",
    )
    resp = client.get("/models", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
