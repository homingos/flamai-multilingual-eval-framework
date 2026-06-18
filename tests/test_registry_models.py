"""
CRUD + lifecycle tests for the /models registry endpoints.
"""
import pytest
from tests.conftest import minimal_model_payload


# ---------------------------------------------------------------------------
# POST /models
# ---------------------------------------------------------------------------

def test_create_model_returns_201(client, registry_write_headers):
    resp = client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    assert resp.status_code == 201
    assert resp.json()["status"] == "created"


def test_create_model_state_is_active(client, registry_write_headers):
    resp = client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    assert resp.json()["data"]["state"] == "active"


def test_create_model_duplicate_id_returns_409(client, registry_write_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    resp = client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    assert resp.status_code == 409


def test_create_model_missing_required_field_returns_422(client, registry_write_headers):
    payload = minimal_model_payload()
    del payload["hf_model_id"]
    resp = client.post("/models", json=payload, headers=registry_write_headers)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

def test_list_models_returns_active_by_default(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload("m1"), headers=registry_write_headers)
    resp = client.get("/models", headers=registry_read_headers)
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


def test_list_models_default_excludes_disabled(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload("m1"), headers=registry_write_headers)
    client.patch("/models/m1/disable", headers=registry_write_headers)
    resp = client.get("/models", headers=registry_read_headers)
    assert resp.json()["total"] == 0


def test_list_models_all_includes_disabled(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload("m1"), headers=registry_write_headers)
    client.patch("/models/m1/disable", headers=registry_write_headers)
    resp = client.get("/models?state=all", headers=registry_read_headers)
    assert resp.json()["total"] == 1


def test_list_models_all_includes_deprecated(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload("m1"), headers=registry_write_headers)
    client.patch("/models/m1/deprecate", headers=registry_write_headers)
    resp = client.get("/models?state=all", headers=registry_read_headers)
    assert resp.json()["total"] == 1


def test_list_models_filter_by_state(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload("m1"), headers=registry_write_headers)
    client.post("/models", json=minimal_model_payload("m2"), headers=registry_write_headers)
    client.patch("/models/m1/disable", headers=registry_write_headers)

    resp = client.get("/models?state=disabled", headers=registry_read_headers)
    assert resp.json()["total"] == 1
    assert resp.json()["models"][0]["id"] == "m1"


# ---------------------------------------------------------------------------
# GET /models/{id}
# ---------------------------------------------------------------------------

def test_get_model_returns_200(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    resp = client.get("/models/test-model-7b", headers=registry_read_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == "test-model-7b"


def test_get_model_not_found_returns_404(client, registry_read_headers):
    resp = client.get("/models/does-not-exist", headers=registry_read_headers)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /models/{id}
# ---------------------------------------------------------------------------

def test_update_model_mutable_fields(client, registry_write_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    resp = client.patch(
        "/models/test-model-7b",
        json={"notes": "updated note", "max_model_len": 4096},
        headers=registry_write_headers,
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["notes"] == "updated note"
    assert data["max_model_len"] == 4096


def test_update_model_state_field_rejected(client, registry_write_headers):
    """state is not an updatable field via PATCH /models/{id}."""
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    resp = client.patch(
        "/models/test-model-7b",
        json={"state": "disabled"},  # should be silently ignored
        headers=registry_write_headers,
    )
    # Still 200, but state must remain active
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["state"] == "active"


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def test_disable_active_model(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    resp = client.patch("/models/test-model-7b/disable", headers=registry_write_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["state"] == "disabled"


def test_enable_disabled_model(client, registry_write_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    client.patch("/models/test-model-7b/disable", headers=registry_write_headers)
    resp = client.patch("/models/test-model-7b/enable", headers=registry_write_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["state"] == "active"


def test_deprecate_active_model(client, registry_write_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    resp = client.patch("/models/test-model-7b/deprecate", headers=registry_write_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["state"] == "deprecated"


def test_deprecate_disabled_model(client, registry_write_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    client.patch("/models/test-model-7b/disable", headers=registry_write_headers)
    resp = client.patch("/models/test-model-7b/deprecate", headers=registry_write_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["state"] == "deprecated"


def test_disable_deprecated_model_returns_409(client, registry_write_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    client.patch("/models/test-model-7b/deprecate", headers=registry_write_headers)
    resp = client.patch("/models/test-model-7b/disable", headers=registry_write_headers)
    assert resp.status_code == 409


def test_deprecated_model_excluded_from_default_list(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    client.patch("/models/test-model-7b/deprecate", headers=registry_write_headers)
    resp = client.get("/models", headers=registry_read_headers)
    assert resp.json()["total"] == 0


def test_disabled_model_excluded_from_default_list(client, registry_write_headers, registry_read_headers):
    client.post("/models", json=minimal_model_payload(), headers=registry_write_headers)
    client.patch("/models/test-model-7b/disable", headers=registry_write_headers)
    resp = client.get("/models", headers=registry_read_headers)
    assert resp.json()["total"] == 0
