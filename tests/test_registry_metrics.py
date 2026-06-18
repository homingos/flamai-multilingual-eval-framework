"""Metric registry endpoint tests."""

_BLEU = {
    "name": "bleu",
    "stage": "post_inference",
    "compute_tier": "light",
    "task_types": ["translation"],
    "category": None,
    "weight": 1.0,
}


def test_create_metric_returns_201(client, registry_write_headers):
    resp = client.post("/metrics", json=_BLEU, headers=registry_write_headers)
    assert resp.status_code == 201
    assert resp.json()["data"]["name"] == "bleu"


def test_create_metric_duplicate_returns_409(client, registry_write_headers):
    client.post("/metrics", json=_BLEU, headers=registry_write_headers)
    resp = client.post("/metrics", json=_BLEU, headers=registry_write_headers)
    assert resp.status_code == 409


def test_list_metrics_returns_all(client, registry_write_headers, registry_read_headers):
    client.post("/metrics", json=_BLEU, headers=registry_write_headers)
    resp = client.get("/metrics", headers=registry_read_headers)
    assert resp.json()["total"] == 1


def test_list_metrics_enabled_only(client, registry_write_headers, registry_read_headers):
    client.post("/metrics", json=_BLEU, headers=registry_write_headers)
    disabled = {**_BLEU, "name": "chrf", "enabled": False}
    client.post("/metrics", json=disabled, headers=registry_write_headers)

    resp = client.get("/metrics?enabled_only=true", headers=registry_read_headers)
    assert resp.json()["total"] == 1
    assert resp.json()["metrics"][0]["name"] == "bleu"


def test_get_metric(client, registry_write_headers, registry_read_headers):
    client.post("/metrics", json=_BLEU, headers=registry_write_headers)
    resp = client.get("/metrics/bleu", headers=registry_read_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["compute_tier"] == "light"


def test_get_metric_not_found_returns_404(client, registry_read_headers):
    resp = client.get("/metrics/no-such-metric", headers=registry_read_headers)
    assert resp.status_code == 404


def test_update_metric_toggle_enabled(client, registry_write_headers):
    client.post("/metrics", json=_BLEU, headers=registry_write_headers)
    resp = client.patch("/metrics/bleu", json={"enabled": False}, headers=registry_write_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["enabled"] is False


def test_update_metric_weight(client, registry_write_headers):
    client.post("/metrics", json=_BLEU, headers=registry_write_headers)
    resp = client.patch("/metrics/bleu", json={"weight": 2.5}, headers=registry_write_headers)
    assert resp.json()["data"]["weight"] == 2.5
