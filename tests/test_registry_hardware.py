"""Hardware registry endpoint tests."""

_HW = {"id": "l4", "gpu": "L4", "vram_gb": 24, "cost_per_hr": 0.80, "use_for": "7B inference"}


def test_create_hardware_returns_201(client, registry_write_headers):
    resp = client.post("/hardware", json=_HW, headers=registry_write_headers)
    assert resp.status_code == 201
    assert resp.json()["data"]["id"] == "l4"


def test_create_hardware_duplicate_returns_409(client, registry_write_headers):
    client.post("/hardware", json=_HW, headers=registry_write_headers)
    resp = client.post("/hardware", json=_HW, headers=registry_write_headers)
    assert resp.status_code == 409


def test_list_hardware(client, registry_write_headers, registry_read_headers):
    client.post("/hardware", json=_HW, headers=registry_write_headers)
    resp = client.get("/hardware", headers=registry_read_headers)
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


def test_get_hardware(client, registry_write_headers, registry_read_headers):
    client.post("/hardware", json=_HW, headers=registry_write_headers)
    resp = client.get("/hardware/l4", headers=registry_read_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["gpu"] == "L4"


def test_get_hardware_not_found_returns_404(client, registry_read_headers):
    resp = client.get("/hardware/does-not-exist", headers=registry_read_headers)
    assert resp.status_code == 404


def test_update_hardware(client, registry_write_headers):
    client.post("/hardware", json=_HW, headers=registry_write_headers)
    resp = client.patch(
        "/hardware/l4",
        json={"cost_per_hr": 0.85, "use_for": "updated use"},
        headers=registry_write_headers,
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["cost_per_hr"] == 0.85
    assert data["use_for"] == "updated use"
