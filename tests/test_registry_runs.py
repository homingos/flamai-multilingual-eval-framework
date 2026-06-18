"""Run creation and listing tests."""


def test_create_run_returns_201(client, registry_write_headers):
    resp = client.post(
        "/runs",
        json={"task_scope": ["translation", "instructions"], "judge_model": "claude-haiku-4-5"},
        headers=registry_write_headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "started"
    assert "run_id" in body
    assert body["data"]["task_scope"] == ["translation", "instructions"]


def test_create_run_generates_unique_ids(client, registry_write_headers):
    payload = {"task_scope": ["translation"], "judge_model": "claude-haiku-4-5"}
    r1 = client.post("/runs", json=payload, headers=registry_write_headers)
    r2 = client.post("/runs", json=payload, headers=registry_write_headers)
    assert r1.json()["run_id"] != r2.json()["run_id"]


def test_list_runs(client, registry_write_headers, registry_read_headers):
    client.post(
        "/runs",
        json={"task_scope": ["translation"], "judge_model": "claude-haiku-4-5"},
        headers=registry_write_headers,
    )
    resp = client.get("/runs", headers=registry_read_headers)
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


def test_get_run(client, registry_write_headers, registry_read_headers):
    create_resp = client.post(
        "/runs",
        json={"task_scope": ["translation"], "judge_model": "claude-haiku-4-5"},
        headers=registry_write_headers,
    )
    run_id = create_resp.json()["run_id"]
    resp = client.get(f"/runs/{run_id}", headers=registry_read_headers)
    assert resp.status_code == 200
    assert resp.json()["run_id"] == run_id


def test_get_run_not_found_returns_404(client, registry_read_headers):
    resp = client.get("/runs/no-such-run", headers=registry_read_headers)
    assert resp.status_code == 404


def test_run_manifest_has_expected_fields(client, registry_write_headers):
    resp = client.post(
        "/runs",
        json={"task_scope": ["translation"], "judge_model": "claude-haiku-4-5", "notes": "test"},
        headers=registry_write_headers,
    )
    data = resp.json()["data"]
    assert data["status"] == "started"
    assert "created_at" in data
    assert data["inference_config"]["temperature"] == 0.0
    assert data["judge"]["model"] == "claude-haiku-4-5"
    assert data["notes"] == "test"
