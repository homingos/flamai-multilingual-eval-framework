"""Task registry endpoint tests. Tasks are pre-seeded — no POST endpoint."""


def test_list_tasks_returns_all_preseeded(client, registry_read_headers):
    resp = client.get("/tasks", headers=registry_read_headers)
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()["tasks"]}
    assert "translation" in names
    assert "instructions" in names
    assert "summarization" in names


def test_list_tasks_active_only(client, registry_read_headers):
    resp = client.get("/tasks?active_only=true", headers=registry_read_headers)
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()["tasks"]}
    assert "translation" in names
    assert "instructions" in names
    assert "summarization" not in names


def test_get_task(client, registry_read_headers):
    resp = client.get("/tasks/translation", headers=registry_read_headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["name"] == "translation"


def test_get_task_not_found_returns_404(client, registry_read_headers):
    resp = client.get("/tasks/nonexistent-task", headers=registry_read_headers)
    assert resp.status_code == 404


def test_update_task_deactivate(client, registry_write_headers, registry_read_headers):
    resp = client.patch(
        "/tasks/translation",
        json={"active": False},
        headers=registry_write_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["active"] is False

    active_resp = client.get("/tasks?active_only=true", headers=registry_read_headers)
    names = {t["name"] for t in active_resp.json()["tasks"]}
    assert "translation" not in names


def test_update_task_description(client, registry_write_headers):
    resp = client.patch(
        "/tasks/instructions",
        json={"description": "Updated description"},
        headers=registry_write_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["description"] == "Updated description"
