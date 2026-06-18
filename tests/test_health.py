def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_health_no_auth_required(client):
    """Health endpoint must work without any Authorization header."""
    resp = client.get("/health")
    assert resp.status_code == 200
