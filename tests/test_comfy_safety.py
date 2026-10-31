"""ComfyUI stays disconnected unless the settings toggle and confirm are both set."""

from pathlib import Path

from app.comfy import HttpxTransport

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_HOST = "192.168.4.47"


def test_python_sources_do_not_bake_in_the_example_host():
    roots = [REPO / "app", REPO / "scripts"]
    for root in roots:
        for path in root.rglob("*.py"):
            assert EXAMPLE_HOST not in path.read_text(encoding="utf-8")


def test_settings_default_off_and_host_blank(client):
    payload = client.get("/api/settings").json()
    assert payload == {"comfyui_enabled": False, "comfyui_host": ""}
    health = client.get("/api/health").json()
    assert health["comfyui_enabled"] is False
    assert health["bind"] == "127.0.0.1"


def test_default_routes_never_call_http(monkeypatch, client, app):
    def boom(self, method, url, json=None):
        raise AssertionError(f"unexpected HTTP {method} {url}")

    monkeypatch.setattr(HttpxTransport, "request", boom)
    app.state.transport = HttpxTransport()

    assert client.get("/").status_code == 200
    assert client.get("/api/dashboard").status_code == 200
    assert client.get("/api/epochs/grid").status_code == 200
    assert client.get("/api/variance").status_code == 200
    assert client.get("/api/bleed").status_code == 200
    assert client.get("/api/provenance").status_code == 200
    assert client.get("/api/profiler").status_code == 200
    assert client.post("/api/epochs/queue").status_code == 200
    assert client.post("/api/epochs/submit", json={"confirm": False}).status_code == 403
    assert client.post("/api/epochs/submit", json={"confirm": True}).status_code == 403
    assert client.post("/api/provenance/fetch-history", json={"confirm": True, "user_initiated": True}).status_code == 403
    assert client.post("/api/profiler/fetch-history", json={"confirm": True, "user_initiated": True}).status_code == 403
    assert client.post("/api/index/scan", json={"folder": "sample_data/loras"}).status_code == 200
    assert client.get("/api/shelf", params={"today": "2026-10-31"}).status_code == 200

    saved = client.post("/api/settings", json={"comfyui_enabled": True, "comfyui_host": ""}).json()
    assert saved["comfyui_enabled"] is True
    assert saved["comfyui_host"] == ""
    blocked = client.post("/api/epochs/submit", json={"confirm": True})
    assert blocked.status_code == 403
    assert "blank" in blocked.json()["detail"].lower() or "off" in blocked.json()["detail"].lower() or "confirm" in blocked.json()["detail"].lower()


def test_http_only_after_toggle_and_confirm(client, app):
    client.post("/api/epochs/queue")
    assert app.state.transport.calls == []

    client.post("/api/settings", json={"comfyui_enabled": True, "comfyui_host": ""})
    assert client.post("/api/epochs/submit", json={"confirm": True}).status_code == 403
    assert app.state.transport.calls == []

    client.post("/api/settings", json={"comfyui_enabled": False, "comfyui_host": "127.0.0.1:9"})
    assert client.post("/api/epochs/submit", json={"confirm": True}).status_code == 403
    assert client.post("/api/epochs/submit", json={"confirm": False}).status_code == 403
    assert app.state.transport.calls == []

    client.post("/api/settings", json={"comfyui_enabled": True, "comfyui_host": "127.0.0.1:9"})
    denied = client.post("/api/provenance/fetch-history", json={"confirm": True, "user_initiated": False})
    assert denied.status_code == 403
    assert app.state.transport.calls == []

    allowed = client.post("/api/epochs/submit", json={"confirm": True})
    assert allowed.status_code == 200
    assert allowed.json()["count"] > 0
    assert app.state.transport.calls
    assert {call["method"] for call in app.state.transport.calls} == {"POST"}
    assert all(call["url"].startswith("http://127.0.0.1:9/prompt") for call in app.state.transport.calls)
    assert all(EXAMPLE_HOST not in call["url"] for call in app.state.transport.calls)

    app.state.transport.calls.clear()
    history = client.post(
        "/api/profiler/fetch-history",
        json={"confirm": True, "user_initiated": True},
    )
    assert history.status_code == 200
    assert history.json()["method"] == "GET"
    assert history.json()["submitted"] is False
    assert len(app.state.transport.calls) == 1
    assert app.state.transport.calls[0]["method"] == "GET"
    assert app.state.transport.calls[0]["url"] == "http://127.0.0.1:9/history"


def test_confirm_string_is_not_enough(client, app):
    client.post("/api/epochs/queue")
    client.post("/api/settings", json={"comfyui_enabled": True, "comfyui_host": "127.0.0.1:9"})
    response = client.post("/api/epochs/submit", json={"confirm": "true"})
    assert response.status_code == 403
    assert app.state.transport.calls == []


def test_ui_warns_before_gpu_submit():
    script = (REPO / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "Zach's PC" in script
    assert "that PC's GPU" in script
    assert "Nothing is sent until you confirm." in script
    page = (REPO / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "checked" not in page
