from pathlib import Path

from app.paths import ART_DIR

REPO = Path(__file__).resolve().parents[1]


def test_ui_shell_and_assets(client):
    page = client.get("/")
    assert page.status_code == 200
    html = page.text
    for label in (
        "Dashboard",
        "Epoch Picker",
        "Seed Variance",
        "Style Bleed",
        "Merge Recipe",
        "Provenance",
        "Node Profiler",
        "Hash Index",
        "Shelf Life",
        "Settings",
    ):
        assert label in html
    assert "/art/cover.png" in html
    assert "/art/icon.png" in html
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    icon = client.get("/favicon.ico")
    assert icon.status_code == 200
    assert icon.headers["content-type"].startswith("image/png")
    assert icon.content == (ART_DIR / "icon.png").read_bytes()
    cover = client.get("/art/cover.png")
    assert cover.status_code == 200
    assert cover.content == (ART_DIR / "cover.png").read_bytes()


def test_modules_answer_on_synthetic_data(client, app):
    grid = client.get("/api/epochs/grid")
    assert grid.status_code == 200
    first = grid.json()["cells"][0]["images"][0]
    image = client.get(first["url"])
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/")

    assert client.get("/media/../README.md").status_code == 404
    from app.paths import sandbox_child
    import pytest
    with pytest.raises(ValueError):
        sandbox_child(Path(app.state.sample_dir), "../README.md")

    variance = client.get("/api/variance").json()
    assert "epoch_03" in variance["unstable_epochs"]
    bleed = client.get("/api/bleed").json()
    assert "epoch_04" in bleed["flagged_epochs"]
    provenance = client.get("/api/provenance", params={"q": "euler"}).json()
    assert provenance["comfyui_called"] is False
    assert provenance["count"] > 0
    profile = client.get("/api/profiler").json()
    assert profile["slowest_type"] == "KSampler"
    assert profile["comfyui_called"] is False
    indexed = client.post("/api/index/scan", json={"folder": "sample_data/loras"}).json()
    assert indexed["count"] == 6
    assert indexed["deleted"] == []
    assert indexed["files_unchanged"] is True
    shelf = client.get("/api/shelf", params={"unused_days": 30, "today": "2026-10-31"}).json()
    assert shelf["deleted"] == []
    assert shelf["files_unchanged"] is True
    assert shelf["comfyui_called"] is False
    assert any(item["action"] == "suggest_archive" for item in shelf["suggestions"])

    saved = client.post("/api/epochs/picks", json={
        "epoch": "epoch_04",
        "prompt_id": "amber_circle",
        "seed": 11,
        "path": "epochs/epoch_04/amber_circle/seed_11.png",
        "notes": "synthetic keeper",
    })
    assert saved.status_code == 200
    assert saved.json()["saved"]["notes"] == "synthetic keeper"

    recipe = client.post("/api/recipes", json={
        "name": "ui_blend",
        "method": "weighted_sum",
        "sources": [
            {"path": "sample_data/loras/style_amber.safetensors", "weight": 0.5},
            {"path": "sample_data/loras/style_teal.safetensors", "weight": 0.5},
        ],
    })
    assert recipe.status_code == 200
    assert recipe.json()["executed"] is False
    assert recipe.json()["recipe"]["sources"][0]["sha256"]
    replay = client.post("/api/recipes/ui_blend/replay")
    assert replay.status_code == 200
    assert replay.json()["executed"] is False
    history = (REPO / "sample_data" / "history" / "history.json").read_bytes()
    uploaded = client.post(
        "/api/profiler/upload",
        files={"file": ("history.json", history, "application/json")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["slowest_type"] == "KSampler"
    png = (REPO / "sample_data" / "epochs" / "epoch_01" / "amber_circle" / "seed_11.png").read_bytes()
    parsed = client.post(
        "/api/provenance/upload",
        files={"file": ("seed_11.png", png, "image/png")},
    )
    assert parsed.status_code == 200
    assert parsed.json()["comfyui_called"] is False
    assert parsed.json()["records"][0]["seed"] == 11
    assert app.state.transport.calls == []


def test_history_upload_and_launcher_files():
    history = (REPO / "sample_data" / "history" / "history.json").read_bytes()
    assert b"192.168." not in history
    for name in ("launch-linux.sh", "launch-mac.command", "launch-windows.bat"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "127.0.0.1" in text
        assert ".venv" in text
        assert "192.168.4.47" not in text
        assert "8188" not in text
        assert "requirements.txt" in text


def test_sample_data_has_no_secrets():
    banned = ("AKIA", "BEGIN PRIVATE", "BEGIN RSA", "api_key", "password", "192.168.4.47")
    for path in (REPO / "sample_data").rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".safetensors"}:
            continue
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text
