"""Local web UI and JSON API. Listens on 127.0.0.1 only."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.comfy import (
    ComfyGuardError,
    HttpxTransport,
    fetch_history,
    submit_workflows,
)
from app.modules import epoch_picker, hash_index, merge_recipe, node_profiler, provenance, seed_variance, shelf_life, style_bleed
from app.paths import ART_DIR, SAMPLE_DIR, STATIC_DIR, REPO_ROOT, resolve_user_path, sandbox_child
from app.settings import load_settings, update_settings

MODULE_BLURBS = [
    {
        "id": "epochs",
        "name": "Epoch Picker",
        "summary": "Contact sheet of prompts by epoch. Zoom, compare, and save a keeper. The queue file stays local.",
    },
    {
        "id": "variance",
        "name": "Seed Variance",
        "summary": "How much seeds disagree inside an epoch. SSIM by default, CLIP only if you configure a local model.",
    },
    {
        "id": "bleed",
        "name": "Style Bleed",
        "summary": "With-LoRA versus without-LoRA on unrelated prompts. High leakage means the LoRA moved the picture.",
    },
    {
        "id": "merge",
        "name": "Merge Recipe",
        "summary": "Record sources, weights, and hashes. Replay writes a CPU script and does not run it.",
    },
    {
        "id": "provenance",
        "name": "Provenance",
        "summary": "PNG prompt chunks and history JSON: seed, sampler, and LoRA strengths. Searchable, read-only.",
    },
    {
        "id": "profiler",
        "name": "Node Profiler",
        "summary": "Rank slow node types from history timings. Reads JSON. Never posts a job.",
    },
    {
        "id": "index",
        "name": "Hash Index",
        "summary": "SHA256, AutoV2, and training metadata for .safetensors files in a folder you choose.",
    },
    {
        "id": "shelf",
        "name": "Shelf Life",
        "summary": "Flags unused, duplicate, and under-documented LoRAs. Suggests an archive. Deletes nothing.",
    },
]


def create_app(data_dir: Path | None = None, sample_dir: Path | None = None) -> FastAPI:
    data = Path(data_dir) if data_dir is not None else REPO_ROOT / "data"
    sample = Path(sample_dir) if sample_dir is not None else SAMPLE_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.data_dir.mkdir(parents=True, exist_ok=True)
        load_settings(app.state.data_dir)
        yield

    app = FastAPI(title="LoraResultLab", version="1.0.0", lifespan=lifespan)
    app.state.data_dir = data
    app.state.sample_dir = sample
    app.state.transport = HttpxTransport()

    static_dir = STATIC_DIR
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = static_dir / "index.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/favicon.ico")
    def favicon():
        return FileResponse(ART_DIR / "icon.png", media_type="image/png")

    @app.get("/art/icon.png")
    def art_icon():
        return FileResponse(ART_DIR / "icon.png", media_type="image/png")

    @app.get("/art/cover.png")
    def art_cover():
        return FileResponse(ART_DIR / "cover.png", media_type="image/png")

    @app.get("/media/{relative:path}")
    def media(relative: str):
        try:
            path = sandbox_child(app.state.sample_dir, relative)
        except ValueError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(404, detail="not found")
        return FileResponse(path)

    @app.get("/api/health")
    def health():
        settings = load_settings(app.state.data_dir)
        return {
            "ok": True,
            "bind": "127.0.0.1",
            "comfyui_enabled": settings.comfyui_enabled,
            "comfyui_host": settings.comfyui_host,
        }

    @app.get("/api/settings")
    def get_settings():
        return load_settings(app.state.data_dir).to_dict()

    @app.post("/api/settings")
    async def post_settings(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, detail="settings must be an object")
        # confirm is per request and is never persisted
        body.pop("confirm", None)
        settings = update_settings(
            app.state.data_dir,
            {
                "comfyui_enabled": body.get("comfyui_enabled", False) if "comfyui_enabled" in body else load_settings(app.state.data_dir).comfyui_enabled,
                "comfyui_host": body.get("comfyui_host", load_settings(app.state.data_dir).comfyui_host),
            },
        )
        return settings.to_dict()

    @app.get("/api/dashboard")
    def dashboard():
        settings = load_settings(app.state.data_dir)
        grid = epoch_picker.build_grid(app.state.sample_dir)
        variance = seed_variance.variance_report(app.state.sample_dir)
        bleed = style_bleed.bleed_report(app.state.sample_dir)
        picks = epoch_picker.load_picks(app.state.data_dir)
        indexed = hash_index.list_indexed(app.state.data_dir / "catalog.sqlite")
        lora_dir = app.state.sample_dir / "loras"
        sample_loras = len(list(lora_dir.glob("*.safetensors"))) if lora_dir.is_dir() else 0
        return {
            "comfyui_enabled": settings.comfyui_enabled,
            "comfyui_host": settings.comfyui_host,
            "comfyui_called": False,
            "epochs": len(grid["epochs"]),
            "prompts": len(grid["prompts"]),
            "sample_images": grid["image_count"],
            "picks": len(picks.get("picks", [])),
            "indexed_loras": len(indexed),
            "sample_loras": sample_loras,
            "unstable_epochs": variance["unstable_epochs"],
            "bleed_flags": bleed["flagged_epochs"],
            "modules": MODULE_BLURBS,
            "today": date.today().isoformat(),
        }

    @app.get("/api/epochs/grid")
    def epoch_grid():
        return epoch_picker.build_grid(app.state.sample_dir)

    @app.get("/api/epochs/picks")
    def get_picks():
        return epoch_picker.load_picks(app.state.data_dir)

    @app.post("/api/epochs/picks")
    async def post_pick(request: Request):
        body = await request.json()
        try:
            return epoch_picker.save_pick(app.state.data_dir, body)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.post("/api/epochs/queue")
    def post_queue():
        result = epoch_picker.write_queue_file(app.state.sample_dir, app.state.data_dir)
        result["message"] = "Queue file written. ComfyUI was not contacted."
        return result

    @app.post("/api/epochs/submit")
    async def post_submit(request: Request):
        body = await _json_object(request)
        confirm = body.get("confirm", False)
        settings = load_settings(app.state.data_dir)
        try:
            queue = epoch_picker.load_queue(app.state.data_dir)
        except FileNotFoundError as exc:
            # Still refuse a disabled submit before complaining about the file,
            # so the default path never looks like it tried the network.
            if confirm is not True or not settings.comfyui_enabled or not (settings.comfyui_host or "").strip():
                try:
                    from app.comfy import assert_comfy_allowed

                    assert_comfy_allowed(settings, confirm)
                except ComfyGuardError as guard:
                    raise HTTPException(403, detail=str(guard)) from guard
            raise HTTPException(404, detail=str(exc)) from exc
        workflows = []
        for item in queue.get("items", []):
            graph = item.get("workflow") or {}
            workflows.append({key: value for key, value in graph.items() if not str(key).startswith("_")})
        try:
            return submit_workflows(
                settings,
                workflows,
                confirm=confirm,
                transport=request.app.state.transport,
            )
        except ComfyGuardError as exc:
            raise HTTPException(403, detail=str(exc)) from exc

    @app.get("/api/variance")
    def get_variance(method: str = "auto"):
        try:
            return seed_variance.variance_report(app.state.sample_dir, method=method)
        except RuntimeError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.get("/api/bleed")
    def get_bleed(method: str = "auto"):
        try:
            return style_bleed.bleed_report(app.state.sample_dir, method=method)
        except RuntimeError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.get("/api/recipes")
    def get_recipes():
        return {"recipes": merge_recipe.list_recipes(app.state.data_dir), "deleted": []}

    @app.post("/api/recipes")
    async def post_recipe(request: Request):
        body = await _json_object(request)
        _fill_recipe_hashes(body)
        try:
            saved = merge_recipe.save_recipe(app.state.data_dir, body)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        saved["deleted"] = []
        saved["executed"] = False
        return saved

    @app.get("/api/recipes/{name}")
    def get_recipe(name: str):
        try:
            return merge_recipe.load_recipe(app.state.data_dir, name)
        except FileNotFoundError as exc:
            raise HTTPException(404, detail="recipe not found") from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.post("/api/recipes/{name}/replay")
    def post_replay(name: str):
        try:
            return merge_recipe.replay_recipe(app.state.data_dir, name)
        except FileNotFoundError as exc:
            raise HTTPException(404, detail="recipe not found") from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.post("/api/recipes/{name}/merge-cpu")
    def post_merge_cpu(name: str):
        try:
            return merge_recipe.merge_cpu(app.state.data_dir, name, device="cpu")
        except FileNotFoundError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.get("/api/provenance")
    def get_provenance(q: str = ""):
        records = provenance.scan_sample(app.state.sample_dir)
        matched = provenance.search_records(records, q)
        return {"count": len(matched), "records": matched, "comfyui_called": False}

    @app.post("/api/provenance/upload")
    async def upload_provenance(file: UploadFile):
        raw = await file.read()
        name = (file.filename or "upload").lower()
        if name.endswith(".json"):
            try:
                payload = json.loads(raw.decode("utf-8"))
                records = provenance.parse_history(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise HTTPException(400, detail=f"history JSON could not be parsed: {exc}") from exc
        elif name.endswith(".png"):
            destination = app.state.data_dir / "uploads" / Path(file.filename or "upload.png").name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            try:
                records = [provenance.parse_png(destination, None)]
            except Exception as exc:
                raise HTTPException(400, detail=f"PNG could not be parsed: {exc}") from exc
        else:
            raise HTTPException(400, detail="upload a .png or a .json history file")
        return {"records": records, "comfyui_called": False, "deleted": []}

    @app.post("/api/provenance/fetch-history")
    async def fetch_provenance_history(request: Request):
        body = await _json_object(request)
        return await _read_history(request, body, kind="provenance")

    @app.get("/api/profiler")
    def get_profiler():
        history_path = app.state.sample_dir / "history" / "history.json"
        if not history_path.exists():
            raise HTTPException(404, detail="sample history is missing")
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        report = node_profiler.rank_nodes(payload)
        report["source"] = "sample"
        return report

    @app.post("/api/profiler/upload")
    async def upload_profiler(file: UploadFile):
        raw = await file.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
            report = node_profiler.rank_nodes(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        report["source"] = "upload"
        report["comfyui_called"] = False
        return report

    @app.post("/api/profiler/fetch-history")
    async def fetch_profiler_history(request: Request):
        body = await _json_object(request)
        return await _read_history(request, body, kind="profiler")

    @app.get("/api/index")
    def get_index():
        rows = hash_index.list_indexed(app.state.data_dir / "catalog.sqlite")
        return {"loras": rows, "count": len(rows), "deleted": []}

    @app.post("/api/index/scan")
    async def post_index_scan(request: Request):
        body = await _json_object(request)
        folder_value = body.get("folder") or "sample_data/loras"
        folder = resolve_user_path(str(folder_value))
        before = _safetensors_snapshot(folder) if folder.is_dir() else {}
        try:
            rows = hash_index.scan_folder(folder, app.state.data_dir / "catalog.sqlite")
        except FileNotFoundError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        after = _safetensors_snapshot(folder)
        return {
            "count": len(rows),
            "loras": rows,
            "deleted": [],
            "files_unchanged": before == after,
        }

    @app.get("/api/shelf")
    def get_shelf(unused_days: int = 30, today: str | None = None, folder: str = "sample_data/loras"):
        if unused_days < 0:
            raise HTTPException(400, detail="unused_days must be >= 0")
        as_of = today or date.today().isoformat()
        try:
            datetime_check = date.fromisoformat(as_of)
        except ValueError as exc:
            raise HTTPException(400, detail="today must be YYYY-MM-DD") from exc
        folder_path = resolve_user_path(folder)
        before = _safetensors_snapshot(folder_path) if folder_path.is_dir() else {}
        db_path = app.state.data_dir / "catalog.sqlite"
        try:
            rows = hash_index.scan_folder(folder_path, db_path)
        except FileNotFoundError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        records = provenance.scan_sample(app.state.sample_dir)
        usage = provenance.usage_by_lora(records)
        try:
            report = shelf_life.analyze(rows, usage, unused_days=unused_days, today=datetime_check.isoformat())
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        after = _safetensors_snapshot(folder_path)
        report["files_unchanged"] = before == after
        report["comfyui_called"] = False
        return report

    async def _read_history(request: Request, body: dict, kind: str):
        settings = load_settings(app.state.data_dir)
        try:
            fetched = fetch_history(
                settings,
                confirm=body.get("confirm", False),
                user_initiated=body.get("user_initiated", False),
                transport=request.app.state.transport,
            )
        except ComfyGuardError as exc:
            raise HTTPException(403, detail=str(exc)) from exc
        try:
            payload = json.loads(fetched["body"])
        except json.JSONDecodeError:
            fetched["parse_error"] = "ComfyUI response was not JSON"
            fetched["kind"] = kind
            return fetched
        if kind == "profiler":
            fetched["profile"] = node_profiler.rank_nodes(payload)
        else:
            fetched["records"] = provenance.parse_history(payload)
        fetched["kind"] = kind
        fetched["posted"] = False
        return fetched

    return app


async def _json_object(request: Request) -> dict:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, detail="expected JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, detail="expected a JSON object")
    return body


def _fill_recipe_hashes(body: dict) -> None:
    for source in body.get("sources") or []:
        raw_path = source.get("path")
        if not raw_path:
            continue
        path = resolve_user_path(str(raw_path))
        if path.is_file() and path.suffix == ".safetensors":
            if not source.get("sha256"):
                source["sha256"] = hash_index.file_sha256(path)
            if not source.get("autov2"):
                source["autov2"] = hash_index.autov2_hash(path)
            source["path"] = str(path)


def _safetensors_snapshot(folder: Path) -> dict:
    if not folder.is_dir():
        return {}
    snapshot = {}
    for path in folder.rglob("*.safetensors"):
        stat = path.stat()
        snapshot[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


app = create_app()
