# LoraResultLab — Build Sheet
Repo: https://github.com/grummpy/LoraResultLab (public). Owner: grummpy.

## Purpose
Judge and organize LoRA training results. Works on already-generated images, ComfyUI output/metadata, and local .safetensors files. Read-only toward ComfyUI by default.

## GPU / ComfyUI safety (hard rule)
- Default: no ComfyUI connection at all. EpochPicker consumes existing images or WRITES A QUEUE FILE (workflow JSON per epoch × prompt × seed) only.
- Optional "Submit to ComfyUI" is OFF by default, hidden behind a settings toggle, host configurable (default blank; documented example 192.168.4.47:8188), and every submit requires an explicit confirm dialog stating it runs on Zach's PC GPU. Never auto-submit; tests assert no HTTP call to ComfyUI unless toggle+confirm set.
- OutputProvenance/NodeTimeProfiler read history JSON from file upload or GET /history only when user clicks, never POST.

## Modules
1. **EpochPicker** — grid: rows = fixed prompt set, columns = epochs; zoom, A/B compare, pick keeper with notes → `picks.json`.
2. **SeedVarianceBand** — per epoch, image similarity (CLIP CPU / SSIM fallback) across seeds; mean ± band chart; flag unstable epochs.
3. **StyleBleedCheck** — paired outputs with vs without LoRA on unrelated prompts; similarity/style-distance score; leakage score per epoch, threshold flag.
4. **LoRAMergeRecipe** — record merge sources (hash, path), weights, method, output hash → YAML recipe; replay produces a merge script/command file (CPU merge via safetensors+torch CPU optional), never auto-runs GPU.
5. **OutputProvenance** — parse PNG tEXt "prompt"/"workflow" and history JSON; link image → workflow, seed, sampler, LoRAs + strengths; searchable table.
6. **NodeTimeProfiler** — parse history execution timing; rank slowest node types/IDs; bar chart; read-only.
7. **LoRAHashIndex** — scan folders for .safetensors; SHA256 + AutoV2 short hash; read header metadata (ss_* training keys, base model, trigger words); SQLite catalog.
8. **LoraShelfLife** — cross-reference index with provenance usage; flag LoRAs unused for N days, duplicates by hash, missing metadata; suggest archive (no deletion).

## UI
Tabs per module + Dashboard. Settings page holds the ComfyUI toggle (default off).

## Synthetic sample data
Generated gradient/shape images per fake epoch/seed, fake PNG metadata, fake history JSON, tiny dummy .safetensors files with synthetic metadata. No real people.

## Tests
Grid builds from sample folder; queue file correct and nothing submitted; variance/bleed math; recipe round-trip; metadata parsing; node ranking; hash correctness; stale flags; ComfyUI-off-by-default test.

## Chris's standing requirements (must all be met)
- Launcher-ready: cover art `art/cover.png` and app icon `art/icon.png` (already committed) used in README and as web UI favicon/header.
- One-click launchers: `launch-mac.command`, `launch-windows.bat`, `launch-linux.sh` — create venv, install deps, start server, open browser.
- Local web UI (FastAPI/Flask + static HTML/JS, localhost only, bind 127.0.0.1).
- Tests (pytest) covering every module + a UI smoke test; CI via GitHub Actions on push.
- README: what it does, screenshots, quick start per OS, safety notes.
- Public repo: SYNTHETIC sample data only (procedurally generated shapes/gradients, fake metadata). No real photos of people, no kids' data, no secrets/tokens/IPs beyond documented defaults.
- Fully launchable and tested before finishing; final work pushed and MERGED to `main` (not left as an open PR).
- No GPU jobs on Zach's PC (ComfyUI 192.168.4.47:8188) without explicit go-ahead.
