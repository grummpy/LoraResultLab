"""Write synthetic sample images, history, and tiny LoRA files.

Shapes and gradients only. No photographs, no people, no secrets.
Run from the repo root: python scripts/generate_samples.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modules.epoch_picker import build_workflow  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

SAMPLE = ROOT / "sample_data"
SIZE = 128

PROMPTS = {
    "training_prompts": [
        {
            "id": "amber_circle",
            "text": "a flat amber circle centered on a warm gradient field",
            "lora": "style_amber.safetensors",
        },
        {
            "id": "teal_bars",
            "text": "stacked teal bars on a cool gradient field",
            "lora": "style_teal.safetensors",
        },
        {
            "id": "violet_triangle",
            "text": "a violet triangle on a dusk gradient field",
            "lora": "style_amber.safetensors",
        },
    ],
    "bleed_prompts": [
        {
            "id": "unrelated_grid",
            "text": "a neutral gray grid with no decoration",
        },
        {
            "id": "unrelated_dots",
            "text": "evenly spaced gray dots on a light field",
        },
    ],
    "seeds": [11, 22, 33],
    "epochs": ["epoch_01", "epoch_02", "epoch_03", "epoch_04"],
    "unstable_epoch": "epoch_03",
    "bleed_epochs": ["epoch_01", "epoch_04"],
    "high_bleed_epoch": "epoch_04",
    "notes": "Procedural shapes and gradients. Not photographs. Not people.",
}


def gradient(c0, c1, axis="x") -> Image.Image:
    start = np.array(c0, dtype=np.float32)
    end = np.array(c1, dtype=np.float32)
    ramp = np.linspace(0, 1, SIZE, dtype=np.float32)
    if axis == "x":
        mix = ramp[None, :, None]
        image = (1 - mix) * start + mix * end
        image = np.repeat(image, SIZE, axis=0)
    else:
        mix = ramp[:, None, None]
        image = (1 - mix) * start + mix * end
        image = np.repeat(image, SIZE, axis=1)
    return Image.fromarray(image.astype(np.uint8), "RGB")


def draw_prompt(prompt_id: str, epoch: str, seed: int) -> Image.Image:
    epoch_index = int(epoch.split("_")[1])
    palettes = {
        "amber_circle": ((92, 42, 18), (232, 168, 64)),
        "teal_bars": ((10, 52, 64), (110, 206, 186)),
        "violet_triangle": ((36, 18, 72), (186, 132, 214)),
    }
    image = gradient(*palettes[prompt_id], axis="x" if prompt_id != "teal_bars" else "y")
    if epoch == "epoch_03":
        if seed == 22:
            return checker(image)
        if seed == 33:
            return stripes()
    draw = ImageDraw.Draw(image)
    shift = {11: 0, 22: 1, 33: -1}[seed]
    grow = {1: 0, 2: 4, 3: 0, 4: 8}[epoch_index]
    if prompt_id == "amber_circle":
        radius = 26 + grow
        cx, cy = SIZE // 2 + shift, SIZE // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(214, 112, 36))
        if epoch_index >= 4:
            draw.ellipse((cx - radius - 8, cy - radius - 8, cx + radius + 8, cy + radius + 8), outline=(120, 48, 16), width=3)
    elif prompt_id == "teal_bars":
        count = 3 + (1 if epoch_index >= 2 else 0) + (1 if epoch_index >= 4 else 0)
        for index in range(count):
            top = 16 + index * 18 + shift
            draw.rectangle((18, top, SIZE - 18, top + 10), fill=(16, 110, 112))
    else:
        inset = 28 - min(grow, 8)
        draw.polygon([(SIZE // 2 + shift, inset), (SIZE - inset, SIZE - inset), (inset, SIZE - inset)], fill=(118, 64, 176))
        if epoch_index >= 4:
            draw.polygon([(SIZE // 2 + shift, inset + 14), (SIZE - inset - 16, SIZE - inset - 8), (inset + 16, SIZE - inset - 8)], fill=(230, 210, 120))
    return image


def checker(image: Image.Image) -> Image.Image:
    out = Image.new("RGB", (SIZE, SIZE), (20, 16, 14))
    draw = ImageDraw.Draw(out)
    for y in range(0, SIZE, 16):
        for x in range(0, SIZE, 16):
            if ((x // 16) + (y // 16)) % 2 == 0:
                draw.rectangle((x, y, x + 15, y + 15), fill=(230, 200, 80))
    return out


def stripes() -> Image.Image:
    out = Image.new("RGB", (SIZE, SIZE), (18, 24, 48))
    draw = ImageDraw.Draw(out)
    for x in range(0, SIZE, 10):
        draw.rectangle((x, 0, x + 4, SIZE), fill=(80, 200, 170))
    return out


def neutral_grid() -> Image.Image:
    image = Image.new("RGB", (SIZE, SIZE), (232, 232, 232))
    draw = ImageDraw.Draw(image)
    for i in range(0, SIZE, 16):
        draw.line((i, 0, i, SIZE - 1), fill=(70, 70, 70))
        draw.line((0, i, SIZE - 1, i), fill=(70, 70, 70))
    return image


def neutral_dots() -> Image.Image:
    image = Image.new("RGB", (SIZE, SIZE), (242, 242, 242))
    draw = ImageDraw.Draw(image)
    for y in range(14, SIZE - 8, 22):
        for x in range(14, SIZE - 8, 22):
            draw.ellipse((x, y, x + 7, y + 7), fill=(55, 55, 55))
    return image


def loud_wash() -> Image.Image:
    bands = [(214, 36, 92), (28, 170, 86), (36, 72, 214)]
    array = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for y in range(SIZE):
        for x in range(SIZE):
            array[y, x] = bands[((x + y) // 8) % 3]
    return Image.fromarray(array, "RGB")


def save_png(path: Path, image: Image.Image, prompt: str, seed: int, epoch: str, prompt_id: str, lora_name: str | None) -> None:
    workflow = build_workflow(
        prompt=prompt,
        seed=seed,
        epoch=epoch,
        prompt_id=prompt_id,
        lora_name=lora_name,
        width=SIZE,
        height=SIZE,
    )
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", json.dumps(workflow))
    info.add_text("workflow", json.dumps({"nodes": workflow, "synthetic": True}))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", pnginfo=info)


def write_lora(path: Path, values, metadata: dict | None) -> None:
    array = np.asarray(values, dtype=np.float32)
    meta = None if not metadata else {key: str(value) for key, value in metadata.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lora_unet.weight": array}, str(path), metadata=meta)


def amber_meta(output_name: str) -> dict:
    return {
        "ss_base_model_version": "sdxl_base_v1-0",
        "ss_sd_model_name": "sd_xl_base_1.0",
        "ss_epoch": "4",
        "ss_network_dim": "8",
        "ss_network_alpha": "4",
        "ss_learning_rate": "0.0001",
        "ss_output_name": output_name,
        "ss_session_id": "synthetic-session-amber",
        "ss_tag_frequency": json.dumps({"synthetic": {"amberstyle": 12, "circle": 9}}),
        "modelspec.trigger_phrase": "amberstyle",
        "modelspec.architecture": "stable-diffusion-xl-v1-base/lora",
    }


def build_history() -> dict:
    amber = build_workflow(
        prompt=PROMPTS["training_prompts"][0]["text"],
        seed=11,
        epoch="epoch_04",
        prompt_id="amber_circle",
        lora_name="style_amber.safetensors",
    )
    teal = build_workflow(
        prompt=PROMPTS["training_prompts"][1]["text"],
        seed=22,
        epoch="epoch_02",
        prompt_id="teal_bars",
        lora_name="style_teal.safetensors",
    )

    def entry(prompt_id, graph, filename, executed_at, timings):
        return {
            "prompt": [1, prompt_id, graph, {"client_id": "synthetic"}, ["9"]],
            "outputs": {"9": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}},
            "status": {
                "status_str": "success",
                "completed": True,
                "messages": [
                    ["execution_start", {"prompt_id": prompt_id, "timestamp": 1_700_000_000_000}],
                    ["execution_success", {"prompt_id": prompt_id, "timestamp": 1_700_000_004_500}],
                ],
            },
            "node_timings": timings,
            "executed_at": executed_at,
        }

    slow = [
        {"node_id": "1", "class_type": "CheckpointLoaderSimple", "duration_ms": 40},
        {"node_id": "2", "class_type": "LoraLoader", "duration_ms": 25},
        {"node_id": "6", "class_type": "KSampler", "duration_ms": 2400},
        {"node_id": "8", "class_type": "VAEDecode", "duration_ms": 180},
        {"node_id": "9", "class_type": "SaveImage", "duration_ms": 15},
    ]
    also_slow = [
        {"node_id": "1", "class_type": "CheckpointLoaderSimple", "duration_ms": 35},
        {"node_id": "2", "class_type": "LoraLoader", "duration_ms": 22},
        {"node_id": "6", "class_type": "KSampler", "duration_ms": 1800},
        {"node_id": "8", "class_type": "VAEDecode", "duration_ms": 150},
        {"node_id": "9", "class_type": "SaveImage", "duration_ms": 12},
    ]
    return {
        "synthetic-amber-recent": entry(
            "synthetic-amber-recent",
            amber,
            "epoch_04_amber_circle_11.png",
            "2026-10-28T15:00:00Z",
            slow,
        ),
        "synthetic-teal-old": entry(
            "synthetic-teal-old",
            teal,
            "epoch_02_teal_bars_22.png",
            "2026-08-01T15:00:00Z",
            also_slow,
        ),
    }


def main() -> None:
    if SAMPLE.exists():
        shutil.rmtree(SAMPLE)
    epochs_root = SAMPLE / "epochs"
    for epoch in PROMPTS["epochs"]:
        for prompt in PROMPTS["training_prompts"]:
            for seed in PROMPTS["seeds"]:
                image = draw_prompt(prompt["id"], epoch, seed)
                destination = epochs_root / epoch / prompt["id"] / f"seed_{seed}.png"
                save_png(
                    destination,
                    image,
                    prompt["text"],
                    seed,
                    epoch,
                    prompt["id"],
                    prompt["lora"],
                )
    bleed_root = SAMPLE / "bleed"
    painters = {"unrelated_grid": neutral_grid, "unrelated_dots": neutral_dots}
    for epoch in PROMPTS["bleed_epochs"]:
        for prompt in PROMPTS["bleed_prompts"]:
            without = painters[prompt["id"]]()
            if epoch == PROMPTS["high_bleed_epoch"]:
                paired = loud_wash()
                lora_name = "style_amber.safetensors"
            else:
                paired = without.copy()
                paired.putpixel((0, 0), (228, 228, 228))
                lora_name = "style_amber.safetensors"
            folder = bleed_root / epoch / prompt["id"]
            save_png(folder / "without_lora" / "seed_11.png", without, prompt["text"], 11, epoch, prompt["id"], None)
            save_png(folder / "with_lora" / "seed_11.png", paired, prompt["text"], 11, epoch, prompt["id"], lora_name)

    lora_dir = SAMPLE / "loras"
    amber_values = [1, 0, 0, 0, 0.5, 0.25, 0.1, 0.05]
    teal_values = [0, 1, 0, 0, 0.2, 0.4, 0.6, 0.8]
    old_values = [0, 0, 1, 0, 0.7, 0.1, 0.2, 0.9]
    mystery_values = [0.3, 0.3, 0.3, 0.3]
    write_lora(lora_dir / "style_amber.safetensors", amber_values, amber_meta("style_amber"))
    write_lora(
        lora_dir / "style_teal.safetensors",
        teal_values,
        {
            "ss_base_model_version": "sdxl_base_v1-0",
            "ss_sd_model_name": "sd_xl_base_1.0",
            "ss_epoch": "2",
            "ss_network_dim": "8",
            "ss_network_alpha": "4",
            "ss_learning_rate": "0.0001",
            "ss_output_name": "style_teal",
            "ss_session_id": "synthetic-session-teal",
            "ss_tag_frequency": json.dumps({"synthetic": {"tealstyle": 8, "bars": 8}}),
            "modelspec.trigger_phrase": "tealstyle",
        },
    )
    write_lora(lora_dir / "old_unused.safetensors", old_values, {
        "ss_base_model_version": "sdxl_base_v1-0",
        "ss_epoch": "1",
        "ss_output_name": "old_unused",
        "ss_network_dim": "4",
        "modelspec.trigger_phrase": "oldstyle",
    })
    write_lora(lora_dir / "mystery_lora.safetensors", mystery_values, None)
    amber_path = lora_dir / "style_amber.safetensors"
    shutil.copyfile(amber_path, lora_dir / "style_amber_copy.safetensors")
    retag = amber_meta("style_amber_retag")
    retag["ss_output_name"] = "style_amber_retag"
    retag["loralab.note"] = "same weights as style_amber, different header"
    write_lora(lora_dir / "style_amber_retag.safetensors", amber_values, retag)

    history_path = SAMPLE / "history" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(build_history(), indent=2) + "\n", encoding="utf-8")
    (SAMPLE / "prompts.json").write_text(json.dumps(PROMPTS, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SAMPLE}")


if __name__ == "__main__":
    main()
