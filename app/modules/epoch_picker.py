"""Contact sheet of epochs × prompts, keeper picks, and a local queue file.

Writing the queue does not submit it. There is no HTTP client in this module.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

SEED_NAME = re.compile(r"^seed_(\d+)\.png$")


def load_prompt_catalog(sample_dir: Path) -> dict:
    path = sample_dir / "prompts.json"
    if not path.exists():
        return {"training_prompts": [], "bleed_prompts": [], "seeds": [], "epochs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_text_map(sample_dir: Path) -> dict[str, str]:
    catalog = load_prompt_catalog(sample_dir)
    mapping = {}
    for item in catalog.get("training_prompts", []) + catalog.get("bleed_prompts", []):
        mapping[item["id"]] = item.get("text", item["id"])
    return mapping


def prompt_lora_map(sample_dir: Path) -> dict[str, str]:
    catalog = load_prompt_catalog(sample_dir)
    mapping = {}
    for item in catalog.get("training_prompts", []):
        if item.get("lora"):
            mapping[item["id"]] = item["lora"]
    return mapping


def build_workflow(
    *,
    prompt: str,
    seed: int,
    epoch: str,
    prompt_id: str,
    lora_name: str | None,
    width: int = 512,
    height: int = 512,
) -> dict:
    """A ComfyUI-shaped workflow stored in the queue file. It is not executed here."""
    clip_source = ["2", 1] if lora_name else ["1", 1]
    model_source = ["2", 0] if lora_name else ["1", 0]
    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "synthetic_base.safetensors"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": clip_source},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "low quality, blurry", "clip": clip_source},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed),
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": model_source,
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": f"{epoch}_{prompt_id}_{seed}",
                "images": ["8", 0],
            },
        },
    }
    if lora_name:
        graph["2"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "strength_model": 0.8,
                "strength_clip": 0.7,
                "model": ["1", 0],
                "clip": ["1", 1],
            },
        }
    graph["_meta"] = {
        "epoch": epoch,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "seed": int(seed),
        "lora_name": lora_name,
    }
    return graph


def _epoch_dirs(root: Path) -> list[Path]:
    epochs_root = root / "epochs"
    if not epochs_root.is_dir():
        return []
    dirs = [path for path in epochs_root.iterdir() if path.is_dir()]
    return sorted(dirs, key=lambda path: path.name)


def _images_for(prompt_dir: Path) -> list[dict]:
    images = []
    if not prompt_dir.is_dir():
        return images
    for path in prompt_dir.iterdir():
        match = SEED_NAME.match(path.name)
        if match and path.is_file():
            images.append({"seed": int(match.group(1)), "file": path})
    images.sort(key=lambda item: item["seed"])
    return images


def build_grid(sample_dir: Path) -> dict:
    """Rows are prompts, columns are epochs. Image URLs point at /media."""
    texts = prompt_text_map(sample_dir)
    epoch_dirs = _epoch_dirs(sample_dir)
    prompt_ids: list[str] = []
    seen = set()
    cells = []
    seeds = set()
    for epoch_dir in epoch_dirs:
        prompt_dirs = sorted((path for path in epoch_dir.iterdir() if path.is_dir()), key=lambda p: p.name)
        for prompt_dir in prompt_dirs:
            if prompt_dir.name not in seen:
                seen.add(prompt_dir.name)
                prompt_ids.append(prompt_dir.name)
            images = []
            for item in _images_for(prompt_dir):
                seeds.add(item["seed"])
                relative = item["file"].resolve().relative_to(sample_dir.resolve()).as_posix()
                images.append(
                    {
                        "seed": item["seed"],
                        "path": relative,
                        "url": "/media/" + relative,
                    }
                )
            cells.append(
                {
                    "epoch": epoch_dir.name,
                    "prompt_id": prompt_dir.name,
                    "text": texts.get(prompt_dir.name, prompt_dir.name),
                    "images": images,
                }
            )
    prompts = [{"id": prompt_id, "text": texts.get(prompt_id, prompt_id)} for prompt_id in prompt_ids]
    return {
        "epochs": [path.name for path in epoch_dirs],
        "prompts": prompts,
        "seeds": sorted(seeds),
        "cells": cells,
        "image_count": sum(len(cell["images"]) for cell in cells),
    }


def write_queue_file(sample_dir: Path, data_dir: Path, grid: dict | None = None) -> dict:
    """Write one workflow per epoch × prompt × seed. Does not contact ComfyUI."""
    grid = grid or build_grid(sample_dir)
    loras = prompt_lora_map(sample_dir)
    items = []
    for cell in grid["cells"]:
        for image in cell["images"]:
            workflow = build_workflow(
                prompt=cell["text"],
                seed=image["seed"],
                epoch=cell["epoch"],
                prompt_id=cell["prompt_id"],
                lora_name=loras.get(cell["prompt_id"]),
            )
            items.append(
                {
                    "id": f"{cell['epoch']}__{cell['prompt_id']}__seed_{image['seed']}",
                    "epoch": cell["epoch"],
                    "prompt_id": cell["prompt_id"],
                    "prompt": cell["text"],
                    "seed": image["seed"],
                    "image": image["path"],
                    "workflow": workflow,
                }
            )
    payload = {
        "version": 1,
        "submitted": False,
        "comfyui_called": False,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "count": len(items),
        "items": items,
    }
    destination = data_dir / "queue" / "epoch_queue.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(destination),
        "relative": "queue/epoch_queue.json",
        "count": len(items),
        "submitted": False,
        "comfyui_called": False,
        "epochs": grid["epochs"],
        "prompts": [prompt["id"] for prompt in grid["prompts"]],
        "seeds": grid["seeds"],
    }


def load_queue(data_dir: Path) -> dict:
    path = data_dir / "queue" / "epoch_queue.json"
    if not path.exists():
        raise FileNotFoundError("No queue file yet. Write one from Epoch Picker first.")
    return json.loads(path.read_text(encoding="utf-8"))


def picks_path(data_dir: Path) -> Path:
    return data_dir / "picks.json"


def load_picks(data_dir: Path) -> dict:
    path = picks_path(data_dir)
    if not path.exists():
        return {"picks": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_pick(data_dir: Path, pick: dict) -> dict:
    required = ("epoch", "prompt_id", "seed", "path")
    missing = [key for key in required if pick.get(key) in (None, "")]
    if missing:
        raise ValueError("pick requires " + ", ".join(missing))
    notes = str(pick.get("notes") or "")
    if len(notes) > 2000:
        raise ValueError("notes are limited to 2000 characters")
    record = {
        "epoch": str(pick["epoch"]),
        "prompt_id": str(pick["prompt_id"]),
        "seed": int(pick["seed"]),
        "path": str(pick["path"]),
        "notes": notes,
        "picked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    payload = load_picks(data_dir)
    picks = payload.get("picks", [])
    picks = [
        item
        for item in picks
        if not (
            item.get("epoch") == record["epoch"]
            and item.get("prompt_id") == record["prompt_id"]
            and int(item.get("seed", -1)) == record["seed"]
        )
    ]
    picks.append(record)
    data_dir.mkdir(parents=True, exist_ok=True)
    picks_path(data_dir).write_text(json.dumps({"picks": picks}, indent=2) + "\n", encoding="utf-8")
    return {"picks": picks, "saved": record}
