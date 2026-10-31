"""Read PNG tEXt prompt/workflow chunks and ComfyUI history JSON.

This module only parses data it is given. It does not open a network connection.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PIL import Image


def _graph_from_prompt_field(prompt_field):
    if isinstance(prompt_field, list) and len(prompt_field) >= 3 and isinstance(prompt_field[2], dict):
        return prompt_field[2]
    if isinstance(prompt_field, dict):
        return prompt_field
    return {}


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_workflow_graph(graph: dict) -> dict:
    """Pull seed, sampler, prompt text, and LoRA strengths out of a node graph."""
    if not isinstance(graph, dict):
        graph = {}
    seeds = []
    sampler = None
    scheduler = None
    steps = None
    cfg = None
    prompts = []
    loras = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        inputs = node.get("inputs") or {}
        if class_type == "KSampler":
            seed = _as_int(inputs.get("seed"))
            if seed is not None:
                seeds.append(seed)
            sampler = sampler or inputs.get("sampler_name")
            scheduler = scheduler or inputs.get("scheduler")
            steps = steps if steps is not None else _as_int(inputs.get("steps"))
            cfg = cfg if cfg is not None else _as_float(inputs.get("cfg"))
        elif class_type == "CLIPTextEncode":
            text = inputs.get("text")
            if isinstance(text, str) and text.strip():
                prompts.append(text.strip())
        elif class_type == "LoraLoader":
            name = inputs.get("lora_name")
            if name:
                loras.append(
                    {
                        "name": str(name),
                        "strength_model": _as_float(inputs.get("strength_model")),
                        "strength_clip": _as_float(inputs.get("strength_clip")),
                        "node_id": str(node_id),
                    }
                )
    positive = prompts[0] if prompts else ""
    meta = graph.get("_meta") if isinstance(graph.get("_meta"), dict) else {}
    return {
        "seed": seeds[0] if seeds else _as_int(meta.get("seed")),
        "sampler": sampler,
        "scheduler": scheduler,
        "steps": steps,
        "cfg": cfg,
        "prompt_text": meta.get("prompt") or positive,
        "negative_text": prompts[1] if len(prompts) > 1 else "",
        "loras": loras,
        "epoch": meta.get("epoch"),
        "prompt_id": meta.get("prompt_id"),
    }


def parse_png(path: Path, sample_root: Path | None = None) -> dict:
    with Image.open(path) as image:
        info = dict(image.info)
    prompt_raw = info.get("prompt")
    workflow_raw = info.get("workflow")
    graph = {}
    if prompt_raw:
        try:
            graph = json.loads(prompt_raw)
        except json.JSONDecodeError:
            graph = {}
    workflow = None
    if workflow_raw:
        try:
            workflow = json.loads(workflow_raw)
        except json.JSONDecodeError:
            workflow = {"raw": workflow_raw}
    parsed = parse_workflow_graph(graph if isinstance(graph, dict) else {})
    relative = str(path)
    if sample_root is not None:
        try:
            relative = path.resolve().relative_to(sample_root.resolve()).as_posix()
        except ValueError:
            relative = path.name
    return {
        "image": relative,
        "source": "png",
        "prompt_id": parsed.get("prompt_id"),
        "executed_at": None,
        "prompt_text": parsed.get("prompt_text") or "",
        "seed": parsed.get("seed"),
        "sampler": parsed.get("sampler"),
        "scheduler": parsed.get("scheduler"),
        "steps": parsed.get("steps"),
        "cfg": parsed.get("cfg"),
        "loras": parsed.get("loras") or [],
        "epoch": parsed.get("epoch"),
        "workflow": workflow,
        "comfy_prompt_id": None,
    }


def parse_history(payload: dict) -> list[dict]:
    """Parse a ComfyUI /history document. Accepts a single prompt object too."""
    if not isinstance(payload, dict):
        raise ValueError("history JSON must be an object")
    records = []
    items = payload.items()
    for prompt_id, entry in items:
        if not isinstance(entry, dict):
            continue
        graph = _graph_from_prompt_field(entry.get("prompt"))
        parsed = parse_workflow_graph(graph)
        outputs = entry.get("outputs") or {}
        filenames = []
        for node_output in outputs.values():
            if isinstance(node_output, dict):
                for image in node_output.get("images") or []:
                    if isinstance(image, dict) and image.get("filename"):
                        filenames.append(str(image["filename"]))
        image_name = filenames[0] if filenames else None
        records.append(
            {
                "image": image_name,
                "source": "history",
                "prompt_id": parsed.get("prompt_id"),
                "executed_at": entry.get("executed_at"),
                "prompt_text": parsed.get("prompt_text") or "",
                "seed": parsed.get("seed"),
                "sampler": parsed.get("sampler"),
                "scheduler": parsed.get("scheduler"),
                "steps": parsed.get("steps"),
                "cfg": parsed.get("cfg"),
                "loras": parsed.get("loras") or [],
                "epoch": parsed.get("epoch"),
                "workflow": graph or None,
                "comfy_prompt_id": str(prompt_id),
                "filenames": filenames,
            }
        )
    return records


def scan_sample(sample_dir: Path) -> list[dict]:
    records = []
    for path in sorted(sample_dir.rglob("*.png")):
        records.append(parse_png(path, sample_dir))
    history_path = sample_dir / "history" / "history.json"
    if history_path.exists():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        records.extend(parse_history(payload))
    return records


def search_records(records: list[dict], query: str) -> list[dict]:
    needle = (query or "").strip().lower()
    if not needle:
        return records
    matched = []
    for record in records:
        lora_bits = " ".join(
            f"{item.get('name', '')} {item.get('strength_model', '')}" for item in record.get("loras") or []
        )
        haystack = " ".join(
            [
                str(record.get("image") or ""),
                str(record.get("prompt_text") or ""),
                str(record.get("seed") or ""),
                str(record.get("sampler") or ""),
                str(record.get("epoch") or ""),
                str(record.get("prompt_id") or ""),
                str(record.get("source") or ""),
                lora_bits,
            ]
        ).lower()
        if needle in haystack:
            matched.append(record)
    return matched


def usage_by_lora(records: list[dict]) -> dict[str, str]:
    """Map a LoRA filename to the latest executed_at date (YYYY-MM-DD) seen in provenance."""
    latest: dict[str, str] = {}
    for record in records:
        stamp = record.get("executed_at")
        if not stamp:
            continue
        day = str(stamp)[:10]
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            continue
        for lora in record.get("loras") or []:
            name = Path(str(lora.get("name") or "")).name
            if not name:
                continue
            current = latest.get(name)
            if current is None or day > current:
                latest[name] = day
    return latest
