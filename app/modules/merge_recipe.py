"""Record a LoRA merge and write a CPU script. Replay never runs the script."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ALLOWED_METHODS = ("weighted_sum", "add_difference", "dare_linear")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_recipe(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("recipe must be a mapping")
    name = str(data.get("name") or "").strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("recipe name must be a plain file stem")
    method = str(data.get("method") or "weighted_sum")
    if method not in ALLOWED_METHODS:
        raise ValueError("unsupported merge method")
    sources = data.get("sources") or []
    if len(sources) < 2:
        raise ValueError("a merge needs at least two sources")
    cleaned_sources = []
    for source in sources:
        weight = float(source.get("weight"))
        cleaned_sources.append(
            {
                "path": str(source.get("path") or ""),
                "sha256": str(source.get("sha256") or ""),
                "autov2": str(source.get("autov2") or ""),
                "weight": weight,
            }
        )
    output = data.get("output") or {}
    filename = str(output.get("filename") or f"{name}.safetensors")
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError("output filename must be a plain file name")
    recipe = {
        "name": name,
        "method": method,
        "created_at": str(data.get("created_at") or _now()),
        "sources": cleaned_sources,
        "output": {
            "filename": filename,
            "sha256": output.get("sha256"),
        },
        "notes": str(data.get("notes") or "Suggestion only. Replay writes a CPU script and does not run it."),
        "deleted": [],
    }
    return recipe


def recipe_to_yaml(recipe: dict) -> str:
    return yaml.safe_dump(normalize_recipe(recipe), sort_keys=False)


def recipe_from_yaml(text: str) -> dict:
    loaded = yaml.safe_load(text)
    return normalize_recipe(loaded)


def recipe_dir(data_dir: Path) -> Path:
    path = data_dir / "recipes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_recipe(data_dir: Path, recipe: dict) -> dict:
    recipe = normalize_recipe(recipe)
    path = recipe_dir(data_dir) / f"{recipe['name']}.yaml"
    path.write_text(recipe_to_yaml(recipe), encoding="utf-8")
    return {"recipe": recipe, "path": str(path)}


def load_recipe(data_dir: Path, name: str) -> dict:
    path = recipe_dir(data_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(name)
    return recipe_from_yaml(path.read_text(encoding="utf-8"))


def list_recipes(data_dir: Path) -> list[dict]:
    folder = data_dir / "recipes"
    if not folder.exists():
        return []
    recipes = []
    for path in sorted(folder.glob("*.yaml")):
        recipes.append(recipe_from_yaml(path.read_text(encoding="utf-8")))
    return recipes


def render_merge_script(recipe: dict) -> str:
    recipe = normalize_recipe(recipe)
    lines = [
        "# LoraResultLab CPU merge script",
        "# This file is a suggestion. LoraResultLab does not execute it.",
        "# Device is CPU. CUDA is intentionally unused.",
        "DEVICE = 'cpu'",
        "DELETED = []  # this script never deletes sources",
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "from safetensors.numpy import load_file, save_file",
        "",
        f"METHOD = {recipe['method']!r}",
        f"OUTPUT = Path({recipe['output']['filename']!r})",
        "SOURCES = [",
    ]
    for source in recipe["sources"]:
        lines.append(
            "    {{'path': Path({path!r}), 'weight': {weight!r}, 'sha256': {sha!r}, 'autov2': {auto!r}}},".format(
                path=source["path"],
                weight=source["weight"],
                sha=source["sha256"],
                auto=source["autov2"],
            )
        )
    lines.extend(
        [
            "]",
            "",
            "def main():",
            "    if DEVICE != 'cpu':",
            "        sys.exit('GPU merge is refused')",
            "    if METHOD != 'weighted_sum':",
            "        sys.exit('Only weighted_sum is implemented in this CPU script. Nothing was written.')",
            "    weights = [item['weight'] for item in SOURCES]",
            "    total = sum(weights)",
            "    if total == 0:",
            "        sys.exit('weights sum to zero')",
            "    loaded = [load_file(str(item['path'])) for item in SOURCES]",
            "    keys = set(loaded[0])",
            "    for tensors in loaded[1:]:",
            "        keys &= set(tensors)",
            "    merged = {}",
            "    for key in sorted(keys):",
            "        acc = None",
            "        for tensors, weight in zip(loaded, weights):",
            "            piece = np.asarray(tensors[key], dtype=np.float32) * (weight / total)",
            "            acc = piece if acc is None else acc + piece",
            "        merged[key] = acc",
            "    metadata = {'loralab.merge': 'weighted_sum', 'loralab.device': 'cpu'}",
            "    save_file(merged, str(OUTPUT), metadata=metadata)",
            "    print('wrote', OUTPUT, 'on cpu')",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )
    return "\n".join(lines)


def render_command_file(recipe: dict, script_name: str) -> str:
    recipe = normalize_recipe(recipe)
    return (
        "# Run by hand if you want a CPU merge. LoraResultLab will not launch this.\n"
        f"# method: {recipe['method']}\n"
        "# device: cpu\n"
        f"python {script_name}\n"
    )


def replay_recipe(data_dir: Path, name: str) -> dict:
    """Write the script and command file. Do not execute either one."""
    recipe = load_recipe(data_dir, name)
    folder = recipe_dir(data_dir)
    script_path = folder / f"{name}_merge.py"
    command_path = folder / f"{name}_merge.command.txt"
    script_path.write_text(render_merge_script(recipe), encoding="utf-8")
    command_path.write_text(render_command_file(recipe, script_path.name), encoding="utf-8")
    return {
        "name": name,
        "script": str(script_path),
        "command": str(command_path),
        "executed": False,
        "deleted": [],
        "device": "cpu",
    }


def merge_cpu(data_dir: Path, name: str, *, device: str = "cpu") -> dict:
    """Explicit in-process weighted sum on CPU. Not called by replay."""
    if device != "cpu":
        raise ValueError("GPU merge is refused. Pass device='cpu'.")
    recipe = load_recipe(data_dir, name)
    if recipe["method"] != "weighted_sum":
        raise ValueError("CPU merge currently implements weighted_sum only. Nothing was written.")
    from safetensors.numpy import load_file, save_file

    weights = [source["weight"] for source in recipe["sources"]]
    total = sum(weights)
    if total == 0:
        raise ValueError("weights sum to zero")
    loaded = []
    for source in recipe["sources"]:
        path = Path(source["path"])
        if not path.is_file():
            raise FileNotFoundError(source["path"])
        loaded.append(load_file(str(path)))
    keys = set(loaded[0])
    for tensors in loaded[1:]:
        keys &= set(tensors)
    if not keys:
        raise ValueError("sources share no tensors")
    merged = {}
    for key in sorted(keys):
        acc = None
        for tensors, weight in zip(loaded, weights):
            piece = np.asarray(tensors[key], dtype=np.float32) * (weight / total)
            acc = piece if acc is None else acc + piece
        merged[key] = acc
    output_path = recipe_dir(data_dir) / recipe["output"]["filename"]
    save_file(merged, str(output_path), metadata={"loralab.merge": "weighted_sum", "loralab.device": "cpu"})
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    recipe["output"]["sha256"] = digest
    save_recipe(data_dir, recipe)
    return {
        "path": str(output_path),
        "sha256": digest,
        "device": "cpu",
        "executed": True,
        "deleted": [],
        "keys": sorted(keys),
    }
