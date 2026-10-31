"""Compare with-LoRA and without-LoRA renders on unrelated prompts.

Leakage is 1 - similarity. A high score means the LoRA moved an unrelated prompt.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.imaging import compare_images
from app.modules.epoch_picker import prompt_text_map

DEFAULT_LEAKAGE_CEILING = 0.20


def _find_pair(folder: Path) -> tuple[Path | None, Path | None]:
    with_dir = folder / "with_lora"
    without_dir = folder / "without_lora"
    def first_png(path: Path) -> Path | None:
        if not path.is_dir():
            return None
        files = sorted(item for item in path.iterdir() if item.suffix.lower() == ".png")
        return files[0] if files else None
    return first_png(with_dir), first_png(without_dir)


def bleed_report(
    sample_dir: Path,
    *,
    method: str = "auto",
    leakage_ceiling: float = DEFAULT_LEAKAGE_CEILING,
) -> dict:
    root = sample_dir / "bleed"
    texts = prompt_text_map(sample_dir)
    epochs = []
    used_method = "ssim"
    if root.is_dir():
        for epoch_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name):
            pairs = []
            for prompt_dir in sorted((path for path in epoch_dir.iterdir() if path.is_dir()), key=lambda p: p.name):
                with_path, without_path = _find_pair(prompt_dir)
                if with_path is None or without_path is None:
                    continue
                with Image.open(with_path) as left, Image.open(without_path) as right:
                    similarity, used_method = compare_images(left.convert("RGB"), right.convert("RGB"), method=method)
                leakage = 1.0 - similarity
                pairs.append(
                    {
                        "prompt_id": prompt_dir.name,
                        "text": texts.get(prompt_dir.name, prompt_dir.name),
                        "similarity": similarity,
                        "leakage": leakage,
                        "with_lora": with_path.resolve().relative_to(sample_dir.resolve()).as_posix(),
                        "without_lora": without_path.resolve().relative_to(sample_dir.resolve()).as_posix(),
                    }
                )
            scores = [pair["leakage"] for pair in pairs]
            mean_leakage = sum(scores) / len(scores) if scores else 0.0
            epochs.append(
                {
                    "epoch": epoch_dir.name,
                    "leakage": mean_leakage,
                    "flagged": bool(scores) and mean_leakage > leakage_ceiling,
                    "pairs": pairs,
                }
            )
    return {
        "method": used_method,
        "leakage_ceiling": leakage_ceiling,
        "epochs": epochs,
        "flagged_epochs": [item["epoch"] for item in epochs if item["flagged"]],
    }
