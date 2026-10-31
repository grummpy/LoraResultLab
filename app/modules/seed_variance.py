"""Seed-to-seed similarity per epoch, with an instability flag.

Auto comparison uses SSIM. CLIP runs only when a local model is configured.
"""

from __future__ import annotations

from pathlib import Path
from statistics import pstdev

from PIL import Image

from app.imaging import compare_images
from app.modules.epoch_picker import build_grid

DEFAULT_MEAN_FLOOR = 0.90
DEFAULT_STD_CEILING = 0.05


def _pair_scores(images: list[Image.Image], method: str) -> tuple[list[float], str]:
    scores = []
    used = "ssim"
    for index, left in enumerate(images):
        for right in images[index + 1 :]:
            score, used = compare_images(left, right, method=method)
            scores.append(score)
    return scores, used


def variance_report(
    sample_dir: Path,
    *,
    method: str = "auto",
    mean_floor: float = DEFAULT_MEAN_FLOOR,
    std_ceiling: float = DEFAULT_STD_CEILING,
) -> dict:
    grid = build_grid(sample_dir)
    by_epoch: dict[str, list] = {epoch: [] for epoch in grid["epochs"]}
    per_prompt = []
    used_method = "ssim"
    open_images: list[Image.Image] = []
    try:
        for cell in grid["cells"]:
            loaded = []
            for image in cell["images"]:
                handle = Image.open(sample_dir / image["path"])
                open_images.append(handle)
                loaded.append(handle.convert("RGB"))
            scores, used_method = _pair_scores(loaded, method) if len(loaded) >= 2 else ([], "ssim")
            by_epoch.setdefault(cell["epoch"], []).extend(scores)
            mean = sum(scores) / len(scores) if scores else 1.0
            spread = pstdev(scores) if scores else 0.0
            per_prompt.append(
                {
                    "epoch": cell["epoch"],
                    "prompt_id": cell["prompt_id"],
                    "mean": mean,
                    "std": spread,
                    "pairs": len(scores),
                }
            )
    finally:
        for handle in open_images:
            handle.close()

    epochs = []
    for epoch, scores in by_epoch.items():
        mean = sum(scores) / len(scores) if scores else 1.0
        spread = pstdev(scores) if scores else 0.0
        unstable = bool(scores) and (mean < mean_floor or spread > std_ceiling)
        epochs.append(
            {
                "epoch": epoch,
                "mean": mean,
                "std": spread,
                "low": mean - spread,
                "high": mean + spread,
                "pairs": len(scores),
                "unstable": unstable,
            }
        )
    return {
        "method": used_method,
        "mean_floor": mean_floor,
        "std_ceiling": std_ceiling,
        "epochs": epochs,
        "prompts": per_prompt,
        "unstable_epochs": [item["epoch"] for item in epochs if item["unstable"]],
    }
