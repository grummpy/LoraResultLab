"""Image similarity. Auto mode uses SSIM. CLIP is optional and never downloads weights."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def ssim_images(image_a: Image.Image, image_b: Image.Image, size: int = 128) -> float:
    """Global SSIM on grayscale images resized to a common square.

    Identical images score 1. The resize keeps comparisons deterministic when
    sample renders differ slightly in size.
    """
    x = np.asarray(image_a.convert("L").resize((size, size)), dtype=np.float64)
    y = np.asarray(image_b.convert("L").resize((size, size)), dtype=np.float64)
    level = 255.0
    c1 = (0.01 * level) ** 2
    c2 = (0.03 * level) ** 2
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(x.var())
    var_y = float(y.var())
    cov = float(((x - mu_x) * (y - mu_y)).mean())
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def clip_similarity(image_a: Image.Image, image_b: Image.Image) -> float | None:
    """CPU CLIP cosine similarity when a local model is configured.

    Returns None when the optional packages or LORALAB_CLIP_MODEL are absent.
    This never downloads weights: pretrained is not fetched from the network.
    """
    model_name = os.environ.get("LORALAB_CLIP_MODEL", "").strip()
    if not model_name:
        return None
    try:
        import open_clip
        import torch
    except ImportError:
        return None
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,
        device=device,
    )
    model.eval()

    def embed(image: Image.Image):
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(tensor.to(device))
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    a = embed(image_a)
    b = embed(image_b)
    return float((a * b).sum().item())


def compare_images(image_a: Image.Image, image_b: Image.Image, method: str = "auto") -> tuple[float, str]:
    chosen = (method or "auto").lower()
    if chosen not in {"auto", "ssim", "clip"}:
        raise ValueError("method must be auto, ssim, or clip")
    if chosen in {"auto", "clip"}:
        score = clip_similarity(image_a, image_b)
        if score is not None:
            return score, "clip"
        if chosen == "clip":
            raise RuntimeError(
                "CLIP is not configured. Set LORALAB_CLIP_MODEL and install "
                "open_clip_torch and torch locally, or use SSIM."
            )
    return ssim_images(image_a, image_b), "ssim"


def load_image(path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
