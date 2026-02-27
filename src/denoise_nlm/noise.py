from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NoiseParams:
    mean: float = 0.0
    var: float = 5.0
    seed: int | None = None


def add_gaussian_noise_u8(img_u8: np.ndarray, params) -> np.ndarray:
    seed = int(params.seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer (>= 0)")
    if float(params.var) < 0:
        raise ValueError("variance must be non-negative (>= 0)")

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(float(params.var))

    noise = rng.normal(loc=float(params.mean), scale=sigma, size=img_u8.shape)
    out = img_u8.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out