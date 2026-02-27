from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NoiseParams:
    mean: float = 0.0
    var: float = 5.0
    seed: int | None = None


def add_gaussian_noise_u8(image_u8: np.ndarray, params: NoiseParams) -> np.ndarray:
    if image_u8.dtype != np.uint8:
        raise TypeError("image_u8 must be uint8")

    rng = np.random.default_rng(params.seed)
    sigma = float(np.sqrt(params.var))
    noise = rng.normal(loc=params.mean, scale=sigma, size=image_u8.shape).astype(np.float32)

    noisy = image_u8.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy
