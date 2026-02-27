from __future__ import annotations

import numpy as np


def mse_u8(ref_u8: np.ndarray, test_u8: np.ndarray) -> float:
    if ref_u8.shape != test_u8.shape:
        raise ValueError("Images must have the same shape")
    ref = ref_u8.astype(np.float32)
    test = test_u8.astype(np.float32)
    return float(np.mean((ref - test) ** 2))


def psnr_u8(ref_u8: np.ndarray, test_u8: np.ndarray, max_val: float = 255.0) -> float:
    m = mse_u8(ref_u8, test_u8)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val * max_val) / m))
