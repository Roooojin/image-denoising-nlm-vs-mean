from __future__ import annotations

import cv2
import numpy as np


def mean_filter_u8(image_u8: np.ndarray, k: int = 3) -> np.ndarray:
    if k % 2 == 0 or k < 1:
        raise ValueError("k must be a positive odd integer")
    if image_u8.dtype != np.uint8:
        raise TypeError("image_u8 must be uint8")

    kernel = np.ones((k, k), dtype=np.float32) / float(k * k)
    # Reflect padding behaves well for natural images
    out = cv2.filter2D(image_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    return out


def nlm_denoise_opencv_u8(
    image_u8: np.ndarray,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    if image_u8.dtype != np.uint8:
        raise TypeError("image_u8 must be uint8")
    if template_window_size % 2 == 0 or search_window_size % 2 == 0:
        raise ValueError("template_window_size and search_window_size must be odd")

    return cv2.fastNlMeansDenoising(
        image_u8,
        None,
        h,
        template_window_size,
        search_window_size,
    )
