from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LoadedImage:
    path: Path
    gray_u8: np.ndarray  # shape (H, W), dtype uint8


def load_grayscale(path: str | Path) -> LoadedImage:
    """Load an image from disk and convert to 8-bit grayscale."""
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return LoadedImage(path=p, gray_u8=gray)


def save_u8(path: str | Path, image_u8: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), image_u8)
    if not ok:
        raise IOError(f"Failed to write image: {p}")
