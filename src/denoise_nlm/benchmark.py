from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Dict, Any

import numpy as np

from .filters import mean_filter_u8, nlm_denoise_opencv_u8
from .metrics import mse_u8, psnr_u8


@dataclass
class MethodResult:
    name: str
    runtime_sec: float
    mse: float
    psnr: float


def benchmark_methods(
    original_u8: np.ndarray,
    noisy_u8: np.ndarray,
    mean_k: int = 3,
    nlm_h: float = 10.0,
    nlm_template: int = 7,
    nlm_search: int = 21,
) -> Dict[str, Any]:
    results: list[MethodResult] = []

    # Mean filter
    t0 = perf_counter()
    mean_out = mean_filter_u8(noisy_u8, k=mean_k)
    t1 = perf_counter()
    results.append(
        MethodResult(
            name=f"mean_{mean_k}x{mean_k}",
            runtime_sec=t1 - t0,
            mse=mse_u8(original_u8, mean_out),
            psnr=psnr_u8(original_u8, mean_out),
        )
    )

    # NLM
    t0 = perf_counter()
    nlm_out = nlm_denoise_opencv_u8(
        noisy_u8,
        h=nlm_h,
        template_window_size=nlm_template,
        search_window_size=nlm_search,
    )
    t1 = perf_counter()
    results.append(
        MethodResult(
            name="nlm",
            runtime_sec=t1 - t0,
            mse=mse_u8(original_u8, nlm_out),
            psnr=psnr_u8(original_u8, nlm_out),
        )
    )

    return {
        "results": [asdict(r) for r in results],
        "images": {"mean": mean_out, "nlm": nlm_out},
    }
