from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark_methods
from .io import load_grayscale, save_u8
from .noise import NoiseParams, add_gaussian_noise_u8
from .metrics import mse_u8, psnr_u8


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mean 3x3 vs Non-Local Means (NLM) denoising with MSE/PSNR/runtime benchmarking."
    )
    p.add_argument("--image", required=True, help="Path to input image (e.g., Baboon.jpg)")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible noise")

    p.add_argument("--noise-mean", type=float, default=0.0, help="Gaussian noise mean")
    p.add_argument("--noise-var", type=float, default=5.0, help="Gaussian noise variance")

    p.add_argument("--mean-kernel", type=int, default=3, help="Mean filter kernel size (odd)")
    p.add_argument("--nlm-h", type=float, default=10.0, help="NLM filter strength parameter h")
    p.add_argument("--nlm-template", type=int, default=7, help="NLM template window size (odd)")
    p.add_argument("--nlm-search", type=int, default=21, help="NLM search window size (odd)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_grayscale(args.image)
    original = loaded.gray_u8

    noise_params = NoiseParams(mean=args.noise_mean, var=args.noise_var, seed=args.seed)
    noisy = add_gaussian_noise_u8(original, noise_params)

    bench = benchmark_methods(
        original_u8=original,
        noisy_u8=noisy,
        mean_k=args.mean_kernel,
        nlm_h=args.nlm_h,
        nlm_template=args.nlm_template,
        nlm_search=args.nlm_search,
    )

    mean_out = bench["images"]["mean"]
    nlm_out = bench["images"]["nlm"]

    # Save images
    save_u8(out_dir / "original_gray.png", original)
    save_u8(out_dir / "noisy.png", noisy)
    save_u8(out_dir / f"mean_{args.mean_kernel}x{args.mean_kernel}.png", mean_out)
    save_u8(out_dir / "nlm.png", nlm_out)

    # Extra: metrics for noisy too (baseline)
    metrics = {
        "input": {
            "image": str(loaded.path),
            "noise_mean": args.noise_mean,
            "noise_var": args.noise_var,
            "seed": args.seed,
        },
        "baseline_noisy": {
            "mse": mse_u8(original, noisy),
            "psnr": psnr_u8(original, noisy),
        },
        "methods": bench["results"],
        "params": {
            "mean_kernel": args.mean_kernel,
            "nlm_h": args.nlm_h,
            "nlm_template": args.nlm_template,
            "nlm_search": args.nlm_search,
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Print summary
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
