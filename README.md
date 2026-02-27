# NLM Denoising (Spatial Domain) — Mean 3×3 vs Non-Local Means

This repo implements **Exercise 2** from the *“Chapter 3: Spatial Domain Image Enhancement”* handout:
- Convert `Baboon.jpg` to grayscale
- Add **Gaussian noise** with **mean = 0** and **variance = 5**
- Denoise using:
  1) simple **mean filter 3×3**
  2) **Non-Local Means (NLM)**
- Compare outputs by **visual quality**, **MSE**, **PSNR**, and **runtime**.

## What is Non-Local Means (NLM)?

Unlike local filters (mean/median) that only use a small neighborhood, **NLM** denoises a pixel by looking for **similar patches** across a larger **search window**.

For each pixel `p`, NLM computes a weighted average of pixels `q` in a search region:

- Extract two patches around `p` and `q` (same patch size)
- Measure similarity with the **squared Euclidean distance** between patches
- Convert distance to a weight (bigger weight → more similar):

`w(p,q) = exp( - ||Patch(p) - Patch(q)||² / h² )`

Then:

`NL(p) = Σ w(p,q) * I(q) / Σ w(p,q)`

Where:
- `patch_size` (OpenCV: `templateWindowSize`) controls how much texture/detail is compared
- `search_size` (OpenCV: `searchWindowSize`) controls how far we look for similar patches
- `h` controls denoising strength (bigger `h` → smoother, but more detail loss)

This repo uses OpenCV’s optimized implementation: `cv2.fastNlMeansDenoising`.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## Run (CLI)

```bash
python -m denoise_nlm.run --image path/to/Baboon.jpg --out outputs
```

Options (examples):

```bash
python -m denoise_nlm.run \
  --image Baboon.jpg \
  --noise-var 5 \
  --mean-kernel 3 \
  --nlm-h 10 \
  --nlm-template 7 \
  --nlm-search 21 \
  --seed 0 \
  --out outputs
```

## Outputs

The script will save:
- `original_gray.png`
- `noisy.png`
- `mean_3x3.png`
- `nlm.png`
- `metrics.json` (MSE/PSNR + timings)

## Notes

- Noise variance is interpreted in **8-bit intensity units** (0–255).  
  For variance = 5 → `sigma = sqrt(5) ≈ 2.236`.
- `PSNR` is computed with `MAX=255`.

## License

MIT
