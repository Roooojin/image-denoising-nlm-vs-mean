from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from denoise_nlm.io import load_grayscale
from denoise_nlm.noise import NoiseParams, add_gaussian_noise_u8
from denoise_nlm.benchmark import benchmark_methods
from denoise_nlm.metrics import mse_u8, psnr_u8

st.set_page_config(page_title="Image Denoising: Mean vs NLM", layout="wide")

st.title("🧼 Image Denoising — Mean 3×3 vs Non-Local Means (NLM)")
st.caption("Interactive GUI for adding Gaussian noise and comparing denoising methods with MSE/PSNR + runtime.")

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.header("⚙️ Parameters")

    seed = st.number_input("Noise seed", value=0, step=1)
    noise_mean = st.number_input("Gaussian mean", value=0.0, step=0.1)
    noise_var = st.number_input("Gaussian variance", value=5.0, step=1.0, min_value=0.0)

    st.divider()
    mean_kernel = st.selectbox("Mean filter kernel (odd)", options=[3, 5, 7], index=0)

    st.divider()
    nlm_h = st.slider("NLM h (strength)", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
    nlm_template = st.selectbox("NLM template window (odd)", options=[3, 5, 7, 9], index=2)
    nlm_search = st.selectbox("NLM search window (odd)", options=[11, 15, 21, 31], index=2)

# ---------------- Input ----------------
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("📥 Input image")
    uploaded = st.file_uploader("Upload an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
    use_sample = st.checkbox("Use local sample (baboon.png) if exists", value=True)

image_path: Path | None = None

if uploaded is not None:
    # Save uploaded file to a temp path (because your loader expects a file path)
    suffix = Path(uploaded.name).suffix.lower() or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    image_path = Path(tmp.name)
elif use_sample and Path("baboon.png").exists():
    image_path = Path("baboon.png")

if image_path is None:
    st.info("⬅️ Upload an image or enable the sample checkbox to use baboon.png")
    st.stop()

# ---------------- Run pipeline ----------------
loaded = load_grayscale(image_path)
original = loaded.gray_u8

noise_params = NoiseParams(mean=float(noise_mean), var=float(noise_var), seed=int(seed))
noisy = add_gaussian_noise_u8(original, noise_params)

bench = benchmark_methods(
    original_u8=original,
    noisy_u8=noisy,
    mean_k=int(mean_kernel),
    nlm_h=float(nlm_h),
    nlm_template=int(nlm_template),
    nlm_search=int(nlm_search),
)

mean_out = bench["images"]["mean"]
nlm_out = bench["images"]["nlm"]

metrics = {
    "input": {
        "image": str(loaded.path),
        "noise_mean": float(noise_mean),
        "noise_var": float(noise_var),
        "seed": int(seed),
    },
    "baseline_noisy": {
        "mse": mse_u8(original, noisy),
        "psnr": psnr_u8(original, noisy),
    },
    "methods": bench["results"],
    "params": {
        "mean_kernel": int(mean_kernel),
        "nlm_h": float(nlm_h),
        "nlm_template": int(nlm_template),
        "nlm_search": int(nlm_search),
    },
}

# ---------------- Display ----------------
with colB:
    st.subheader("📊 Metrics summary")
    st.json(metrics, expanded=False)

c1, c2, c3, c4 = st.columns(4)
c1.image(original, caption="Original (Gray)", clamp=True)
c2.image(noisy, caption="Noisy (Gaussian)", clamp=True)
c3.image(mean_out, caption=f"Mean {mean_kernel}×{mean_kernel}", clamp=True)
c4.image(nlm_out, caption="NLM (OpenCV)", clamp=True)

# ---------------- Download ----------------
st.divider()
st.subheader("⬇️ Download metrics.json")

metrics_bytes = json.dumps(metrics, indent=2).encode("utf-8")
st.download_button(
    label="Download metrics.json",
    data=metrics_bytes,
    file_name="metrics.json",
    mime="application/json",
)