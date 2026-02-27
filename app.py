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
st.caption("Upload an image → add Gaussian noise → denoise with Mean & NLM → compare MSE/PSNR + runtime.")

# ---------------- Input ----------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📥 Upload image")
    uploaded = st.file_uploader("Choose an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

    # Optional sample (NOT default)
    use_sample = st.checkbox("Use sample image (baboon.png) if no upload", value=False)

image_path: Path | None = None

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower() or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    image_path = Path(tmp.name)
elif use_sample and Path("baboon.png").exists():
    image_path = Path("baboon.png")

if image_path is None:
    st.info("⬅️ Please upload an image (or enable the sample checkbox).")
    st.stop()

# Show original preview immediately (before running)
loaded = load_grayscale(image_path)
original = loaded.gray_u8

with left:
    st.image(original, caption="Original (Gray preview)", clamp=True)

# ---------------- Controls + Run Button (Form) ----------------
with st.sidebar:
    st.header("⚙️ Parameters")

    with st.form("params_form", clear_on_submit=False):
        seed = st.number_input("Noise seed", value=0, step=1)
        noise_mean = st.number_input("Gaussian mean", value=0.0, step=0.1)
        noise_var = st.number_input("Gaussian variance", value=5.0, step=1.0, min_value=0.0)

        st.divider()
        mean_kernel = st.selectbox("Mean filter kernel (odd)", options=[3, 5, 7], index=0)

        st.divider()
        nlm_h = st.slider("NLM h (strength)", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
        nlm_template = st.selectbox("NLM template window (odd)", options=[3, 5, 7, 9], index=2)
        nlm_search = st.selectbox("NLM search window (odd)", options=[11, 15, 21, 31], index=2)

        run_clicked = st.form_submit_button("🚀 Run")

# ---------------- Run pipeline only when button clicked ----------------
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if run_clicked:
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

    st.session_state["last_result"] = {
        "noisy": noisy,
        "mean": mean_out,
        "nlm": nlm_out,
        "metrics": metrics,
    }

# ---------------- Display ----------------
result = st.session_state["last_result"]

with right:
    st.subheader("📊 Results")
    if result is None:
        st.warning("Adjust parameters in the sidebar and click **Run** to generate results.")
        st.stop()

    st.json(result["metrics"], expanded=False)

    c1, c2, c3 = st.columns(3)
    c1.image(result["noisy"], caption="Noisy (Gaussian)", clamp=True)
    c2.image(result["mean"], caption=f"Mean {mean_kernel}×{mean_kernel}", clamp=True)
    c3.image(result["nlm"], caption="NLM (OpenCV)", clamp=True)

    st.divider()
    st.subheader("⬇️ Download metrics.json")
    metrics_bytes = json.dumps(result["metrics"], indent=2).encode("utf-8")
    st.download_button(
        label="Download metrics.json",
        data=metrics_bytes,
        file_name="metrics.json",
        mime="application/json",
    )