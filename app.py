from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import streamlit as st

from denoise_nlm.io import load_grayscale
from denoise_nlm.noise import NoiseParams, add_gaussian_noise_u8
from denoise_nlm.benchmark import benchmark_methods
from denoise_nlm.metrics import mse_u8, psnr_u8


def to_png_bytes(img_u8):
    ok, buf = cv2.imencode(".png", img_u8)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG")
    return buf.tobytes()


def _get_time_ms(method_dict: Dict[str, Any]) -> Optional[float]:
    for k in ("time_ms", "runtime_ms", "elapsed_ms"):
        if k in method_dict and method_dict[k] is not None:
            return float(method_dict[k])

    for k in ("time_sec", "runtime_sec", "elapsed_sec"):
        if k in method_dict and method_dict[k] is not None:
            return float(method_dict[k]) * 1000.0

    return None


def fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.2f} ms"


st.set_page_config(page_title="Image Denoising: Mean vs NLM", layout="wide")
st.title("🧼 Image Denoising — Mean 3×3 vs Non-Local Means (NLM)")
st.caption("Upload an image → set parameters → click Run → compare Mean vs NLM (MSE/PSNR + runtime)")

left, right = st.columns([1, 1])

# ---------------- Input ----------------
with left:
    st.subheader("📥 Upload image")
    uploaded = st.file_uploader("Choose an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
    use_sample = st.checkbox("Use sample image (baboon.png) if no upload", value=False)

image_path: Optional[Path] = None

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower() or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    image_path = Path(tmp.name)
elif use_sample and Path("baboon.png").exists():
    image_path = Path("baboon.png")

if image_path is None:
    st.info("⬅️ Upload an image (or enable sample checkbox).")
    st.stop()

loaded = load_grayscale(image_path)
original = loaded.gray_u8

with left:
    st.image(original, caption="Original (Grayscale preview)", clamp=True)

# ---------------- Controls (Form) ----------------
with st.sidebar:
    st.header("⚙️ Parameters")

    with st.form("params_form", clear_on_submit=False):
        seed = st.number_input("Noise seed (>=0)", min_value=0, value=0, step=1)
        noise_mean = st.number_input("Gaussian mean", value=0.0, step=0.1)
        noise_var = st.number_input("Gaussian variance (>=0)", min_value=0.0, value=5.0, step=1.0)

        st.divider()
        mean_kernel = st.selectbox("Mean filter kernel (odd)", options=[3, 5, 7], index=0)

        st.divider()
        nlm_h = st.slider("NLM h (strength)", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
        nlm_template = st.selectbox("NLM template window (odd)", options=[3, 5, 7, 9], index=2)
        nlm_search = st.selectbox("NLM search window (odd)", options=[11, 15, 21, 31], index=2)

        run_clicked = st.form_submit_button("🚀 Run")

# ---------------- Run only on click ----------------
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if run_clicked:
    with st.spinner("Running denoising pipeline..."):
        try:
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
                "methods": bench["results"],  # expects mean/nlm metrics + runtime keys
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

        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()

# ---------------- Display ----------------
result = st.session_state["last_result"]

with right:
    st.subheader("📊 Results")

    if result is None:
        st.warning("Set parameters in the sidebar and click **Run** to generate results.")
        st.stop()

    methods_raw = result["metrics"].get("methods", {})

    # Normalize methods to a dict { "mean": {...}, "nlm": {...} }
    if isinstance(methods_raw, list):
        methods = {}
        for item in methods_raw:
            if not isinstance(item, dict):
                continue
            key = (
                    item.get("name")
                    or item.get("method")
                    or item.get("algo")
                    or item.get("filter")
            )
            if key is None:
                continue
            methods[str(key).lower()] = item

        # fallback: if list has 2 dicts and no name keys, map by order
        if not methods and len(methods_raw) >= 2 and all(isinstance(x, dict) for x in methods_raw[:2]):
            methods = {"mean": methods_raw[0], "nlm": methods_raw[1]}

    elif isinstance(methods_raw, dict):
        # make keys lowercase for easier matching
        methods = {str(k).lower(): v for k, v in methods_raw.items()}
    else:
        methods = {}


    # pick mean/nlm robustly even if keys are like "mean_3x3" or "fastnlmeans"
    def pick(methods_dict, target: str):
        if target in methods_dict:
            return methods_dict[target]
        # try fuzzy match
        for k, v in methods_dict.items():
            if target == "mean" and "mean" in k:
                return v
            if target == "nlm" and ("nlm" in k or "non" in k or "fast" in k):
                return v
        return {}


    mean_info = pick(methods, "mean")
    nlm_info = pick(methods, "nlm")
    # Runtime cards
    t1, t2 = st.columns(2)
    t1.metric("Mean runtime", fmt_ms(_get_time_ms(mean_info)))
    t2.metric("NLM runtime", fmt_ms(_get_time_ms(nlm_info)))

    # Quality cards (optional but nice)
    q1, q2, q3 = st.columns(3)
    q1.metric("Noisy PSNR", f'{result["metrics"]["baseline_noisy"]["psnr"]:.2f} dB')
    if "psnr" in mean_info:
        q2.metric("Mean PSNR", f'{float(mean_info["psnr"]):.2f} dB')
    else:
        q2.metric("Mean PSNR", "N/A")

    if "psnr" in nlm_info:
        q3.metric("NLM PSNR", f'{float(nlm_info["psnr"]):.2f} dB')
    else:
        q3.metric("NLM PSNR", "N/A")

    st.divider()
    st.json(result["metrics"], expanded=False)

    # Images
    c1, c2, c3 = st.columns(3)
    c1.image(result["noisy"], caption="Noisy", clamp=True)
    c2.image(result["mean"],
             caption=f"Mean {int(result['metrics']['params']['mean_kernel'])}×{int(result['metrics']['params']['mean_kernel'])}",
             clamp=True)
    c3.image(result["nlm"], caption="NLM", clamp=True)

    # Downloads
    st.divider()
    st.subheader("⬇️ Download outputs")

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "noisy.png",
        data=to_png_bytes(result["noisy"]),
        file_name="noisy.png",
        mime="image/png",
    )
    d2.download_button(
        "mean.png",
        data=to_png_bytes(result["mean"]),
        file_name="mean.png",
        mime="image/png",
    )
    d3.download_button(
        "nlm.png",
        data=to_png_bytes(result["nlm"]),
        file_name="nlm.png",
        mime="image/png",
    )

    metrics_bytes = json.dumps(result["metrics"], indent=2).encode("utf-8")
    d4.download_button(
        "metrics.json",
        data=metrics_bytes,
        file_name="metrics.json",
        mime="application/json",
    )
