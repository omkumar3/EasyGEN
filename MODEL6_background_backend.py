# EasyGen MODEL6 — copy each "# CELL" block into Google Colab (T4 GPU)
# Matches app.html Change Background (mode=bg, prompt, mask, mask_blur)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install → then Runtime → Restart session
# ══════════════════════════════════════════════════════════════════════════════
# !pip install -q diffusers transformers accelerate safetensors
# !pip install -q opencv-python-headless flask flask-cors pyngrok pillow

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — ngrok
# ══════════════════════════════════════════════════════════════════════════════
# from pyngrok import ngrok
# ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Load SD Inpainting (prompt-based backgrounds)
# ══════════════════════════════════════════════════════════════════════════════
import io
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pyngrok import ngrok

SD_MODEL = "runwayml/stable-diffusion-inpainting"

print(f"Loading {SD_MODEL}...")
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
sd_pipe = sd_pipe.to("cuda")
sd_pipe.enable_attention_slicing()
print("SD ready on GPU")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Helpers
# ══════════════════════════════════════════════════════════════════════════════

def prepare_background_mask(image_pil, mask_pil, blur_radius=8):
    m = mask_pil.resize(image_pil.size, Image.LANCZOS).convert("L")
    arr = np.array(m)
    arr = 255 - ((arr > 128).astype(np.uint8) * 255)
    arr = cv2.dilate(arr, np.ones((5, 5), np.uint8), iterations=2)
    if blur_radius > 0:
        k = blur_radius * 2 + 1
        arr = cv2.GaussianBlur(arr, (k, k), blur_radius)
    return Image.fromarray(arr.astype(np.uint8))


def run_sd_background(image_pil, mask_pil, prompt, steps=35, guidance=7.5, blur=8):
    mask = prepare_background_mask(image_pil, mask_pil, blur_radius=blur)
    w, h = image_pil.size
    scale = min(1.0, 768 / max(w, h))
    nw = max(64, int(w * scale) // 8 * 8)
    nh = max(64, int(h * scale) // 8 * 8)
    img = image_pil.resize((nw, nh), Image.LANCZOS)
    msk = mask.resize((nw, nh), Image.LANCZOS)
    full_prompt = prompt.strip()
    if "background" not in full_prompt.lower():
        full_prompt += ", detailed background, photorealistic, 8k"
    neg = "blurry, low quality, distorted, watermark, text, bad anatomy"
    with torch.inference_mode():
        out = sd_pipe(
            prompt=full_prompt,
            negative_prompt=neg,
            image=img,
            mask_image=msk,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
        ).images[0]
    return out.resize(image_pil.size, Image.LANCZOS)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Flask + ngrok
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": SD_MODEL, "prompt_based": True})


@app.route("/api/inpaint", methods=["POST"])
def inpaint():
    try:
        if "image" not in request.files or "mask" not in request.files:
            return jsonify({"error": "Need image and mask"}), 400
        mode = request.form.get("mode", "bg")
        prompt = request.form.get("prompt", "").strip()
        mask_blur = max(0, min(20, int(request.form.get("mask_blur", 8))))
        steps = max(15, min(60, int(request.form.get("num_steps", 35))))
        guidance = max(1.0, min(20.0, float(request.form.get("guidance_scale", 7.5))))
        if mode != "bg":
            return jsonify({"error": "Use mode=bg"}), 400
        if not prompt:
            return jsonify({"error": "Prompt required"}), 400
        image_pil = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        mask_pil = Image.open(io.BytesIO(request.files["mask"].read())).convert("L")
        print(f"bg {image_pil.size} prompt={prompt[:60]!r}")
        result = run_sd_background(image_pil, mask_pil, prompt, steps=steps, guidance=guidance, blur=mask_blur)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    url = ngrok.connect(5001).public_url
    print("LIVE:", url)
    print(f'BASE_URL_BG = "{url}"')
