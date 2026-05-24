# =============================================================================
# EasyGen Pro — Background Replacement Backend (Google Colab)
# =============================================================================
# Runtime: GPU (T4 recommended)
# Web app sends: image, mask (subject painted white), mode=bg, prompt, mask_blur
# Paste the ngrok URL into app.html → BASE_URL_BG
# =============================================================================

# ── CELL 1: Install (run once, then Runtime → Restart session) ─────────────

import subprocess, sys

subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y",
     "numpy", "opencv-python", "opencv-python-headless", "simple-lama-inpainting"],
    capture_output=True,
)

!pip install -q "numpy>=1.24,<2.1"
!pip install -q opencv-python-headless
!pip install -q simple-lama-inpainting
!pip install -q flask flask-cors pyngrok pillow

import numpy as np
import cv2
import torch
print("numpy:", np.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Restart runtime, then run cells below.")
"""

# ── CELL 2: ngrok auth token ───────────────────────────────────────────────
"""
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")  # https://dashboard.ngrok.com/get-started/your-authtoken
print("ngrok token set")
"""

# ── CELL 3: Load LaMa model ───────────────────────────────────────────────
"""
import io
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pyngrok import ngrok
from simple_lama_inpainting import SimpleLama

print("Loading LaMa (~200 MB)...")
lama = SimpleLama()
device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"LaMa ready on {device}")


# ── CELL 4: Mask helpers + Flask API ──────────────────────────────────────

def prepare_lama_mask(image_pil, mask_pil, blur_radius=4):
    White = inpaint, black = keep (object removal / custom fill).
    mask_aligned = mask_pil.resize(image_pil.size, Image.LANCZOS).convert("L")
    mask_arr = np.array(mask_aligned)
    mask_arr = (mask_arr > 128).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_arr = cv2.dilate(mask_arr, kernel, iterations=2)
    if blur_radius > 0:
        k = blur_radius * 2 + 1
        mask_arr = cv2.GaussianBlur(mask_arr, (k, k), blur_radius)
        mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_arr)


def prepare_background_mask(image_pil, mask_pil, blur_radius=6):
    
    Background replace (mode=bg):
    User paints SUBJECT to KEEP (white on mask).
    We invert so white = background to fill.
    """
    mask_aligned = mask_pil.resize(image_pil.size, Image.LANCZOS).convert("L")
    mask_arr = np.array(mask_aligned)
    # Invert: painted subject (white) -> keep (black), rest -> fill (white)
    mask_arr = 255 - (mask_arr > 128).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
    if blur_radius > 0:
        k = blur_radius * 2 + 1
        mask_arr = cv2.GaussianBlur(mask_arr, (k, k), blur_radius)
    return Image.fromarray(mask_arr.astype(np.uint8))


app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "LaMa",
        "modes": ["bg", "remove", "custom"],
    })


@app.route("/api/inpaint", methods=["POST"])
def inpaint():
  
  Form fields (multipart):
    image      — original photo (file)
    mask       — user mask: white = painted subject to KEEP when mode=bg
    mode       — "bg" for background replace (default "remove")
    mask_blur  — 0–20 edge feather (default 6 for bg)
    prompt     — stored for logging / future SD backend (LaMa ignores it)

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400
        if "mask" not in request.files:
            return jsonify({"error": "No mask"}), 400

        mode = request.form.get("mode", "bg").strip().lower()
        mask_blur = max(0, min(20, int(request.form.get("mask_blur", 6))))
        prompt = request.form.get("prompt", "").strip()

        image_pil = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        mask_pil = Image.open(io.BytesIO(request.files["mask"].read())).convert("L")

        print(f"\nmode={mode} | size={image_pil.size} | blur={mask_blur} | prompt={prompt[:80]!r}")

        if mode == "bg":
            clean_mask = prepare_background_mask(image_pil, mask_pil, blur_radius=mask_blur)
        elif mode == "remove":
            clean_mask = prepare_lama_mask(image_pil, mask_pil, blur_radius=mask_blur)
        else:
            clean_mask = prepare_lama_mask(image_pil, mask_pil, blur_radius=mask_blur)

        result_pil = lama(image_pil, clean_mask)

        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        buf.seek(0)
        print("Done")
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def start_server(port=5001):
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    public_url = ngrok.connect(port).public_url
    print("=" * 60)
    print("Background backend LIVE:", public_url)
    print('app.html → const BASE_URL_BG = "' + public_url + '";')
    print("=" * 60)
    return public_url


# Colab: after defining everything above, run:
# start_server(5001)
