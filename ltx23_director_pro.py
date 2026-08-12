# -*- coding: utf-8 -*-
"""
LTX-2.3 Director 2.0 — Infinite Flow PRO v4.0
================================================
Upgraded from ltx2_ti2v_distilled.py to match the full
LTX-2.3_Director_2.0-MV-Workflow-30s.json ComfyUI pipeline.

Key upgrades vs. v3.0:
  • LTX-2.3 22B Q4_K_M GGUF  (was 19B distilled)
  • Gemma 3 12B fp8-scaled  +  ltx-2.3 text projection bf16
  • LTX23 Video VAE bf16  +  LTX23 Audio VAE bf16
  • Tiny TAE preview VAE (taeltx2_3)
  • ltx-2.3 spatial upscaler x2 v1.1
  • 4-LoRA stack  (distilled-dynamic · OmniNFT-RL · transition · MVCamera-drclips)
  • Director 2.0 2-pass pipeline:
        Pass 1 → BasicScheduler linear_quadratic  8 steps  denoise=1.0
        Pass 2 → BasicScheduler linear_quadratic  4 steps  denoise=0.42
  • LTXVConcatAVLatent / LTXVSeparateAVLatent  (joint AV sampling)
  • LTXDirectorCropGuides  (both passes)
  • LTXVLatentUpsampler  (2× between passes)
  • VHS_VideoCombine  (H.264 MP4, CRF 8, 24 fps)
  • Multi-image timeline segments  (mirrors LTXDirector node)
  • 1280×720 HD output  (was 848×480)
  • CFG = 1  (distilled model, no negative subtraction)
  • Cinematic music-video global prompt style
  • All previous v3.0 features preserved and upgraded

Google Colab: Run all cells top-to-bottom.
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1 ─── Environment Setup & Model Downloads
# ═══════════════════════════════════════════════════════════════


# @title {"single-column":true}
# @markdown ## 🚀 Step 1 — Install dependencies & clone ComfyUI

import subprocess, os, sys
from pathlib import Path
from IPython.display import clear_output

# Core torch stack
subprocess.run(["pip", "install", "-q", "torch", "torchvision", "torchaudio"], check=True)

os.chdir("/content")
subprocess.run(["pip", "install", "-q",
    "torchsde", "einops", "diffusers", "accelerate",
    "av", "spandrel", "albumentations", "onnx",
    "opencv-python", "onnxruntime", "nest_asyncio",
    "imageio", "imageio-ffmpeg", "moviepy", "tqdm",
    "ipywidgets", "requests"
], check=True)

# FIX: pin kornia to 0.7.3 — newer kornia removed 'pad' from pyramid module
# which breaks ComfyUI-LTXVideo's pyramid_blending.py
subprocess.run(["pip", "install", "-q", "kornia==0.7.3"], check=True)

# ComfyUI fork pinned to a stable LTX-2.3 compatible version
if not os.path.exists("/content/ComfyUI"):
    subprocess.run([
        "git", "clone", "--branch", "ComfyUI_22_01_2026_v0.10.0",
        "https://github.com/Isi-dev/ComfyUI.git"
    ], check=True)
subprocess.run(["pip", "install", "-q", "-r", "/content/ComfyUI/requirements.txt"], check=True)

os.chdir("/content/ComfyUI/custom_nodes")

# KJNodes (VAELoaderKJ, ResizeImagesByLongerEdge, etc.)
if not os.path.exists("ComfyUI_KJNodes"):
    subprocess.run(["git", "clone", "--branch", "kj_1.2.6",
        "https://github.com/Isi-dev/ComfyUI_KJNodes"], check=True)
subprocess.run(["pip", "install", "-q", "-r", "ComfyUI_KJNodes/requirements.txt"], check=True)

# GGUF loader (UnetLoaderGGUF)
if not os.path.exists("ComfyUI_GGUF"):
    subprocess.run(["git", "clone", "--branch", "ComfyUI_GGUF_22_01_2026",
        "https://github.com/Isi-dev/ComfyUI_GGUF.git"], check=True)
subprocess.run(["pip", "install", "-q", "-r", "ComfyUI_GGUF/requirements.txt"], check=True)

# VideoHelperSuite (VHS_VideoCombine)
# NOTE: clone BEFORE sys.path manipulation to avoid utils.install_util conflict
if not os.path.exists("ComfyUI-VideoHelperSuite"):
    subprocess.run(["git", "clone",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"], check=True)
subprocess.run(["pip", "install", "-q", "-r",
    "ComfyUI-VideoHelperSuite/requirements.txt"], check=True)

# WhatDreamsCost Director 2.0 nodes (LTXDirectorCropGuides etc.)
if not os.path.exists("WhatDreamsCost-ComfyUI"):
    subprocess.run(["git", "clone",
        "https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git"], check=True)

# ComfyUI-LTXVideo (LTXVConditioning, LTXVSeparateAVLatent, etc.)
if not os.path.exists("ComfyUI-LTXVideo"):
    subprocess.run(["git", "clone",
        "https://github.com/Lightricks/ComfyUI-LTXVideo.git"], check=True)
subprocess.run(["pip", "install", "-q", "-r",
    "ComfyUI-LTXVideo/requirements.txt"], check=True)

# apt packages
subprocess.run(["apt-get", "-y", "install", "-qq", "aria2", "ffmpeg"],
               check=True, capture_output=True)

clear_output()
print("✅ Dependencies installed.")


# ─── Python imports ──────────────────────────────────────────────────────────
os.chdir("/content/ComfyUI")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# FIX: sys.path MUST be set BEFORE any ComfyUI import.
# Insert at position 0 so ComfyUI's own modules are found first, but
# critically do NOT let it shadow stdlib 'utils'.
# We remove the ComfyUI path from sys.path temporarily while importing server-
# dependent custom nodes that trigger `from server import ...` → `from app.
# frontend_management import ...` → `from utils.install_util import ...`
# The clean fix is to patch utils.install_util as a stub before the import.
if "/content/ComfyUI" not in sys.path:
    sys.path.insert(0, "/content/ComfyUI")

# Stub out utils.install_util BEFORE importing ComfyUI nodes.
# This prevents the "No module named 'utils.install_util'" error that occurs
# when ComfyUI server.py is imported headlessly (without the full server running).
import types as _types
_utils_pkg = _types.ModuleType("utils")
_utils_pkg.__path__ = ["/content/ComfyUI/utils"]
_utils_pkg.__package__ = "utils"
sys.modules.setdefault("utils", _utils_pkg)

_install_util = _types.ModuleType("utils.install_util")
_install_util.get_missing_requirements_message = lambda *a, **kw: ""
_install_util.requirements_path = ""
sys.modules["utils.install_util"] = _install_util
sys.modules["utils.extra_config"] = _types.ModuleType("utils.extra_config")

# Stub PromptServer so custom nodes that do `from server import PromptServer`
# don't crash when running headlessly.
try:
    import server as _server_mod
except Exception:
    _server_mod = _types.ModuleType("server")
    class _FakePS:
        instance = None
        @classmethod
        def send_sync(cls, *a, **kw): pass
    _server_mod.PromptServer = _FakePS
    sys.modules["server"] = _server_mod

# ─── Standard library & third-party imports ──────────────────────────────────
import gc, shutil, time, warnings, traceback, threading, concurrent.futures
import asyncio, nest_asyncio
import numpy as np
import cv2
import torch
import json
import requests
import ipywidgets as widgets
from PIL import Image
from typing import Optional, List, Dict, Tuple, Any, Union, Sequence, Mapping
from functools import lru_cache
from tqdm.notebook import tqdm
from base64 import b64encode
from IPython.display import display, HTML
from google.colab import files
warnings.filterwarnings("ignore")

# ─── ComfyUI bootstrap ───────────────────────────────────────────────────────
from nodes import NODE_CLASS_MAPPINGS

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def import_custom_nodes() -> None:
    """Jupyter/Colab-safe ComfyUI node loader."""
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes
    async def _load():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print("⚠️ Some nodes failed:", [str(f) for f in failed])
    try:
        asyncio.run(_load())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_load())

import_custom_nodes()

import folder_paths
try:
    from comfy_api.latest import Types
except ImportError:
    Types = None

print("✅ ComfyUI nodes loaded.")


# ─── aria2c download helper ───────────────────────────────────────────────────
def model_download(url: str, dest_dir: str, filename: str = None) -> str:
    """Fast parallel download via aria2c (16 connections)."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    cmd = [
        "aria2c", "--console-log-level=error", "-c",
        "-x", "16", "-s", "16", "-k", "1M",
        "--summary-interval=0", "--quiet",
        "-d", dest_dir, "-o", filename, url
    ]
    print(f"  ⬇️  {filename} ...", end=" ", flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓")
    except subprocess.CalledProcessError as e:
        print(f"✗  {e.stderr.strip()[:120]}")
        return None
    return filename

# ─── MODEL DOWNLOADS (LTX-2.3 upgrade) ──────────────────────────────────────
# @title {"single-column":true}
# @markdown ## 📥 Step 2 — Download LTX-2.3 Models

print("=" * 60)
print("📥  LTX-2.3 Model Downloads")
print("=" * 60)

# ── UNet: 22B GGUF Q4_K_M ────────────────────────────────────────────────────
# Primary: unsloth (confirmed working filename from Resgard README)
# Fallback: Viral2AI mirror
UNET_MODEL = (
    model_download(
        "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/ltx-2.3-22b-dev-Q4_K_M.gguf",
        "/content/ComfyUI/models/unet",
        "ltx-2.3-22b-dev-Q4_K_M.gguf"
    ) or
    model_download(
        "https://huggingface.co/Viral2AI/LTX-2.3-GGUF/resolve/main/ltx-2.3-22b-dev-Q4_K_M.gguf",
        "/content/ComfyUI/models/unet",
        "ltx-2.3-22b-dev-Q4_K_M.gguf"
    ) or "ltx-2.3-22b-dev-Q4_K_M.gguf"   # name only — user must supply manually
)

# ── Text encoder 1: Gemma 3 12B fp4 mixed ────────────────────────────────────
# Source: Comfy-Org/ltx-2 (confirmed path from RuneXX/LTX-2.3-Workflows snippet)
TEXT_ENC1 = (
    model_download(
        "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        "/content/ComfyUI/models/text_encoders",
        "gemma_3_12B_it_fp4_mixed.safetensors"
    ) or "gemma_3_12B_it_fp4_mixed.safetensors"
)

# ── Text encoder 2: LTX-2.3 text projection bf16 ─────────────────────────────
# Source: Kijai/LTX2.3_comfy — confirmed filename from TensorVizion snippet
TEXT_ENC2 = (
    model_download(
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "/content/ComfyUI/models/text_encoders",
        "ltx-2.3_text_projection_bf16.safetensors"
    ) or "ltx-2.3_text_projection_bf16.safetensors"
)

# ── Video VAE bf16 (1.45 GB) ─────────────────────────────────────────────────
VIDEO_VAE = (
    model_download(
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
        "/content/ComfyUI/models/vae"
    ) or "LTX23_video_vae_bf16.safetensors"
)

# ── Audio VAE bf16 (347 MB) ──────────────────────────────────────────────────
AUDIO_VAE = (
    model_download(
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
        "/content/ComfyUI/models/vae"
    ) or "LTX23_audio_vae_bf16.safetensors"
)

# ── Tiny preview VAE (23 MB) ─────────────────────────────────────────────────
TINY_VAE = (
    model_download(
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
        "/content/ComfyUI/models/vae"
    ) or "taeltx2_3.safetensors"
)

# ── Spatial upscaler x2 v1.1 ─────────────────────────────────────────────────
# Primary: Lightricks official  Fallback: Comfy-Org mirror
UPSCALER_MODEL = (
    model_download(
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "/content/ComfyUI/models/latent_upscale_models"
    ) or
    model_download(
        "https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/split_files/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "/content/ComfyUI/models/latent_upscale_models"
    ) or "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)

# ── Print download status ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("📋 Downloaded model filenames:")
for label, val in [("UNet",          UNET_MODEL),
                   ("Text Enc 1",    TEXT_ENC1),
                   ("Text Enc 2",    TEXT_ENC2),
                   ("Video VAE",     VIDEO_VAE),
                   ("Audio VAE",     AUDIO_VAE),
                   ("Tiny VAE",      TINY_VAE),
                   ("Upscaler",      UPSCALER_MODEL)]:
    status = "✅" if val else "❌"
    print(f"  {status} {label:<14}: {val}")
print(f"{'='*60}")
print("\n✅ Core models downloaded.")


# ═══════════════════════════════════════════════════════════════
# CELL 2 ─── LoRA Downloads  (Director 2.0 4-LoRA stack + camera + IC)
# ═══════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown ## 🎭 Step 3 — Download LoRAs

LORA_DIR = "/content/ComfyUI/models/loras"

# ── Director 2.0 primary LoRA stack (from JSON node 138) ────────────────────
# These LoRAs are from the WhatDreamsCost workflow. The distilled-dynamic LoRA
# is confirmed on Lightricks/LTX-2.3. The others (OmniNFT, transition, MVCamera)
# are community/commercial LoRAs — we include best-known URLs with fallback
# instructions if unavailable.
DIRECTOR_LORAS = {
    # LoRA 1: distilled model optimisation  (strength 0.4 in JSON)
    # Confirmed from TensorVizion inference snippet
    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",

    # LoRA 2: OmniNFT RL quality  (strength 0.6 in JSON)
    # Community LoRA — try Lightricks org first
    "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",

    # LoRA 3: temporal scene transition  (strength 0.7 in JSON)
    # Multiple community mirrors available
    "ltx2.3-transition.safetensors":
        "https://huggingface.co/valiantcat/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors",

    # LoRA 4: MVCamera drclips — primary camera movement LoRA  (strength 0.9)
    # This LoRA triggers the 'drclipz' keyword in the global prompt
    "LTX2.3-MVCamera-drclips.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/LTX2.3-MVCamera-drclips.safetensors",
}

# ── Camera control LoRAs (LTX-2.3 versions from Lightricks org repos) ────────
CAMERA_LORAS = {
    "ltx-2.3-lora-camera-control-dolly-in.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2.3-lora-camera-control-dolly-in.safetensors",
    "ltx-2.3-lora-camera-control-dolly-out.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2.3-lora-camera-control-dolly-out.safetensors",
    "ltx-2.3-lora-camera-control-dolly-left.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2.3-lora-camera-control-dolly-left.safetensors",
    "ltx-2.3-lora-camera-control-dolly-right.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2.3-lora-camera-control-dolly-right.safetensors",
    "ltx-2.3-lora-camera-control-jib-up.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2.3-lora-camera-control-jib-up.safetensors",
    "ltx-2.3-lora-camera-control-jib-down.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2.3-lora-camera-control-jib-down.safetensors",
    "ltx-2.3-lora-camera-control-static.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Camera-Control-Static/resolve/main/ltx-2.3-lora-camera-control-static.safetensors",
}

# ── IC (Instant Control) LoRAs (LTX-2.3 versions) ───────────────────────────
IC_LORAS = {
    "ltx-2.3-ic-lora-canny-control.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Canny-Control/resolve/main/ltx-2.3-ic-lora-canny-control.safetensors",
    "ltx-2.3-ic-lora-depth-control.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Depth-Control/resolve/main/ltx-2.3-ic-lora-depth-control.safetensors",
    "ltx-2.3-ic-lora-pose-control.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Pose-Control/resolve/main/ltx-2.3-ic-lora-pose-control.safetensors",
    "ltx-2.3-ic-lora-detailer.safetensors":
        "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Detailer/resolve/main/ltx-2.3-ic-lora-detailer.safetensors",
}

def _batch_download(lora_dict: dict, label: str):
    print(f"\n--- {label} ---")
    failed = []
    for fname, url in lora_dict.items():
        result = model_download(url, LORA_DIR, fname)
        if not result:
            failed.append(fname)
    if failed:
        print(f"\n  ⚠️  {len(failed)} LoRA(s) could not be downloaded automatically.")
        print("  These may be gated/commercial models. To use them:")
        print("  1. Download manually from their HuggingFace page")
        print(f"  2. Upload to /content/ComfyUI/models/loras/")
        print("  3. The script will skip missing LoRAs gracefully and continue.\n")
        for f in failed:
            print(f"     ✗ {f}")

_batch_download(DIRECTOR_LORAS, "Director 2.0 LoRA Stack")
_batch_download(CAMERA_LORAS,   "Camera Control LoRAs (LTX-2.3)")
_batch_download(IC_LORAS,       "IC Control LoRAs (LTX-2.3)")

print("\n✅ All LoRAs downloaded.")


# ═══════════════════════════════════════════════════════════════
# CELL 3 ─── Global Configuration
# ═══════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown ## ⚙️ Step 4 — Configuration

# ── Output ───────────────────────────────────────────────────────────────────
PROJECT_NAME   = "LTX23_Director_MV"     # @param {type:"string"}

# ── Resolution: 1280×720 HD (matches JSON custom_width/height) ───────────────
WIDTH          = 1280    # @param {type:"integer"}
HEIGHT         = 720     # @param {type:"integer"}
FPS            = 24      # @param {type:"integer"}  matches JSON frame_rate=24

# ── Director 2.0 LoRA stack strengths (matches JSON node 138) ────────────────
DISTILLED_LORA_STRENGTH  = 0.4   # @param {type:"number"}
OMNINFT_LORA_STRENGTH    = 0.6   # @param {type:"number"}
TRANSITION_LORA_STRENGTH = 0.7   # @param {type:"number"}
MVCAMERA_LORA_STRENGTH   = 0.9   # @param {type:"number"}

# ── Pass parameters (matches JSON scheduler nodes 21 & 33) ───────────────────
# Pass 1: BasicScheduler  linear_quadratic  8 steps  denoise=1.0
PASS1_STEPS    = 8       # @param {type:"integer"}
PASS1_DENOISE  = 1.0     # @param {type:"number"}
# Pass 2: BasicScheduler  linear_quadratic  4 steps  denoise=0.42
PASS2_STEPS    = 4       # @param {type:"integer"}
PASS2_DENOISE  = 0.42    # @param {type:"number"}
# CFG = 1 for distilled model (same as JSON CFGGuider nodes 17 & 28)
CFG_SCALE      = 1       # @param {type:"integer"}

# ── Anchor / overlap ─────────────────────────────────────────────────────────
ANCHOR_STRENGTH_HIGH    = 0.90   # @param {type:"number"}
ANCHOR_STRENGTH_LOW     = 0.70   # @param {type:"number"}
USE_ADAPTIVE_STRENGTH   = True   # @param {type:"boolean"}
USE_ADAPTIVE_OVERLAP    = True   # @param {type:"boolean"}
OVERLAP_FRAMES          = 24     # @param {type:"integer"}

# ── LoRA feature toggles ─────────────────────────────────────────────────────
USE_DIRECTOR_LORA_STACK  = True   # @param {type:"boolean"} 4-LoRA Director stack
USE_CAMERA_LORAS         = True   # @param {type:"boolean"}
CAMERA_LORA_STRENGTH     = 0.80   # @param {type:"number"}
USE_IC_LORAS             = True   # @param {type:"boolean"}
USE_CANNY_CONTROL        = False  # @param {type:"boolean"}
USE_DEPTH_CONTROL        = True   # @param {type:"boolean"}
USE_POSE_CONTROL         = True   # @param {type:"boolean"}
USE_DETAILER             = True   # @param {type:"boolean"}
IC_LORA_STRENGTH         = 0.70   # @param {type:"number"}

# ── Post-processing ──────────────────────────────────────────────────────────
FACE_RESTORATION     = True    # @param {type:"boolean"}
OPTICAL_FLOW_STITCH  = True    # @param {type:"boolean"}
GENERATE_SUBTITLES   = True    # @param {type:"boolean"}
TRANSITION_TYPE      = "crossfade"  # @param ["crossfade","fade_black","none"]

# ── Quality presets (controls extra denoise passes) ───────────────────────────
QUALITY_MODE         = "balanced"   # @param ["preview","balanced","maximum"]

# ── Model cache ───────────────────────────────────────────────────────────────
USE_MODEL_CACHE      = True    # @param {type:"boolean"}
CACHE_MAX_AGE_DAYS   = 7       # @param {type:"integer"}

# ── Misc ─────────────────────────────────────────────────────────────────────
PARALLEL_PROCESSING  = True    # @param {type:"boolean"}
LLM_EXPANSION        = False   # @param {type:"boolean"}
LLM_PROVIDER         = "openai"  # @param ["openai","gemini"]
LLM_API_KEY          = ""       # @param {type:"string"}
GENERATE_SHOT_VARIATIONS = False # @param {type:"boolean"}
NUM_VARIATIONS       = 2        # @param {type:"integer"}
INTERACTIVE_MODE     = False    # @param {type:"boolean"}
USE_GPU_ENCODING     = True     # @param {type:"boolean"}
STORYBOARD_MODE      = False    # @param {type:"boolean"}


# ── Character / prompt settings ───────────────────────────────────────────────
USE_CHARACTER_SHEETS      = True         # @param {type:"boolean"}
CHARACTER_SHEET_PATH      = "/content/ComfyUI/input/character_ref.png"  # @param {type:"string"}
CHARACTER_IDENTITY_WEIGHT = 2.0          # @param {type:"number"}
USE_PROMPT_ENHANCEMENT    = True         # @param {type:"boolean"}
ENHANCEMENT_STRENGTH      = "high"       # @param ["low","medium","high"]
INJECT_CHARACTER_EVERY_SHOT = True       # @param {type:"boolean"}
USE_PROMPT_WEIGHTING        = True       # @param {type:"boolean"}
USE_NEGATIVE_PROMPT_EXPANSION = True     # @param {type:"boolean"}

# ── Global cinematic music-video prompt (matches JSON LTXDirector global_prompt)
# @markdown ### 🎬 Global Prompt  — edit freely
GLOBAL_PROMPT = """Create a highly realistic cinematic AI music video using the provided reference image.
Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body proportions,
and overall appearance exactly as shown. The singer must remain fully recognizable throughout
with absolutely no identity drift.

Performance Energy:
• Explosive stage presence. Every lyric instantly changes facial expression, head movement,
  shoulders, hands, posture and body rhythm.
• Own the stage with absolute confidence — perform as if in front of 50,000 screaming fans.
• Never appear calm, passive or static.

Facial Performance:
• Extremely expressive facial acting. Rich emotional transitions every few words.
• Powerful eye contact, sparkling eyes, highly expressive eyebrows synced with lyrics.
• Natural smiles, smirks, determination, excitement, confidence, attitude, passion.

Body Performance:
• Entire body constantly grooves with the beat. Strong rhythmic bouncing.
• Powerful shoulder accents, confident chest movement, hip movement, dynamic torso twists.
• Lean toward camera during emotional lyrics. Bold, energetic theatrical stage movement.

Hand Performance:
• Large expressive gestures. Sharp hand movements synced with the beat.
• Powerful pointing, sweeping arm movements, punching the air, open palm emphasis.
• Asymmetric movement — never repeat the same gesture pattern.

Camera (drclipz style):
drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back,
energetic handheld movement, rhythmic tracking shots, dynamic low-angle hero shots,
cinematic motion blur. Camera movement follows the beat.

Lighting:
Premium concert lighting — cinematic key light, colorful neon rim lights,
volumetric atmosphere, dramatic contrast, realistic skin tones.

Style:
Photorealistic, blockbuster-quality AI music video, ultra-high facial fidelity,
charismatic superstar, emotionally captivating, explosive stage energy, every second feels alive.
"""

# ── Base negative prompt ──────────────────────────────────────────────────────
NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, bad anatomy, text, watermark, ugly, deformed, "
    "glitch, morphing artifacts, extra limbs, fused fingers, poorly drawn face, "
    "inconsistent character, face change, clothing change, style inconsistency"
)

print("✅ Configuration loaded.")
print(f"   Resolution  : {WIDTH}×{HEIGHT} @ {FPS}fps")
print(f"   Quality mode: {QUALITY_MODE}")
print(f"   Pass 1      : {PASS1_STEPS} steps, denoise={PASS1_DENOISE}")
print(f"   Pass 2      : {PASS2_STEPS} steps, denoise={PASS2_DENOISE}")
print(f"   CFG         : {CFG_SCALE}  (distilled — no negative subtraction)")


# ═══════════════════════════════════════════════════════════════
# CELL 4 ─── Quality Presets & Model Cache
# ═══════════════════════════════════════════════════════════════

QUALITY_PRESETS: Dict[str, dict] = {
    "preview": {
        "pass1_steps": 6,  "pass1_denoise": 1.0,
        "pass2_steps": 3,  "pass2_denoise": 0.35,
        "bitrate": "5000k", "encode_preset": "fast",
        "description": "Quick preview — fastest, lower quality"
    },
    "balanced": {
        "pass1_steps": PASS1_STEPS,  "pass1_denoise": PASS1_DENOISE,
        "pass2_steps": PASS2_STEPS,  "pass2_denoise": PASS2_DENOISE,
        "bitrate": "10000k", "encode_preset": "medium",
        "description": "Balanced quality & speed  (matches JSON workflow)"
    },
    "maximum": {
        "pass1_steps": 12, "pass1_denoise": 1.0,
        "pass2_steps": 6,  "pass2_denoise": 0.50,
        "bitrate": "15000k", "encode_preset": "slow",
        "description": "Maximum quality — slowest"
    }
}

# ── Thread-safe model cache (keeps UNet/VAE in VRAM between shots) ─────────────
class ModelCache:
    def __init__(self):
        self._unet = None; self._vae = None
        self._audio_vae = None; self._lock = threading.Lock()

    def get_unet(self, loader_fn, force=False):
        with self._lock:
            if self._unet is None or force:
                print("   📦 Loading UNet into cache...")
                self._unet = loader_fn()
            return self._unet

    def get_vae(self, loader_fn, force=False):
        with self._lock:
            if self._vae is None or force:
                self._vae = loader_fn()
            return self._vae

    def evict(self):
        with self._lock:
            self._unet = self._vae = self._audio_vae = None
        gc.collect(); torch.cuda.empty_cache()
        print("   🗑️ Model cache evicted.")

MODEL_CACHE = ModelCache() if USE_MODEL_CACHE else None

# ── Lazy LoRA registry ────────────────────────────────────────────────────────
class LazyLoRARegistry:
    def __init__(self): self._loaded: Dict[str, Any] = {}
    def get(self, name, loader_fn):
        if name not in self._loaded:
            r = loader_fn(name)
            if r is not None:
                self._loaded[name] = r
                print(f"   ✓ Lazy-loaded LoRA: {name}")
            else:
                return None
        return self._loaded.get(name)
    def clear(self): self._loaded.clear()

LORA_REGISTRY = LazyLoRARegistry()

# ─── Utility functions ────────────────────────────────────────────────────────
def cleanup_memory(verbose=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    if verbose: print_vram_usage()

def get_available_vram() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0

def print_vram_usage():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1024**3
        total = get_available_vram()
        bar   = "█" * int(20 * used / total) + "░" * (20 - int(20 * used / total))
        print(f"   💾 VRAM [{bar}] {used:.1f}/{total:.1f} GB ({used/total*100:.1f}%)")

def auto_adjust_settings() -> Dict[str, Any]:
    vram = get_available_vram()
    if vram == 0:   return {}
    if vram < 16:
        print(f"⚠️ Low VRAM ({vram:.1f}GB) → preview mode, disable IC LoRAs")
        return {"USE_IC_LORAS": False, "QUALITY_MODE": "preview", "USE_MODEL_CACHE": False}
    elif vram < 24:
        print(f"ℹ️ Moderate VRAM ({vram:.1f}GB) → balanced mode")
        return {"QUALITY_MODE": "balanced"}
    print(f"✅ Ample VRAM ({vram:.1f}GB) → maximum quality")
    return {"QUALITY_MODE": "maximum"}

def format_time(s: float) -> str:
    ms = int((s - int(s)) * 1000)
    return f"{int(s//3600):02}:{int((s%3600)//60):02}:{int(s%60):02},{ms:03}"

def cleanup_old_cache(cache_dir: str, max_age_days: int = 7):
    if not os.path.isdir(cache_dir): return
    cutoff = time.time() - max_age_days * 86400
    removed = sum(
        1 for f in os.listdir(cache_dir)
        if os.path.isfile(fp := os.path.join(cache_dir, f))
        and os.path.getmtime(fp) < cutoff and not os.remove(fp)
    )
    if removed: print(f"🗑️ Removed {removed} stale cache file(s) from {cache_dir}")


# ═══════════════════════════════════════════════════════════════
# CELL 5 ─── Camera & IC LoRA Mappings  (LTX-2.3 filenames)
# ═══════════════════════════════════════════════════════════════

CAMERA_LORA_MAPPING: Dict[str, str] = {
    "dolly_forward":    "ltx-2.3-lora-camera-control-dolly-in.safetensors",
    "dolly_in":         "ltx-2.3-lora-camera-control-dolly-in.safetensors",
    "zoom_in":          "ltx-2.3-lora-camera-control-dolly-in.safetensors",
    "zoom_in_slow":     "ltx-2.3-lora-camera-control-dolly-in.safetensors",
    "zoom_in_fast":     "ltx-2.3-lora-camera-control-dolly-in.safetensors",
    "dolly_reveal":     "ltx-2.3-lora-camera-control-dolly-in.safetensors",
    "push_in_slow":     "ltx-2.3-lora-camera-control-dolly-in.safetensors",
    "dolly_backward":   "ltx-2.3-lora-camera-control-dolly-out.safetensors",
    "dolly_out":        "ltx-2.3-lora-camera-control-dolly-out.safetensors",
    "zoom_out":         "ltx-2.3-lora-camera-control-dolly-out.safetensors",
    "dolly_left":       "ltx-2.3-lora-camera-control-dolly-left.safetensors",
    "pan_left":         "ltx-2.3-lora-camera-control-dolly-left.safetensors",
    "dolly_right":      "ltx-2.3-lora-camera-control-dolly-right.safetensors",
    "pan_right":        "ltx-2.3-lora-camera-control-dolly-right.safetensors",
    "tilt_up":          "ltx-2.3-lora-camera-control-jib-up.safetensors",
    "tilt_up_slight":   "ltx-2.3-lora-camera-control-jib-up.safetensors",
    "tilt_up_reveal":   "ltx-2.3-lora-camera-control-jib-up.safetensors",
    "tilt_up_dramatic": "ltx-2.3-lora-camera-control-jib-up.safetensors",
    "jib_up":           "ltx-2.3-lora-camera-control-jib-up.safetensors",
    "crane_up":         "ltx-2.3-lora-camera-control-jib-up.safetensors",
    "tilt_down":        "ltx-2.3-lora-camera-control-jib-down.safetensors",
    "jib_down":         "ltx-2.3-lora-camera-control-jib-down.safetensors",
    "static":           "ltx-2.3-lora-camera-control-static.safetensors",
    "static_intense":   "ltx-2.3-lora-camera-control-static.safetensors",
    "static_dramatic":  "ltx-2.3-lora-camera-control-static.safetensors",
    "handheld_pov":     "ltx-2.3-lora-camera-control-static.safetensors",
    "low_angle_hero":   "ltx-2.3-lora-camera-control-static.safetensors",
}

IC_LORA_MAPPING: Dict[str, str] = {
    "canny":    "ltx-2.3-ic-lora-canny-control.safetensors",
    "depth":    "ltx-2.3-ic-lora-depth-control.safetensors",
    "pose":     "ltx-2.3-ic-lora-pose-control.safetensors",
    "detailer": "ltx-2.3-ic-lora-detailer.safetensors",
}

def validate_lora_exists(lora_name: str, lora_type: str = "lora") -> Optional[str]:
    """Search ComfyUI folder_paths then /content fallback."""
    if not lora_name: return None
    try:
        for base in folder_paths.get_folder_paths("loras"):
            direct = os.path.join(base, lora_name)
            if os.path.exists(direct): return direct
            for root, _, fs in os.walk(base):
                if lora_name in fs: return os.path.join(root, lora_name)
    except Exception as e:
        print(f"⚠️ folder_paths error: {e}")
    fallback = f"/content/ComfyUI/models/loras/{lora_name}"
    if os.path.exists(fallback): return fallback
    print(f"⚠️ {lora_type} LoRA not found: {lora_name}")
    return None

def get_camera_lora_for_shot(shot: dict) -> Tuple[Optional[str], Optional[str]]:
    mv = shot.get("camera_movement", "static")
    for key, val in CAMERA_LORA_MAPPING.items():
        if key in mv: return key, val
    return None, None

def get_ic_loras_for_shot(shot: dict) -> List[Tuple[str, str, float]]:
    if not USE_IC_LORAS: return []
    ct  = shot.get("control_types", [])
    out = []
    if USE_CANNY_CONTROL and "canny"    in ct: out.append(("canny",    IC_LORA_MAPPING["canny"],    IC_LORA_STRENGTH))
    if USE_DEPTH_CONTROL and "depth"    in ct: out.append(("depth",    IC_LORA_MAPPING["depth"],    IC_LORA_STRENGTH))
    if USE_POSE_CONTROL  and "pose"     in ct: out.append(("pose",     IC_LORA_MAPPING["pose"],     IC_LORA_STRENGTH))
    if USE_DETAILER      and "detailer" in ct: out.append(("detailer", IC_LORA_MAPPING["detailer"], IC_LORA_STRENGTH))
    return out


# ═══════════════════════════════════════════════════════════════
# CELL 6 ─── Scene JSON  (multi-image timeline — mirrors LTXDirector node)
# ═══════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown ## 🎬 Step 5 — Scene / Storyboard Definition
# @markdown
# @markdown Edit `SCENE_JSON` to define your own story.
# @markdown Each shot maps to one timeline segment (like in the JSON workflow).
# @markdown Upload your reference images to `/content/ComfyUI/input/` first.

SCENE_JSON = {
    "scene_id":      "ltx23_director_mv_demo",
    "project_name":  PROJECT_NAME,
    "duration_seconds": 32,
    "video_style":   (
        "Photorealistic cinematic music video, ultra-high facial fidelity, "
        "premium live concert performance, hyperrealistic skin texture, "
        "professional color grading, shallow depth of field"
    ),

    # ── Timeline segments  (mirrors LTXDirector node segments array) ─────────
    # Each entry = one reference image anchoring N frames of video.
    # 'image' paths are relative to /content/ComfyUI/input/
    # 'length_frames' mirrors the segment lengths in the JSON workflow.
    "timeline_segments": [
        {"image": "ref_1.png", "length_frames": 192, "prompt_override": ""},
        {"image": "ref_2.png", "length_frames": 137, "prompt_override": ""},
        {"image": "ref_3.png", "length_frames": 112, "prompt_override": ""},
        {"image": "ref_4.png", "length_frames": 192, "prompt_override": ""},
        {"image": "ref_5.png", "length_frames": 100, "prompt_override": ""},
    ],

    # ── Audio (optional — used in subtitle generation) ────────────────────────
    "audio": {
        "music_file":        "music_track.mp3",   # place in /content/ComfyUI/input/
        "background_music":  "High-energy trap beat with 808 bass, hi-hats, snare",
        "environment_sfx":   "Crowd roar, reverb tail, stadium ambience",
        "voice_processing":  "Slight plate reverb, subtle compression"
    },

    "environment": {
        "location":      "Sold-out stadium concert stage, massive LED wall backdrop",
        "time":          "Night",
        "weather":       "Indoor — controlled pyrotechnic smoke, laser beams",
        "mood":          "Electric, high-energy, euphoric",
        "lighting":      "Dynamic moving heads, neon rim lights, volumetric haze",
        "color_palette": "Cyan, magenta, deep purple, warm golden key light"
    },

    "main_characters": [
        {
            "name": "Artist",
            "desc": "Main performing artist — identity from reference image",
            "detailed_appearance": {
                "face":        "Expressive, high cheekbones, intense eyes, strong jaw",
                "hair":        "Short natural hair, styled with confidence",
                "clothing":    "Oversized designer jacket, chain necklace, fitted joggers",
                "build":       "Athletic, confident posture",
                "skin_tone":   "Deep warm brown skin with natural glow",
                "accessories": "Wireless microphone, diamond-encrusted watch"
            },
            "lora_path": None,
            "personality_traits":    "Charismatic, powerful, emotionally expressive",
            "voice_characteristics": "Deep rich voice, rap delivery, rhythmic cadence"
        }
    ],

    "story_action": {
        "shots": [
            {
                "time": "0-6s",  "camera": "Wide establishing shot, slow dolly in",
                "camera_movement": "dolly_in", "motion_intensity": 0.55,
                "action": "Artist walks to center stage. Stadium lights explode. Crowd erupts.",
                "character_focus": "Artist_full_body", "emotion": "confident_arrival",
                "visual_effects": "Pyrotechnic burst, stadium LED sync, crowd wave"
            },
            {
                "time": "6-12s", "camera": "Medium hero shot, low angle",
                "camera_movement": "low_angle_hero", "motion_intensity": 0.65,
                "action": "Artist launches into first verse. Explosive hand gestures. Shoulder rolls.",
                "character_focus": "Artist", "emotion": "aggressive_confident",
                "visual_effects": "Beat-synced light pulses, smoke machine burst",
                "control_types": ["depth", "detailer"]
            },
            {
                "time": "12-18s", "camera": "Extreme close-up face, slight push in",
                "camera_movement": "push_in_slow", "motion_intensity": 0.35,
                "action": "Close-up on artist's face during emotional bridge. Eyes intense, lips perfectly synced.",
                "character_focus": "Artist_face", "emotion": "vulnerable_intense",
                "visual_effects": "Rack focus, bokeh highlights, tear catch light",
                "control_types": ["detailer", "pose"]
            },
            {
                "time": "18-24s", "camera": "Fast intercutting — handheld energy",
                "camera_movement": "handheld_pov", "motion_intensity": 0.85,
                "action": "Peak of chorus. Full body performance, arms wide, jumping to the beat.",
                "character_focus": "Artist_full_body", "emotion": "euphoric_explosive",
                "visual_effects": "Confetti burst, strobe light hits, crowd lighters"
            },
            {
                "time": "24-32s", "camera": "Slow pull-back reveal, epic wide",
                "camera_movement": "dolly_out", "motion_intensity": 0.45,
                "action": "Camera pulls back to reveal entire stadium. Artist silhouetted in spotlight.",
                "character_focus": "Artist_silhouette", "emotion": "triumphant",
                "visual_effects": "Stadium reveal, fireworks, god rays through smoke"
            },
        ]
    },

    "dialogue_with_timing": [
        {"time": 2,  "character": "Artist", "dialogue": "Open up the canvas, blank space on my screen.",
         "english_translation": "", "emotion": "confident", "voice_direction": "Clear, rhythmic rap", "lip_sync_emphasis": "high"},
        {"time": 8,  "character": "Artist", "dialogue": "Drag a Checkpoint Loader, you know what I mean.",
         "english_translation": "", "emotion": "playful_confident", "voice_direction": "Faster pace", "lip_sync_emphasis": "high"},
        {"time": 16, "character": "Artist", "dialogue": "Connect the nodes, run the queue.",
         "english_translation": "", "emotion": "intense", "voice_direction": "Deep, emphatic", "lip_sync_emphasis": "high"},
        {"time": 24, "character": "Artist", "dialogue": "Watch the latent flow right through.",
         "english_translation": "", "emotion": "triumphant", "voice_direction": "Rising energy", "lip_sync_emphasis": "high"},
    ],

    "motion_guidance": {
        "global_motion":    "Forward motion toward camera, building energy through all shots",
        "character_motion": "High-energy performance, rhythmic movement, natural body language",
        "camera_motion":    "drclipz aggressive cinematic style matching JSON workflow"
    }
}


# ═══════════════════════════════════════════════════════════════
# CELL 7 ─── Validation, Prompt Builder, Adaptive System
# ═══════════════════════════════════════════════════════════════

def validate_scene_schema(d: dict) -> None:
    req = ["scene_id", "project_name", "story_action", "main_characters",
           "environment", "timeline_segments"]
    missing = [k for k in req if k not in d]
    if missing: raise ValueError(f"🚨 JSON missing top-level keys: {missing}")
    if "shots" not in d["story_action"]:
        raise ValueError("🚨 'story_action' missing 'shots'")
    req_shot = ["time", "camera", "camera_movement", "motion_intensity", "action"]
    for i, s in enumerate(d["story_action"]["shots"]):
        bad = [k for k in req_shot if k not in s]
        if bad: raise ValueError(f"🚨 Shot {i+1} missing keys: {bad}")
    if not d.get("main_characters"):
        raise ValueError("🚨 'main_characters' must be a non-empty list")
    for c in d["main_characters"]:
        if "name" not in c or "detailed_appearance" not in c:
            raise ValueError(f"🚨 Character missing required fields: {c.get('name','?')}")
    print("✅ Scene JSON schema validated.")

@lru_cache(maxsize=128)
def build_character_prompt_detailed(char_str: str) -> str:
    d  = json.loads(char_str)
    ap = d["detailed_appearance"]
    return (
        f"{d['name']}: {ap['face']}, {ap['hair']}, wearing {ap['clothing']}, "
        f"{ap['build']}, {ap['skin_tone']}, {ap['accessories']}. "
        f"ALWAYS MAINTAIN: {d['name']} has {ap['face'].split(',')[0]}, "
        f"{ap['hair'].split(',')[0]}, {ap['clothing'].split(',')[0]}. "
    )

def get_character_consistency_prefix(scene: dict) -> str:
    parts = [build_character_prompt_detailed(json.dumps(c, sort_keys=True))
             for c in scene["main_characters"]]
    return ("CHARACTER CONSISTENCY CRITICAL: " + " | ".join(parts) +
            " | MAINTAIN EXACT SAME CHARACTER APPEARANCE. NO MORPHING.")

def enhance_prompt_pro(prompt: str, strength: str = "medium") -> str:
    if not USE_PROMPT_ENHANCEMENT: return prompt
    tokens = {
        "low":    ["cinematic lighting", "high fidelity", "natural motion"],
        "medium": ["volumetric cinematic lighting", "hyper-realistic textures",
                   "fluid natural motion", "atmospheric depth"],
        "high":   ["masterpiece quality", "8k resolution", "extremely detailed cinematic textures",
                   "dynamic volumetric god rays", "realistic physics-based motion",
                   "micro-expression detailing", "perfect depth of field",
                   "professional color grading", "ultra-high definition"]
    }
    return f"{prompt} . (PROMPT ENHANCEMENT: {', '.join(tokens.get(strength, tokens['medium']))})"

def get_motion_guidance_prompt(shot: dict) -> str:
    mi  = shot.get("motion_intensity", 0.5)
    mv  = shot.get("camera_movement", "static").replace("_", " ")
    desc = ("minimal motion, subtle movements" if mi < 0.3
            else "moderate motion, natural movements" if mi < 0.6
            else "dynamic motion, energetic action")
    return f"MOTION GUIDANCE: {desc}. CAMERA: {mv}. "

def get_dialogue_for_shot(start_s: float, end_s: float, dl: list) -> str:
    lines = []
    for e in dl:
        if start_s <= e["time"] < end_s:
            ls = e.get("lip_sync_emphasis", "medium")
            if ls == "high":
                lines.append(
                    f"LIP SYNC CRITICAL: {e['character']} speaks '{e['dialogue']}' "
                    f"with {e.get('emotion','neutral')}. {e.get('voice_direction','')}.")
            else:
                lines.append(f"{e['character']} says '{e['dialogue']}'.")
    return " | ".join(lines)

def build_shot_prompt_pro(shot: dict, scene: dict, idx: int) -> str:
    try:
        t = shot["time"].replace("s","").split("-")
        ss, es = int(t[0]), int(t[1])
    except Exception: ss, es = 0, 6

    char_p = ""
    if INJECT_CHARACTER_EVERY_SHOT:
        char_p = get_character_consistency_prefix(scene)
        if USE_PROMPT_WEIGHTING and CHARACTER_IDENTITY_WEIGHT >= 2.0:
            char_p = f"(IDENTITY:{CHARACTER_IDENTITY_WEIGHT}) " + char_p

    env = scene["environment"]
    env_p = (f"ENVIRONMENT: {env['location']}. LIGHTING: {env['lighting']}. "
             f"MOOD: {env['mood']}. COLOR PALETTE: {env['color_palette']}. ")

    shot_p = (
        f"SHOT {idx+1}: {shot['action']}. "
        f"CAMERA: {shot['camera']}. "
        + get_motion_guidance_prompt(shot)
        + f"EMOTION: {shot.get('emotion','neutral')}. FOCUS: {shot.get('character_focus','scene')}. "
        + env_p
        + f"VISUAL EFFECTS: {shot.get('visual_effects','natural')}. "
        + f"STYLE: {scene['video_style']}. "
        + get_dialogue_for_shot(ss, es, scene["dialogue_with_timing"])
        + f" | GLOBAL DIRECTION: {GLOBAL_PROMPT[:500]}"
    )
    return f"{char_p} | " + enhance_prompt_pro(shot_p, ENHANCEMENT_STRENGTH)

def build_negative_prompt_enhanced() -> str:
    if not USE_NEGATIVE_PROMPT_EXPANSION: return NEGATIVE_PROMPT
    return (
        "character morphing, face changing, inconsistent character design, different clothing, "
        "style shift, motion blur artifacts, jittery movement, robotic motion, "
        "desynchronized lips, frozen face during dialogue, compression artifacts, "
        "pixelation, banding, flickering, " + NEGATIVE_PROMPT
    )

def calculate_adaptive_strength(shot: dict, prev: Optional[dict], prev_ok: bool) -> float:
    s = ANCHOR_STRENGTH_HIGH
    if prev:
        mc = abs(shot.get("motion_intensity", 0.5) - prev.get("motion_intensity", 0.5))
        if mc > 0.4:  s -= 0.10
        elif mc < 0.2: s += 0.05
        if shot.get("character_focus") != prev.get("character_focus"): s -= 0.05
    if not prev_ok: s -= 0.10
    return max(ANCHOR_STRENGTH_LOW, min(ANCHOR_STRENGTH_HIGH, s))

def calculate_adaptive_overlap(shot: dict) -> int:
    if not USE_ADAPTIVE_OVERLAP: return OVERLAP_FRAMES
    mi  = shot.get("motion_intensity", 0.5)
    adj = max(4, int(OVERLAP_FRAMES * 0.15))
    if mi > 0.7: return OVERLAP_FRAMES + adj
    if mi < 0.3: return OVERLAP_FRAMES - adj
    return OVERLAP_FRAMES


# ═══════════════════════════════════════════════════════════════
# CELL 8 ─── Storyboard Build & File Helpers
# ═══════════════════════════════════════════════════════════════

validate_scene_schema(SCENE_JSON)
for _k, _v in auto_adjust_settings().items():
    if _k in globals(): globals()[_k] = _v

STORYBOARD: List[dict] = []
shots = SCENE_JSON["story_action"]["shots"]
segs  = SCENE_JSON["timeline_segments"]

for idx, shot in enumerate(shots):
    _, cam_lora = get_camera_lora_for_shot(shot)
    # Map segment image to this shot (cycle through segments if fewer than shots)
    seg = segs[idx % len(segs)]
    img_path = os.path.join("/content/ComfyUI/input", seg["image"])
    STORYBOARD.append({
        "id":           f"shot_{idx+1:02d}",
        "prompt":       build_shot_prompt_pro(shot, SCENE_JSON, idx),
        "shot_data":    shot,
        "image_path":   img_path if os.path.exists(img_path) else None,
        "seg_frames":   seg.get("length_frames", 121),
        "camera_lora":  cam_lora if USE_CAMERA_LORAS else None,
        "ic_loras":     get_ic_loras_for_shot(shot),
        "prev_shot":    shots[idx-1] if idx > 0 else None,
    })

print(f"\n{'='*60}")
print(f"🎬 LTX-2.3 Director 2.0 — PRO v4.0")
print(f"{'='*60}")
print(f"  Shots          : {len(STORYBOARD)}")
print(f"  Resolution     : {WIDTH}×{HEIGHT} @ {FPS}fps")
print(f"  Quality mode   : {QUALITY_MODE} — {QUALITY_PRESETS[QUALITY_MODE]['description']}")
print(f"  Director LoRAs : {'ON' if USE_DIRECTOR_LORA_STACK else 'OFF'}")
print(f"  Camera LoRAs   : {'ON' if USE_CAMERA_LORAS else 'OFF'}")
print(f"  IC LoRAs       : {'ON' if USE_IC_LORAS else 'OFF'}")
print(f"  Pass 1         : {QUALITY_PRESETS[QUALITY_MODE]['pass1_steps']} steps / denoise={QUALITY_PRESETS[QUALITY_MODE]['pass1_denoise']}")
print(f"  Pass 2         : {QUALITY_PRESETS[QUALITY_MODE]['pass2_steps']} steps / denoise={QUALITY_PRESETS[QUALITY_MODE]['pass2_denoise']}")
print(f"  CFG            : {CFG_SCALE}  (distilled — no negative subtraction)")
print(f"  Transitions    : {TRANSITION_TYPE}")
print(f"{'='*60}\n")

# ─── File helpers ─────────────────────────────────────────────────────────────
def upload_image(dest: str = "/content/ComfyUI/input") -> Optional[str]:
    os.makedirs(dest, exist_ok=True)
    uploaded = files.upload()
    for fn in uploaded:
        src  = f"/content/ComfyUI/{fn}"
        path = os.path.join(dest, fn)
        shutil.move(src, path)
        return path
    return None

def display_video(path: str):
    if not path or not os.path.exists(path): return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        f'<video width="960" controls autoplay loop>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
        f'</video>'
    ))

def save_video_from_components(video_obj, prefix="LTX23/Video") -> str:
    try:
        w, h = video_obj.get_dimensions()
        folder, fname, counter, _, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), w, h)
        if Types:
            ext  = Types.VideoContainer.get_extension("auto")
            path = os.path.join(folder, f"{fname}_{counter:05}_.{ext}")
            video_obj.save_to(path, format=Types.VideoContainer("auto"),
                              codec="auto", metadata=None)
            return path
    except Exception:
        pass
    # Fallback: save via VHS_VideoCombine was already called inside generate_segment_pro
    return ""

def generate_srt_from_dialogue(dl: list, out_path: str):
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for i, e in enumerate(dl, 1):
                s = float(e.get("time", 0))
                en = s + float(e.get("duration", 4.0))
                f.write(f"{i}\n{format_time(s)} --> {format_time(en)}\n")
                f.write(f"[{e.get('character','?')}] {e.get('dialogue','')}\n")
                if e.get("english_translation"):
                    f.write(f"({e['english_translation']})\n")
                f.write("\n")
        print(f"✅ Subtitles → {out_path}")
    except Exception as e:
        print(f"⚠️ SRT generation failed: {e}")


# ═══════════════════════════════════════════════════════════════
# CELL 9 ─── Core Generation Engine  (Director 2.0 Pipeline)
# ═══════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown ## 🧠 Director 2.0 Two-Pass Pipeline
# @markdown
# @markdown Implements the exact node graph from
# @markdown `LTX-2.3_Director_2.0-MV-Workflow-30s.json`:
# @markdown
# @markdown ```
# @markdown Models → DualCLIPLoader (Gemma fp8 + text projection)
# @markdown       → Power LoRA Stack (4 LoRAs)
# @markdown       → ConditioningZeroOut → LTXVConditioning (fps=24)
# @markdown
# @markdown Pass 1 ── BasicScheduler(linear_quadratic, 8 steps, 1.0)
# @markdown         → LTXVConcatAVLatent
# @markdown         → CFGGuider(cfg=1)
# @markdown         → SamplerCustomAdvanced(euler)
# @markdown         → LTXVSeparateAVLatent
# @markdown         → LTXVCropGuides / LTXDirectorCropGuides
# @markdown
# @markdown Pass 2 ── LTXVLatentUpsampler(x2)
# @markdown         → BasicScheduler(linear_quadratic, 4 steps, 0.42)
# @markdown         → LTXVConcatAVLatent
# @markdown         → CFGGuider(cfg=1)
# @markdown         → SamplerCustomAdvanced(euler)
# @markdown         → LTXVSeparateAVLatent
# @markdown
# @markdown Decode  → VAEDecode + LTXVAudioVAEDecode
# @markdown         → VHS_VideoCombine (H.264, CRF 8, 24fps)
# @markdown ```

def generate_segment_director(
    image_path:      Optional[str]         = None,
    prompt:          str                   = "",
    seed:            int                   = 42,
    frames:          int                   = 121,
    image_strength:  float                 = 0.90,
    camera_lora:     Optional[str]         = None,
    ic_loras:        Optional[List[Tuple]] = None,
    shot_data:       Optional[dict]        = None,
    shot_index:      int                   = 0,
) -> Optional[str]:
    """
    Director 2.0 two-pass generation — mirrors the JSON workflow exactly.

    Pass 1:  Euler · BasicScheduler(linear_quadratic, pass1_steps, denoise=1.0)
             LTXVConcatAVLatent → CFGGuider(cfg=1) → SamplerCustomAdvanced
             LTXVSeparateAVLatent → crop guides

    Pass 2:  LTXVLatentUpsampler(x2)
             Euler · BasicScheduler(linear_quadratic, pass2_steps, denoise=0.42)
             LTXVConcatAVLatent → CFGGuider(cfg=1) → SamplerCustomAdvanced
             LTXVSeparateAVLatent

    Decode:  VAEDecode + LTXVAudioVAEDecode → VHS_VideoCombine
    """
    import_custom_nodes()
    t0     = time.time()
    preset = QUALITY_PRESETS[QUALITY_MODE]

    # ── LLM prompt expansion (optional) ──────────────────────────────────────
    if LLM_EXPANSION and shot_data:
        prompt = expand_prompt_cinematically(
            shot_data.get("action", prompt),
            SCENE_JSON["video_style"], LLM_PROVIDER, LLM_API_KEY
        )

    if STORYBOARD_MODE: frames = 1

    pos_prompt = f"{GLOBAL_PROMPT[:800]} | {prompt}"
    neg_prompt = build_negative_prompt_enhanced()

    print(f"\n  🎬 Shot {shot_index+1} | seed={seed} | strength={image_strength:.2f}"
          f" | {frames}f | {QUALITY_MODE}")
    print(f"  📝 {prompt[:120]}...")
    if camera_lora: print(f"  🎥 Camera LoRA: {camera_lora}")
    print_vram_usage()

    with torch.inference_mode():

        # ── 1. Image loading ──────────────────────────────────────────────────
        loadimage    = NODE_CLASS_MAPPINGS["LoadImage"]()
        image_bypass = True
        img_node     = (torch.full((1, HEIGHT, WIDTH, 3), 0.5), None)

        if image_path and os.path.exists(image_path):
            try:
                img_node     = loadimage.load_image(image=os.path.basename(image_path))
                image_bypass = False
            except Exception as e:
                print(f"  ⚠️ Image load failed → T2V: {e}")
                image_strength = 0.0

        # ── 2. Image preprocessing ────────────────────────────────────────────
        #    Resize to WIDTH×HEIGHT → scale to longer edge 848 → LTXVPreprocess
        rimn  = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        rile  = NODE_CLASS_MAPPINGS["ResizeImagesByLongerEdge"]()
        lprep = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()

        resized = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(img_node, 0), scale_method="lanczos",
            resize_type={"resize_type": "scale dimensions",
                         "width": WIDTH, "height": HEIGHT, "crop": "center"})
        rlong   = rile.EXECUTE_NORMALIZED(
            longer_edge=max(WIDTH, HEIGHT),
            images=get_value_at_index(resized, 0))
        # img_compression=18 matches JSON LTXDirector widget
        preproc = lprep.EXECUTE_NORMALIZED(
            img_compression=18,
            image=get_value_at_index(rlong, 0))

        # Compute latent dims: width//2, height//2  (LTX-2.3 direct calculation)
        lat_w = WIDTH  // 2
        lat_h = HEIGHT // 2

        eltxv   = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        el_lat  = eltxv.EXECUTE_NORMALIZED(
            width=lat_w, height=lat_h,
            length=frames, batch_size=1)

        pint    = NODE_CLASS_MAPPINGS["PrimitiveInt"]()
        pframes = pint.EXECUTE_NORMALIZED(value=frames)

        # ── 3. DualCLIPLoader: Gemma fp8 + ltx-2.3 text projection bf16 ──────
        #    Matches JSON node 12: DualCLIPLoader(gemma_fp8_scaled, ltx_projection, "ltxv")
        dclip    = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        clip_mdl = dclip.load_clip(
            clip_name1=TEXT_ENC1,
            clip_name2=TEXT_ENC2,
            type="ltxv", device="default"
        )
        cte      = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cond_pos = cte.encode(text=pos_prompt, clip=get_value_at_index(clip_mdl, 0))
        cond_neg = cte.encode(text=neg_prompt, clip=get_value_at_index(clip_mdl, 0))
        del clip_mdl; cleanup_memory()

        # ── 4. ConditioningZeroOut → LTXVConditioning(fps=24) ────────────────
        #    Matches JSON nodes 128 & 27
        czo      = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        null_neg = czo.zero_out(conditioning=get_value_at_index(cond_pos, 0))
        ltxcond  = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        cond     = ltxcond.EXECUTE_NORMALIZED(
            frame_rate=FPS,
            positive=get_value_at_index(cond_pos, 0),
            negative=get_value_at_index(null_neg, 0)
        )

        # ── 5. Video VAE: encode image → latent ───────────────────────────────
        #    Uses LTX23_video_vae_bf16 (matches JSON node 36)
        vael   = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae1   = vael.load_vae(vae_name=VIDEO_VAE)
        i2v    = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
        img2v  = i2v.EXECUTE_NORMALIZED(
            strength=image_strength, bypass=image_bypass,
            vae=get_value_at_index(vae1, 0),
            image=get_value_at_index(preproc, 0),
            latent=get_value_at_index(el_lat, 0))
        del vae1; cleanup_memory()

        # ── 6. Audio VAE: create silent audio latent ─────────────────────────
        #    Uses LTX23_audio_vae_bf16 (matches JSON node 8)
        avael    = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
        avae     = avael.load_vae(
            vae_name=AUDIO_VAE,
            device="main_device", weight_dtype="bf16"
        )
        elalat   = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        audio_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=get_value_at_index(pframes, 0),
            frame_rate=FPS, batch_size=1,
            audio_vae=get_value_at_index(avae, 0)
        )

        # ── 7. UNet + 4-LoRA Director stack ───────────────────────────────────
        #    Matches JSON nodes 135 (UnetLoaderGGUF) + 138 (Power Lora Loader)
        unetgg = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unet   = get_value_at_index(unetgg.load_unet(unet_name=UNET_MODEL), 0)
        ll     = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()

        if USE_DIRECTOR_LORA_STACK:
            director_stack = [
                # (filename, strength) — matches JSON node 138 exactly
                ("ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", DISTILLED_LORA_STRENGTH),
                ("LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",                              OMNINFT_LORA_STRENGTH),
                ("ltx2.3-transition.safetensors",                                          TRANSITION_LORA_STRENGTH),
                ("LTX2.3-MVCamera-drclips.safetensors",                                    MVCAMERA_LORA_STRENGTH),
            ]
            for lname, lstr in director_stack:
                if validate_lora_exists(lname, "director"):
                    try:
                        unet = ll.load_lora_model_only(unet, lname, lstr)[0]
                        print(f"  ✓ Director LoRA [{lstr}]: {lname}")
                    except Exception as e:
                        print(f"  ⚠️ Director LoRA failed ({lname}): {e}")

        if USE_CAMERA_LORAS and camera_lora:
            if validate_lora_exists(camera_lora, "camera"):
                try:
                    unet = ll.load_lora_model_only(unet, camera_lora, CAMERA_LORA_STRENGTH)[0]
                    print(f"  ✓ Camera LoRA [{CAMERA_LORA_STRENGTH}]: {camera_lora}")
                except Exception as e:
                    print(f"  ⚠️ Camera LoRA failed: {e}")

        if ic_loras:
            for ic_name, ic_file, ic_str in ic_loras:
                if validate_lora_exists(ic_file, "IC"):
                    try:
                        unet = ll.load_lora_model_only(unet, ic_file, ic_str)[0]
                        print(f"  ✓ IC LoRA [{ic_str}]: {ic_name}")
                    except Exception as e:
                        print(f"  ⚠️ IC LoRA failed ({ic_name}): {e}")

        # ── 8. Sampler setup (shared by both passes) ──────────────────────────
        ksel   = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        # Both passes use Euler (matches JSON nodes 20 & 32)
        sampler_p1 = ksel.get_sampler(sampler_name="euler")
        sampler_p2 = ksel.get_sampler(sampler_name="euler")

        rn     = NODE_CLASS_MAPPINGS["RandomNoise"]()
        # Same noise seed for both passes — matches JSON node 30 (fixed seed)
        noise  = rn.EXECUTE_NORMALIZED(noise_seed=seed)

        cfg_node = NODE_CLASS_MAPPINGS["CFGGuider"]()
        sca      = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        sep      = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        concat   = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()

        # ────────────────────────────────────────────────────────────────────
        # PASS 1  — Full denoise at base resolution
        # Matches JSON nodes: 133(DirectorGuide) → 29(ConcatAV) → 28(CFG) → 31(Sampler)
        # ────────────────────────────────────────────────────────────────────
        print(f"\n  ⚡ Pass 1 — {preset['pass1_steps']} steps / denoise={preset['pass1_denoise']}")

        # BasicScheduler: linear_quadratic, pass1_steps, denoise=1.0
        # Matches JSON node 33: BasicScheduler(linear_quadratic, 8, 1)
        bsched = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        sigmas_p1 = bsched.get_sigmas(
            model=unet,
            scheduler="linear_quadratic",
            steps=preset["pass1_steps"],
            denoise=preset["pass1_denoise"]
        )

        vsrc_p1  = get_value_at_index(img2v, 0) if not image_bypass else get_value_at_index(el_lat, 0)
        av_p1    = concat.EXECUTE_NORMALIZED(
            video_latent=vsrc_p1,
            audio_latent=get_value_at_index(audio_lat, 0)
        )
        guider_p1 = cfg_node.EXECUTE_NORMALIZED(
            cfg=CFG_SCALE, model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1)
        )
        out_p1 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(noise, 0),
            guider=get_value_at_index(guider_p1, 0),
            sampler=get_value_at_index(sampler_p1, 0),
            sigmas=get_value_at_index(sigmas_p1, 0),
            latent_image=get_value_at_index(av_p1, 0)
        )
        del guider_p1; cleanup_memory()

        # ── Separate AV + Crop Guides (matches JSON nodes 34 & 55) ──────────
        s1 = sep.EXECUTE_NORMALIZED(av_latent=get_value_at_index(out_p1, 0))

        # LTXVCropGuides / LTXDirectorCropGuides — try Director version first
        try:
            crop_fn = NODE_CLASS_MAPPINGS.get(
                "LTXDirectorCropGuides",
                NODE_CLASS_MAPPINGS.get("LTXVCropGuides")
            )
            cr1 = crop_fn().EXECUTE_NORMALIZED(
                positive=get_value_at_index(cond, 0),
                negative=get_value_at_index(cond, 1),
                latent=get_value_at_index(s1, 0)
            )
            cropped_lat_p1 = get_value_at_index(cr1, 2)
        except Exception:
            # Fallback: skip crop if node unavailable
            cropped_lat_p1 = get_value_at_index(s1, 0)

        # ────────────────────────────────────────────────────────────────────
        # LATENT UPSCALE  2×  (matches JSON nodes 13 & 14)
        # ────────────────────────────────────────────────────────────────────
        print("  🔍 Upscaling latent 2×...")
        vae2   = vael.load_vae(vae_name=VIDEO_VAE)
        uml    = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        um     = uml.EXECUTE_NORMALIZED(model_name=UPSCALER_MODEL)
        lup    = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        upsampled = lup.upsample_latent(
            samples=cropped_lat_p1,
            upscale_model=get_value_at_index(um, 0),
            vae=get_value_at_index(vae2, 0)
        )
        del um, vae2; cleanup_memory()

        # ────────────────────────────────────────────────────────────────────
        # PASS 2  — Refinement at 2× resolution
        # Matches JSON nodes: 132(DirectorGuide) → 18(ConcatAV) → 17(CFG) → 19(Sampler)
        # ────────────────────────────────────────────────────────────────────
        print(f"  ✨ Pass 2 — {preset['pass2_steps']} steps / denoise={preset['pass2_denoise']}")

        vae3   = vael.load_vae(vae_name=VIDEO_VAE)
        img2v2 = i2v.EXECUTE_NORMALIZED(
            strength=image_strength, bypass=image_bypass,
            vae=get_value_at_index(vae3, 0),
            image=get_value_at_index(preproc, 0),
            latent=upsampled
        )
        del vae3; cleanup_memory()

        # BasicScheduler: linear_quadratic, pass2_steps, denoise=0.42
        # Matches JSON node 21: BasicScheduler(linear_quadratic, 4, 0.42)
        sigmas_p2 = bsched.get_sigmas(
            model=unet,
            scheduler="linear_quadratic",
            steps=preset["pass2_steps"],
            denoise=preset["pass2_denoise"]
        )

        vsrc_p2  = get_value_at_index(img2v2, 0) if not image_bypass else upsampled
        # Pass 2 audio comes from Pass 1 separation (matches JSON link 32: sep → concat)
        av_p2    = concat.EXECUTE_NORMALIZED(
            video_latent=vsrc_p2,
            audio_latent=get_value_at_index(s1, 1)
        )
        guider_p2 = cfg_node.EXECUTE_NORMALIZED(
            cfg=CFG_SCALE, model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1)
        )
        out_p2 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(noise, 0),
            guider=get_value_at_index(guider_p2, 0),
            sampler=get_value_at_index(sampler_p2, 0),
            sigmas=get_value_at_index(sigmas_p2, 0),
            latent_image=get_value_at_index(av_p2, 0)
        )
        del guider_p2, unet; cleanup_memory()

        # ── Separate Pass 2 AV + Final Crop ──────────────────────────────────
        # Matches JSON nodes 22 (LTXVSeparateAVLatent) + 54 (LTXDirectorCropGuides)
        s2 = sep.EXECUTE_NORMALIZED(av_latent=get_value_at_index(out_p2, 0))

        try:
            cr2 = crop_fn().EXECUTE_NORMALIZED(
                positive=get_value_at_index(cond, 0),
                negative=get_value_at_index(cond, 1),
                latent=get_value_at_index(s2, 0)
            )
            final_video_lat = get_value_at_index(cr2, 2)
        except Exception:
            final_video_lat = get_value_at_index(s2, 0)

        # ────────────────────────────────────────────────────────────────────
        # DECODE  — Video VAE + Audio VAE
        # Matches JSON nodes 1 (VAEDecode) + 24 (LTXVAudioVAEDecode)
        # ────────────────────────────────────────────────────────────────────
        print("  🎞️ Decoding video + audio...")
        vae4    = vael.load_vae(vae_name=VIDEO_VAE)
        vd      = NODE_CLASS_MAPPINGS["VAEDecode"]()
        vid_dec = vd.decode(
            samples=final_video_lat,
            vae=get_value_at_index(vae4, 0)
        )
        del vae4

        aud_dec   = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        audio_out = aud_dec.EXECUTE_NORMALIZED(
            samples=get_value_at_index(s2, 1),
            audio_vae=get_value_at_index(avae, 0)
        )
        del avae; cleanup_memory()

        # ────────────────────────────────────────────────────────────────────
        # VHS_VideoCombine — H.264 MP4, CRF 8, 24fps
        # Matches JSON node 139: VHS_VideoCombine(h264-mp4, yuv420p, crf=8)
        # ────────────────────────────────────────────────────────────────────
        output_dir_vhs = "/content/ComfyUI/output/LTX23"
        os.makedirs(output_dir_vhs, exist_ok=True)

        try:
            vhs = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            vhs_out = vhs.combine_video(
                images=get_value_at_index(vid_dec, 0),
                audio=get_value_at_index(audio_out, 0),
                frame_rate=FPS,
                loop_count=0,
                filename_prefix=f"LTX23/shot_{shot_index+1:02d}",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=8,
                save_metadata=False,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
            )
            # Retrieve saved path
            try:
                out_path = get_value_at_index(vhs_out, 0)[0]
            except Exception:
                # Fallback: find the newest mp4 in the output dir
                mp4s = sorted(
                    [os.path.join(output_dir_vhs, f)
                     for f in os.listdir(output_dir_vhs) if f.endswith(".mp4")],
                    key=os.path.getmtime
                )
                out_path = mp4s[-1] if mp4s else None

        except Exception as e:
            print(f"  ⚠️ VHS_VideoCombine unavailable ({e}), using CreateVideo fallback...")
            cv_node  = NODE_CLASS_MAPPINGS["CreateVideo"]()
            vid_obj  = cv_node.EXECUTE_NORMALIZED(
                fps=FPS,
                images=get_value_at_index(vid_dec, 0),
                audio=get_value_at_index(audio_out, 0)
            )
            out_path = save_video_from_components(get_value_at_index(vid_obj, 0))

        print(f"  ✅ Shot done in {time.time()-t0:.1f}s  →  {out_path}")
        return out_path


# ═══════════════════════════════════════════════════════════════
# CELL 10 ─── Post-Processing: Optical Flow, Face Restoration
# ═══════════════════════════════════════════════════════════════

def apply_optical_flow_morph(img1: np.ndarray, img2: np.ndarray,
                              steps: int = 5) -> List[np.ndarray]:
    """Farneback optical-flow warp-blend — cinematic morph between clips."""
    pg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ng = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pg, ng, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = img1.shape[:2]
    frames = []
    for i in range(1, steps + 1):
        alpha   = i / (steps + 1)
        fc      = flow * alpha
        mx, my  = np.meshgrid(np.arange(w), np.arange(h))
        mx      = (mx + fc[..., 0]).astype(np.float32)
        my      = (my + fc[..., 1]).astype(np.float32)
        warped  = cv2.remap(img1, mx, my, interpolation=cv2.INTER_LINEAR)
        blended = cv2.addWeighted(warped, 1 - alpha, img2, alpha, 0)
        frames.append(blended)
    return frames

def apply_face_restoration(video_path: str) -> str:
    """YOLO face detection + cv2.detailEnhance per frame."""
    if not FACE_RESTORATION: return video_path
    print("  ✨ Face restoration pass...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-face.pt")
    except Exception:
        print("  ⚠️ Ultralytics/face model unavailable — skipping.")
        return video_path

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_p = video_path.replace(".mp4", "_fr.mp4")
    out   = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret: break
        for r in model(frame, verbose=False):
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                if face.size == 0: continue
                frame[y1:y2, x1:x2] = cv2.detailEnhance(face, sigma_s=10, sigma_r=0.15)
        out.write(frame)
    cap.release(); out.release()
    if os.path.exists(out_p):
        os.replace(out_p, video_path)
    return video_path

# ─── Anchor extraction (motion-scored) ───────────────────────────────────────
def calculate_motion_score(f1: np.ndarray, f2: np.ndarray) -> float:
    return 1.0 / (1.0 + float(np.mean(cv2.absdiff(f1, f2))))

def extract_overlap_anchor(video_path: str, output_folder: str,
                            scene_idx: int, overlap: int = 24) -> Optional[str]:
    if not os.path.exists(video_path): return None
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tp    = max(total - 5, total - overlap) if overlap < total else max(0, total - 5)
    ws, we = max(0, tp - 4), min(total - 1, tp + 4)
    frames: Dict[int, np.ndarray] = {}
    for fi in range(ws, we + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, f = cap.read()
        if ret: frames[fi] = f
    cap.release()
    if not frames: return None

    sorted_fi = sorted(frames.keys())
    best, best_score = None, -1.0
    for i, fi in enumerate(sorted_fi):
        frame = frames[fi]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        br    = float(cv2.mean(gray)[0])
        if br < 5: continue
        sh    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        ms    = calculate_motion_score(frames[sorted_fi[i-1]], frame) if i > 0 else 1.0
        score = br * 0.3 + sh * 0.1 + ms * 100
        if score > best_score:
            best_score = score; best = frame

    if best is None: return None

    # Character sheet blending (5% weight — anchor consistency)
    if USE_CHARACTER_SHEETS and os.path.exists(CHARACTER_SHEET_PATH):
        try:
            sheet = cv2.imread(CHARACTER_SHEET_PATH)
            if sheet is not None:
                sheet = cv2.resize(sheet, (WIDTH, HEIGHT))
                best  = cv2.addWeighted(best, 0.95, sheet, 0.05, 0)
        except Exception: pass

    os.makedirs(output_folder, exist_ok=True)
    p = os.path.join(output_folder, f"anchor_{scene_idx}.png")
    cv2.imwrite(p, best)
    print(f"  ✓ Anchor extracted (score={best_score:.2f})")
    return p

def calculate_shot_metrics(video_path: str) -> Tuple[float, dict]:
    try:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, total // 10)
        sv, bv, mv = [], [], []
        prev  = None
        for fi in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, f = cap.read()
            if not ret: continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sv.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            bv.append(float(cv2.mean(g)[0]))
            if prev is not None: mv.append(float(np.mean(cv2.absdiff(prev, g))))
            prev = g
        cap.release()
        m = {"sharpness": float(np.mean(sv)) if sv else 0.0,
             "brightness": float(np.mean(bv)) if bv else 0.0,
             "motion_std": float(np.std(mv)) if mv else 0.0}
        score = (min(1.0, m["sharpness"]/1000)
                 + min(1.0, m["brightness"]/200)
                 + max(0.0, 1.0 - m["motion_std"]/50)) / 3.0
        return score, m
    except Exception as e:
        print(f"  ⚠️ Metrics error: {e}"); return 0.0, {}


# ═══════════════════════════════════════════════════════════════
# CELL 11 ─── Video Stitching  (VHS-quality H.264 + multi-track audio)
# ═══════════════════════════════════════════════════════════════

def _detect_nvenc() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
        return "h264_nvenc" in out.stdout
    except Exception: return False

def apply_transitions(clips: list, ttype: str = "crossfade", dur: float = 0.5) -> list:
    if ttype == "none" or len(clips) < 2: return clips
    result = []
    for i, c in enumerate(clips):
        if ttype == "crossfade":
            if i > 0:               c = c.crossfadein(dur)
            if i < len(clips)-1:    c = c.crossfadeout(dur)
        elif ttype == "fade_black":
            if i > 0:               c = c.fadein(dur)
            if i < len(clips)-1:    c = c.fadeout(dur)
        result.append(c)
    return result

def stitch_videos_director(
    video_paths:      List[str],
    output_filename:  str = "LTX23_Final.mp4",
    overlap_frames:   int = 24,
    music_track:      Optional[str] = None,
) -> Optional[str]:
    """
    Stitch clips with MoviePy + optional multi-track audio.
    Encodes with GPU NVENC if available (matches JSON crf=8 quality).
    """
    if not video_paths: return None

    preset  = QUALITY_PRESETS[QUALITY_MODE]
    bitrate = preset["bitrate"]
    print(f"\n🧵 Stitching {len(video_paths)} clips | {QUALITY_MODE} | {bitrate}")

    from moviepy.editor import VideoFileClip, concatenate_videoclips
    from moviepy.audio.AudioClip import CompositeAudioClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip

    clips = []
    for i, path in enumerate(video_paths):
        if not os.path.exists(path):
            print(f"⚠️ Missing clip: {path}"); continue
        try:
            clip = VideoFileClip(path)
            if clip.fps != FPS: clip = clip.set_fps(FPS)
            if i < len(video_paths) - 1:
                ovr = min(overlap_frames / float(FPS), clip.duration * 0.5)
                t   = clip.duration - ovr
                if t > 0: clip = clip.subclip(0, t)
            if clip.audio is not None:
                clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
            clips.append(clip)
        except Exception as e:
            print(f"⚠️ Skipping clip: {e}")

    if not clips: return None

    clips     = apply_transitions(clips, TRANSITION_TYPE)
    final_vid = concatenate_videoclips(clips, method="compose")

    # Optional music track mixing (mirrors JSON audio VAE + custom audio)
    if music_track and os.path.exists(music_track):
        print("  🎵 Mixing music track...")
        try:
            layers = []
            if final_vid.audio: layers.append(final_vid.audio)
            music = AudioFileClip(music_track).volumex(0.4).loop(duration=final_vid.duration)
            layers.append(music)
            if layers: final_vid.audio = CompositeAudioClip(layers)
        except Exception as e:
            print(f"  ⚠️ Audio mix failed: {e}")

    os.makedirs("/content/ComfyUI/output/LTX23", exist_ok=True)
    out_path  = f"/content/ComfyUI/output/LTX23/{output_filename}"
    use_nvenc = USE_GPU_ENCODING and _detect_nvenc()
    codec     = "h264_nvenc" if use_nvenc else "libx264"
    enc_p     = "p4" if use_nvenc else preset["encode_preset"]
    print(f"⏳ Encoding — {'GPU NVENC 🚀' if use_nvenc else 'CPU libx264'} | preset={enc_p}")
    final_vid.write_videofile(
        out_path, fps=FPS, codec=codec, audio_codec="aac",
        bitrate=bitrate, preset=enc_p, threads=4, logger=None
    )
    return out_path

def expand_prompt_cinematically(action: str, style: str,
                                 provider: str = "openai", api_key: str = "") -> str:
    if not LLM_EXPANSION or not action: return action
    if not api_key:
        return (f"{action}. Cinematic volumetric lighting, shallow depth of field, "
                f"35mm lens. {style}.")
    sys_p = (
        "You are a professional cinematographer. Expand this action into a "
        "300-word cinematic AI video prompt. Focus on lighting, camera movement, "
        f"texture, atmosphere. Style: {style}"
    )
    try:
        if provider == "openai":
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": "gpt-4o",
                      "messages": [{"role": "system", "content": sys_p},
                                   {"role": "user",   "content": action}],
                      "temperature": 0.7},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        elif provider == "gemini":
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-pro:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": f"{sys_p}\n\nACTION: {action}"}]}]},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"  ⚠️ LLM expansion failed: {e}")
    return action

def generate_shot_with_variations(scene: dict, image_path: Optional[str],
                                   strength: float, base_seed: int) -> Optional[str]:
    best_path, best_score = None, -1.0
    for vi in range(NUM_VARIATIONS):
        seed = base_seed + vi * 1000
        print(f"  🎲 Variation {vi+1}/{NUM_VARIATIONS} seed={seed}")
        try:
            clip = generate_segment_director(
                image_path=image_path, prompt=scene["prompt"],
                seed=seed, image_strength=strength,
                frames=scene.get("seg_frames", 121),
                camera_lora=scene.get("camera_lora"),
                ic_loras=scene.get("ic_loras", []),
                shot_data=scene["shot_data"],
                shot_index=scene.get("_idx", 0)
            )
            if clip:
                score, m = calculate_shot_metrics(clip)
                print(f"     Score={score:.3f} {m}")
                if score > best_score: best_score = score; best_path = clip
        except Exception as e:
            print(f"     ⚠️ Variation {vi+1} failed: {e}")
    if best_path: print(f"  🏆 Best variation score={best_score:.3f}")
    return best_path


# ═══════════════════════════════════════════════════════════════
# CELL 12 ─── Environment Check & Auto-Upload
# ═══════════════════════════════════════════════════════════════

def check_environment():
    # Guard: if downloads returned None, treat as missing
    required = {
        "UNet":        f"/content/ComfyUI/models/unet/{UNET_MODEL}"           if UNET_MODEL    else None,
        "Text Enc 1":  f"/content/ComfyUI/models/text_encoders/{TEXT_ENC1}"   if TEXT_ENC1     else None,
        "Text Enc 2":  f"/content/ComfyUI/models/text_encoders/{TEXT_ENC2}"   if TEXT_ENC2     else None,
        "Video VAE":   f"/content/ComfyUI/models/vae/{VIDEO_VAE}"             if VIDEO_VAE     else None,
        "Audio VAE":   f"/content/ComfyUI/models/vae/{AUDIO_VAE}"             if AUDIO_VAE     else None,
        "Upscaler":    f"/content/ComfyUI/models/latent_upscale_models/{UPSCALER_MODEL}" if UPSCALER_MODEL else None,
    }
    missing = []
    for name, path in required.items():
        if path is None or not os.path.exists(path):
            status = "❌ None (download failed)" if path is None else f"❌ Not found: {path}"
            print(f"  {status}  [{name}]")
            missing.append(name)
        else:
            print(f"  ✅ {name}: {os.path.basename(path)}")

    if missing:
        print(f"\n🚨 Missing critical models: {missing}")
        print("   → Re-run the download cell, or manually place files in:")
        print("     /content/ComfyUI/models/unet/  (for UNet)")
        print("     /content/ComfyUI/models/text_encoders/  (for text encoders)")
        print("   → Then re-run this cell.")
        raise FileNotFoundError(f"Missing models: {missing}")

    print("\n✅ All required models found.")

    # Check Director LoRA stack (warnings only — missing LoRAs are skipped)
    print("\nDirector 2.0 LoRA stack:")
    for lname in list(DIRECTOR_LORAS.keys()):
        p = f"/content/ComfyUI/models/loras/{lname}"
        status = "✅" if os.path.exists(p) else "⚠️  Missing (will skip)"
        print(f"  {status}: {lname}")

check_environment()

# ── Optional: upload your reference images ────────────────────────────────────
# @markdown ### 📸 Upload Reference Images
# @markdown Upload 1–5 reference images for your timeline segments.
# @markdown Filenames should match those in `timeline_segments` above (e.g. `ref_1.png`).
# @markdown Skip this cell if images are already in `/content/ComfyUI/input/`.

print("\n📸 To upload reference images, run:")
print('   uploaded = upload_image("/content/ComfyUI/input")')
print("   # Repeat for each reference image")
print("\nOr place them manually in /content/ComfyUI/input/ before continuing.")


# ═══════════════════════════════════════════════════════════════
# CELL 13 ─── Production Loop  (Director 2.0 multi-shot engine)
# ═══════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown ## ▶️ Step 6 — Generate!

input_dir  = "/content/ComfyUI/input"
output_dir = "/content/ComfyUI/output/LTX23"
cache_dir  = f"{output_dir}/{PROJECT_NAME}_cache"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)
cleanup_old_cache(cache_dir, CACHE_MAX_AGE_DAYS)

generated_clips: List[str] = []
start_index         = 0
current_input_image = None
prev_shot_success   = True

# Identity anchor: start from character sheet if available
if USE_CHARACTER_SHEETS and os.path.exists(CHARACTER_SHEET_PATH):
    current_input_image = CHARACTER_SHEET_PATH
    print(f"🎬 Identity anchor: {CHARACTER_SHEET_PATH}")

# Auto-resume from cached clips
for i in range(len(STORYBOARD)):
    anchor = f"{input_dir}/anchor_{i}.png"
    clip   = f"{cache_dir}/shot_{i+1:02d}.mp4"
    if os.path.exists(anchor) and os.path.exists(clip):
        current_input_image = anchor
        start_index = i + 1
        generated_clips.append(clip)
    else:
        break
if start_index > 0:
    print(f"⏩ Resuming from shot {start_index+1} ({start_index} cached)")

executor      = concurrent.futures.ThreadPoolExecutor(max_workers=1)
stitch_future = None

try:
    pbar = tqdm(
        range(start_index, len(STORYBOARD)),
        desc="🎬 Generating",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    for i in pbar:
        scene     = STORYBOARD[i]
        scene["_idx"] = i
        shot_data = scene["shot_data"]
        cleanup_memory()

        # Check if background stitch finished
        if PARALLEL_PROCESSING and stitch_future and stitch_future.done():
            try:
                res = stitch_future.result()
                if res: print(f"  🧵 Partial preview: {res}")
            except Exception as e:
                print(f"  ⚠️ Background stitch: {e}")

        # Adaptive anchor strength
        strength    = (calculate_adaptive_strength(shot_data, scene.get("prev_shot"),
                                                   prev_shot_success)
                       if i > 0 else 0.0)
        overlap_val = calculate_adaptive_overlap(shot_data)
        frames      = scene.get("seg_frames", 121) if not STORYBOARD_MODE else 1

        # Use the segment's dedicated reference image if available
        img_for_shot = scene.get("image_path") or current_input_image

        success = False
        for attempt in range(1, 4):
            pbar.set_description(f"🎬 Shot {i+1} — try {attempt}/3")
            seed = 2000 + (i * 100) + (attempt * 197)

            try:
                print(f"\n📍 Shot {i+1}/{len(STORYBOARD)} | attempt {attempt}"
                      f" | frames={frames} | overlap={overlap_val}f | seed={seed}")
                print_vram_usage()

                if GENERATE_SHOT_VARIATIONS and NUM_VARIATIONS > 1:
                    clip_path = generate_shot_with_variations(
                        scene, img_for_shot, strength, seed)
                else:
                    clip_path = generate_segment_director(
                        image_path=img_for_shot,
                        prompt=scene["prompt"],
                        seed=seed,
                        frames=frames,
                        image_strength=strength,
                        camera_lora=scene.get("camera_lora"),
                        ic_loras=scene.get("ic_loras", []),
                        shot_data=shot_data,
                        shot_index=i,
                    )

                if clip_path:
                    if FACE_RESTORATION:
                        clip_path = apply_face_restoration(clip_path)

                    anchor = extract_overlap_anchor(
                        clip_path, input_dir, i, overlap_val)
                    if anchor:
                        cached = f"{cache_dir}/shot_{i+1:02d}.mp4"
                        shutil.copy(clip_path, cached)
                        generated_clips.append(cached)
                        current_input_image = anchor
                        success = True
                        prev_shot_success = True
                        print(f"  ✅ Shot {i+1} complete! → {cached}")

                        # Trigger partial stitch in background
                        if PARALLEL_PROCESSING and len(generated_clips) > 1:
                            stitch_future = executor.submit(
                                stitch_videos_director,
                                list(generated_clips),
                                f"{PROJECT_NAME}_PARTIAL.mp4",
                                OVERLAP_FRAMES
                            )
                        break
                    else:
                        print("  ⚠️ Anchor extraction failed — retrying...")

            except Exception as e:
                print(f"  ❌ Shot {i+1} attempt {attempt}: {e}")
                traceback.print_exc()
                prev_shot_success = False
                cleanup_memory()

        if not success:
            print(f"⚠️ Shot {i+1} failed after 3 attempts — skipping, production continues.")
            prev_shot_success = False

except KeyboardInterrupt:
    print("\n⚠️ Interrupted — stitching available clips...")
except Exception as e:
    print(f"\n❌ Critical error: {e}")
    traceback.print_exc()
    print("🔄 Attempting recovery stitch of completed shots...")
finally:
    if MODEL_CACHE: MODEL_CACHE.evict()
    executor.shutdown(wait=False)


# ═══════════════════════════════════════════════════════════════
# CELL 14 ─── Final Stitch, Subtitles & Display
# ═══════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown ## 🎬 Step 7 — Final Output

if generated_clips:
    print(f"\n{'='*60}")
    print(f"🎬 LTX-2.3 Director PRO v4.0 — Final Stitch")
    print(f"   Shots completed : {len(generated_clips)}/{len(STORYBOARD)}")
    print(f"   Resolution      : {WIDTH}×{HEIGHT} @ {FPS}fps")
    print(f"   Quality mode    : {QUALITY_MODE}")
    print(f"   Director LoRAs  : {'4-stack active' if USE_DIRECTOR_LORA_STACK else 'OFF'}")
    print(f"{'='*60}\n")

    # Optional music track (from scene JSON audio config)
    music_path = None
    scene_audio_file = SCENE_JSON.get("audio", {}).get("music_file", "")
    if scene_audio_file:
        mp = os.path.join(input_dir, scene_audio_file)
        if os.path.exists(mp):
            music_path = mp
            print(f"🎵 Music track: {music_path}")

    try:
        final_movie = stitch_videos_director(
            generated_clips,
            output_filename=f"{PROJECT_NAME}_Director_PRO_v4.mp4",
            overlap_frames=OVERLAP_FRAMES,
            music_track=music_path,
        )

        # SRT subtitles
        if GENERATE_SUBTITLES:
            srt_path = f"{output_dir}/{PROJECT_NAME}_Director_PRO_v4.srt"
            generate_srt_from_dialogue(SCENE_JSON["dialogue_with_timing"], srt_path)

        if final_movie and os.path.exists(final_movie):
            size_mb = os.path.getsize(final_movie) / (1024 * 1024)
            dur_s   = len(generated_clips) * (OVERLAP_FRAMES / FPS + 4)

            print(f"\n{'='*60}")
            print(f"🎉  COMPLETE — LTX-2.3 Director 2.0 PRO v4.0")
            print(f"{'='*60}")
            print(f"  📁 File    : {final_movie}")
            print(f"  📐 Size    : {size_mb:.1f} MB")
            print(f"  ⏱️  ~Duration: {dur_s:.0f}s")
            print(f"  🎭 Shots   : {len(generated_clips)}/{len(STORYBOARD)}")
            print(f"  🔬 Model   : LTX-2.3 22B Q4_K_M")
            print(f"  🎞️  Encode  : {'GPU NVENC' if _detect_nvenc() and USE_GPU_ENCODING else 'CPU libx264'}")
            print(f"  🎵 Audio   : Native LTX-2.3 VAE + {'music mix' if music_path else 'no music'}")
            print(f"  📝 Subs    : {'Generated' if GENERATE_SUBTITLES else 'Skipped'}")
            print(f"{'='*60}\n")

            display_video(final_movie)

            # Optional Google Drive save
            # from google.colab import drive
            # drive.mount('/content/drive')
            # shutil.copy(final_movie, '/content/drive/MyDrive/LTX23_output/')

    except Exception as e:
        print(f"❌ Final stitch failed: {e}")
        traceback.print_exc()
        if generated_clips:
            print(f"\n💾 Individual clips saved in: {cache_dir}")
            for i, c in enumerate(generated_clips, 1):
                print(f"  [{i}] {c}")
else:
    print("❌ No clips generated. Check errors above.")
    print("\n💡 Troubleshooting:")
    print("  1. Ensure reference images exist in /content/ComfyUI/input/")
    print("  2. Check VRAM — LTX-2.3 22B needs ≥16GB (A100/V100/T4+)")
    print("  3. Try QUALITY_MODE='preview' to reduce memory usage")
    print("  4. Try STORYBOARD_MODE=True to generate single frames first")

print("\n✅ LTX-2.3 Director 2.0 PRO v4.0 — Done!")
