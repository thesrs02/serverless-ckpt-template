#!/usr/bin/env python

import os
import shutil
import sys
from transformers.utils.hub import move_cache


from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)

move_cache()
sys.path.append(".")


from src.sdxl_runner import MODEL_CACHE_DIR, VAE_ID, MODEL_ID, CONTROL_NET_MODE_ID


if os.path.exists(MODEL_CACHE_DIR):
    shutil.rmtree(MODEL_CACHE_DIR)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Load to download automatically
controlnet = ControlNetModel.from_pretrained(
    CONTROL_NET_MODE_ID,
    cache_dir=MODEL_CACHE_DIR,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    MODEL_ID,
    controlnet=controlnet,
    cache_dir=MODEL_CACHE_DIR,
)

vae = AutoencoderKL.from_pretrained(
    VAE_ID,
    cache_dir=MODEL_CACHE_DIR,
)
