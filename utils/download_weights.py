#!/usr/bin/env python

import os
import shutil
import sys
from transformers.utils.hub import move_cache


from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)

move_cache()
sys.path.append(".")


from src.sdxl_runner import (
    VAE_ID,
    MODEL_ID,
    REFINER_ID,
    MODEL_CACHE_DIR,
    CONTROL_NET_MODE_ID,
)


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

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    REFINER_ID,
    variant="fp16",
    # torch_dtype=torch.float16,
)
