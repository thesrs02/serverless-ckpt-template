#!/usr/bin/env python

import os
import shutil
import sys

from transformers.utils.hub import move_cache

from controlnet_aux import HEDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker


from models import (
    loras_list,
    base_models,
    safety_model,
    controlnet_path,
    models_cache_dir,
    controlnet_model,
    single_file_base_models,
)

move_cache()
sys.path.append(".")


if os.path.exists(models_cache_dir):
    shutil.rmtree(models_cache_dir)
os.makedirs(models_cache_dir, exist_ok=True)

# Load to download automatically
hed = HEDdetector.from_pretrained(controlnet_path)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model)

controlnet = ControlNetModel.from_pretrained(
    controlnet_model,
    cache_dir=models_cache_dir,
)

for base_model in base_models:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        controlnet_model,
        controlnet=controlnet,
        cache_dir=models_cache_dir,
    )

for single_file_path in single_file_base_models:
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        single_file_path,
        controlnet=controlnet,
    )

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_models[0],
    controlnet=controlnet,
    cache_dir=models_cache_dir,
)

for lora_weight_name in loras_list:
    pipe = pipe.load_lora_weights("rehanhaider/sd-loras", weight_name=lora_weight_name)
