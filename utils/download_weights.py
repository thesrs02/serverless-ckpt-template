#!/usr/bin/env python

import os
import shutil
import sys

from transformers.utils.hub import move_cache

from controlnet_aux import HEDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
)
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
print("HED Downloaded")
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model)

print("safety_checker Downloaded")

controlnet = ControlNetModel.from_pretrained(
    controlnet_model,
    cache_dir=models_cache_dir,
)

print("controlnet Downloaded")

for base_model in base_models:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        cache_dir=models_cache_dir,
    )

print("all base_models Downloaded")

for single_file_path in single_file_base_models:
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        single_file_path,
        controlnet=controlnet,
    )
    
print("all single_file_models Downloaded")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_models[0],
    controlnet=controlnet,
    cache_dir=models_cache_dir,
)

for lora_weight_name in loras_list:
    pipe.load_lora_weights("rehanhaider/sd-loras", weight_name=lora_weight_name, cache_dir=models_cache_dir)
    
print("all LORAS Downloaded")
