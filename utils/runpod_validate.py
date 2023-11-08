# utils/runpod_validate.py
import os
from runpod.serverless.utils.rp_validator import validate

INPUT_SCHEMA = {
    "prompt": {"type": str, "required": False, "default": ""},
    "image_url": {"type": str, "required": True},
    "negative_prompt": {"type": str, "required": False, "default": None},
    "lora_scale": {
        "type": float,
        "default": 0.9,
        "required": False,
        "constraints": lambda guidance_scale: 0.1 <= guidance_scale <= 1,
    },
    "guess_mode": {
        "type": bool,
        "required": False,
        "default": True,
    },
    #
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 14,
        "constraints": lambda num_inference_steps: num_inference_steps in range(1, 500),
    },
    "guidance_scale": {
        "type": float,
        "default": 3.5,
        "required": False,
        "constraints": lambda guidance_scale: 0 <= guidance_scale <= 20,
    },
    "controlnet_conditioning_scale": {
        "type": float,
        "required": False,
        "default": 0.9,
        "constraints": lambda guidance_scale: 0.1 <= guidance_scale <= 1,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": int.from_bytes(os.urandom(2), "big"),
    },
    "base_model_name": {"type": str, "default": None, "required": True},
    "model_file_url": {"type": str, "default": None, "required": False},
    "lora_weights_name": {"type": str, "default": None, "required": False},
}


def validate_job_input(input):
    return validate(input, INPUT_SCHEMA)
