import os
from runpod.serverless.utils.rp_validator import validate

INPUT_SCHEMA = {
    "prompt": {"type": str, "required": True},
    "negative_prompt": {"type": str, "required": False, "default": None},
    "image_url": {"type": str, "required": True},
    "input_image_max_size": {
        "type": int,
        "required": False,
        "default": 1024,
        "constraints": lambda width: width in [768, 832, 896, 960, 1024],
    },
    "controlnet_conditioning_scale": {
        "type": float,
        "required": False,
        "default": 0.5,
        "constraints": lambda guidance_scale: 0.1 <= guidance_scale <= 1,
    },
    "output_width": {
        "type": int,
        "required": False,
        "default": 768,
        "constraints": lambda width: width in [768, 832, 896, 960, 1024],
    },
    "output_height": {
        "type": int,
        "required": False,
        "default": 1024,
        "constraints": lambda height: height in [768, 832, 896, 960, 1024],
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 14,
        "constraints": lambda num_inference_steps: num_inference_steps in range(1, 500),
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 7,
        "constraints": lambda guidance_scale: 0 <= guidance_scale <= 20,
    },
    "scheduler": {
        "type": str,
        "required": False,
        "default": "DPMSolverMultistep",
        "constraints": lambda scheduler: scheduler in ["DPMSolverMultistep"],
    },
    "seed": {
        "type": int,
        "required": False,
        "default": int.from_bytes(os.urandom(2), "big"),
    },
}


def validate_job_input(input):
    return validate(input, INPUT_SCHEMA)
