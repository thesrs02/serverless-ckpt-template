import os
from runpod.serverless.utils.rp_validator import validate

INPUT_SCHEMA = {
    "image_url": {"type": str, "required": True},
    #
    "prompt": {"type": str, "required": True},
    "prompt_2": {"type": str, "required": False, "default": None},
    #
    "negative_prompt": {"type": str, "required": False, "default": None},
    "negative_prompt_2": {"type": str, "required": False, "default": None},
    #
    "controlnet_conditioning_scale": {
        "type": float,
        "required": False,
        "default": 0.5,
        "constraints": lambda guidance_scale: 0.1 <= guidance_scale <= 1,
    },
    #
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 14,
        "constraints": lambda num_inference_steps: num_inference_steps in range(1, 500),
    },
    "refiner_num_inference_steps": {
        "type": int,
        "required": False,
        "default": 60,
        "constraints": lambda num_inference_steps: num_inference_steps in range(1, 500),
    },
    #
    "guidance_scale": {
        "type": float,
        "default": 7,
        "required": False,
        "constraints": lambda guidance_scale: 0 <= guidance_scale <= 20,
    },
    "refiner_guidance_scale": {
        "type": float,
        "required": False,
        "default": 7,
        "constraints": lambda guidance_scale: 0 <= guidance_scale <= 20,
    },
    #
    "canny_min_threshold": {
        "type": int,
        "default": 100,
        "required": False,
        "constraints": lambda num_inference_steps: num_inference_steps in range(1, 500),
    },
    "canny_max_threshold": {
        "type": int,
        "default": 300,
        "required": False,
        "constraints": lambda num_inference_steps: num_inference_steps in range(1, 500),
    },
    "resize_output_to": {
        "type": int,
        "default": 1024,
        "required": False,
        "constraints": lambda width: width in [256, 512, 768, 832, 896, 960, 1024],
    },
    "apply_input_image_enhancers": {"type": bool, "required": False, "default": True},
    # "scheduler": {
    #     "type": str,
    #     "required": False,
    #     "default": "DPMSolverMultistep",
    #     "constraints": lambda scheduler: scheduler in ["DPMSolverMultistep"],
    # },
    # "seed": {
    #     "type": int,
    #     "required": False,
    #     "default": int.from_bytes(os.urandom(2), "big"),
    # },
}


def validate_job_input(input):
    return validate(input, INPUT_SCHEMA)
