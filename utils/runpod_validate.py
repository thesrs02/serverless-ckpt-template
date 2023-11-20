from runpod.serverless.utils.rp_validator import validate

INPUT_SCHEMA = {
    "prompt": {"type": str, "required": False, "default": ""},
    "negative_prompt": {"type": str, "default": "", "required": False},
    "image_url": {"type": str, "required": True},
    #
    "model_name": {"type": str, "required": True, "default": "openjourney"},
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
}


def validate_job_input(input):
    return validate(input, INPUT_SCHEMA)
